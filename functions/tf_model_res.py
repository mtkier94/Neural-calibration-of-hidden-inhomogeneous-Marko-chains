
import numpy as np
from numba import njit, prange
import time
from copy import deepcopy, copy

import tensorflow as tf
from tensorflow.keras.layers import Dense,GRU, Input, SimpleRNN
from tensorflow.keras.models import  Model, clone_model
from tensorflow.keras.regularizers import L1, L2

from functions.tf_loss_custom import compute_loss_mae



def create_mortality_res_net(hidden_layers = [40,40,20], param_l2_penalty = 0.1, input_shape=(None, 4), n_out=2):
    '''
    Create neural net (seq2seq-type) that predicts residual, 1-step transition probabilities. 
    Model architecture: Dense, .. , Dense, GRU, Dense-Output Layer
    Note: To smoothen the descrepancies we apply l2-regularization
    We assume  1-step-trans-prob(x) = baseline_mortality(x)+ mortality_res(x).

    Inputs:
    -------
        hidden_layers:  widths (and implicitely count) of hidden layers
        n_in:           number of inputs units
        n_out:          number of output units

    Outputs:
    --------
        model       tf.model.Model() architecture, no trained weights yet
    '''

    assert(len(hidden_layers)>2)

    INPUT = Input(shape=input_shape)
    h = Dense(units = hidden_layers[0], activation = 'relu', kernel_regularizer=L2(param_l2_penalty))(INPUT)
    for k in hidden_layers[1:-1]:
        h = Dense(units = k, activation = 'relu', kernel_regularizer=L2(param_l2_penalty))(h)
    h = GRU(units = hidden_layers[-1], activation = 'tanh', return_sequences = True, kernel_regularizer=L2(param_l2_penalty))(h)
    # create activation as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
    # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
    h = Dense(units = n_out, activation = 'linear', kernel_regularizer=L2(param_l2_penalty))(h)
    OUTPUT = tf.keras.layers.Activation('softmax')(h)

    model = Model(INPUT, OUTPUT)
    model.compile(loss = tf.keras.losses.KLDivergence(), metrics=['mae'], optimizer='adam') 

    return model



def combine_models(model_base, model_res, bool_masking):
    ''' 
    Create a model (seq2seq-type) which predicts one-step-transition probabilities.
    We assume  1-step-trans-prob(x) = baseline_mortality(x)+ mortality_res(x).
    Note:   The baseline will be fixed (non-trainable)
            To combine the two models we further assume that for both holds model.layer[-1] -> softmax-activation layer
                

    Inputs:
    -------
        model_base: baseline_mortality
        model_res:  residual mortality, e.g. for specific type of customer

    Outputs:
    --------
        model   tf.model.Model() which combines the two input model
    '''
    # work with copies of models to avoid unintended interaction (at the cost of memory)
    # NOTE: clone_model seems to run into layer-name-conflicts when saving the model ...
    # w_init = model_res.get_weights()
    # model_res = clone_model(model_res) # Note: layers are re-created and hence also weights reinitialized
    # model_res.set_weights(w_init)
    model_base.trainable = False
    

    if bool_masking:
        # insert masking-layer after input
        input_base, input_res = model_base.input, model_res.input
        input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(input_base), tf.keras.layers.Masking(mask_value=0.0)(input_res)

        model_tail = Model(inputs=[model_base.input, model_res.input], 
                        outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
        
        OUTPUT = model_tail([input_base_masked, input_res_masked])

        # add masking right after input
        # disadvantage: model summary contains functional tf-model -> layers cannot simply be iterated over
        return Model(inputs = [input_base, input_res], outputs= OUTPUT )
    else:
        return Model(inputs=[model_base.input, model_res.input], 
                    outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))


def combine_models_with_mask(model_base, model_res):
    '''
    Model evaluation up to. 50% faster when not inserting embedding layer! However, training seems to progress much faster with inserted masking layer
    Note: As we also zero-pad the target-values y (CFs), y naturally provide a mask since they are multiplied with the predictions in custom tf-loss function compute_loss_mae!!
    New version of original combine_base_and_res_model. Add masking of zero-padded input-sequence values to model.
    Note: slicing of input, e.g. via a lambda layer is impractical -> we stick with two separate input heads for model_base and model_res

    Inputs:
    -------
        model_base: baseline_mortality
        model_res:  residual mortality, e.g. for specific type of customer
        

    Outputs:
    --------
        model   tf.model.Model() which combines the two input model
    '''


 
    input_base, input_res = model_base.input, model_res.input
    input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(input_base), tf.keras.layers.Masking(mask_value=0.0)(input_res)

    model_tail = Model(inputs=[model_base.input, model_res.input], 
                    outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
    
    OUTPUT = model_tail([input_base_masked, input_res_masked])

    # add masking right after input
    # disadvantage: model summary contains functional tf-model -> layers cannot simply be iterated over
    return Model(inputs = [input_base, input_res], outputs= OUTPUT )




def random_pretraining(model_base, x, y, res_features = 4, iterations = 10, masking = False):
    '''
    Create the full model and (optionally) check random initializations for a sound initialization
    Note: We use the default architecture for create_mortalitiy_res_net.

    Inputs:
    -------
        model_base: baseline mortality model, pretrained tf-model
        x, y:   input- and target-data; used to evaluate goodness of initializations
        iterations: how many random initializations to look at

    Outputs:
    --------
        model: combined, compiled tf-model consisting of model_base and model_res
        w_init: initialized weights; used for later purpose of resetting weights during HPSearch

    '''

    model_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, res_features), n_out=2)
    if masking == False:
        model = combine_models(model_base = model_base, model_res = model_res)
    else: 
        model = combine_models_with_mask(model_base = model_base, model_res = model_res)
    model.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    loss_init, _ = model.evaluate(x,y, batch_size = 2**10, verbose = 0)
    print(loss_init)
    w_init = copy(model.get_weights())

    for _ in range(iterations-1):
        model_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, res_features), n_out=2)
        if masking == False:
            model = combine_models(model_base = model_base, model_res = model_res)
        else: 
            model = combine_models_with_mask(model_base = model_base, model_res = model_res)
        model.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
        loss, _ = model.evaluate(x,y, batch_size = 2**10, verbose = 0)
        print(loss)
        if loss < loss_init:
            w_init = deepcopy(model.get_weights())

    return model, w_init



def train_combined_model(pmodel, x, y, iterations_per_epoch, epochs, base_feat_in, res_feat_in):
    '''
    Helper function to modulize training.

    Inputs:
    -------
        pmodel: tf-model to be trained, combining baseline_mortality(x)+ mortality_res(x)
                where baseline_mortality(x) should be fixed
                Note: the two components might require different slices of x as input
        x:      List with one batch of contracts (of equal no. of iterations) per entry
        y:      List of (discounted) cash-flows for contracts in x
        iterations_per_epoch:   no. of iterations each list entry in x should be trained per epoch
        epochs: no. of epochs
        base_feat_in:   list of which columns of x are input to pmodel_base, i.e. x[:,:,base_feat_in]
        res_feat_in:   list of which columns of x are input to pmodel_res, i.e. x[:,:,res_feat_in]

    Outputs:
    --------
        pmodel (now trained)
        history list with history of training losses (np.array) as list elements
    '''

    N_batches = len(x)
    history = []

    for it in range(epochs):
        # store training history
        
        for k, (x_val, y_val) in enumerate(zip(x,y)):
            
            x_val_base, x_val_res = x_val[:,:,base_feat_in], x_val[:,:,res_feat_in]

            print('Training batch {}/{}, epoch {}.'.format(k+1, N_batches, it+1))
            tic = time.time()
            # Note: dynamic whole-batch training -> avoid overfitting locally as single (low-duration) contracts only access small manifold on feature space
            pmodel.fit(x=[x_val_base,x_val_res], y=y_val, epochs= iterations_per_epoch, verbose=0, batch_size= len(x_val_base))
            print('\t training complete after {} sec.'.format(np.round_(time.time()-tic,2)))
            # history of batch k
            history += pmodel.history.history['loss']

    return history


def train_combined_model_on_padded_data(pmodel, tf_data=None, x = None, y = None, epochs = 1, callbacks = None, batch_sz = 64, verbose = 0):
    '''
    Helper function to modulize training.

    Inputs:
    -------
        pmodel: tf-model to be trained, combining baseline_mortality(x)+ mortality_res(x)
                where baseline_mortality(x) should be fixed
                Note: the two components might require different slices of x as input
        tf_data: tf.Data.Dataset to be fed for trainng to pmodel
        x,y:    Training data, alternative to tf_data

    Outputs:
    --------
        pmodel (now trained)
        history list with history of training losses (np.array) as list elements
    '''

    history = [] # store training history

    tic = time.time()
    # Note: dynamic whole-batch training -> avoid overfitting locally as single (low-duration) contracts only access small manifold on feature space
    if type(tf_data)!= type(None):
        # note: batching handled by dataset
        pmodel.fit(tf_data, epochs= epochs, callbacks = callbacks, verbose=verbose)
    else:
        pmodel.fit(x,y,epochs=epochs, callbacks = callbacks, verbose=verbose, batch_size=batch_sz)
    print('\t training complete after {} sec.'.format(np.round_(time.time()-tic,2)))
    # history of batch k
    history += pmodel.history.history['loss']

    return history
   



# --------------------------------------------------------------------------------------------
# Legacy code


# def combine_models_with_masking(model_base, model_res):
#     ''' 
#     DEPRECIATED !!! MASKING-LAYER SHOWS UNEXPECTED BEHAVIOUR
#     Create a model (seq2seq-type) which predicts one-step-transition probabilities.
#     We assume  1-step-trans-prob(x) = baseline_mortality(x)+ mortality_res(x).
#     Note:   The baseline will be fixed (non-trainable)
#             To combine the two models we further assume that for both holds model.layer[-1] -> softmax-activation layer
                

#     Inputs:
#     -------
#         model_base: baseline_mortality
#         model_res:  residual mortality, e.g. for specific type of customer

#     Outputs:
#     --------
#         model   tf.model.Model() which combines the two input model
#     '''

#     assert 1==False, 'function currently not in use!'

#     # mbase = (model_base)
#     # mres = (model_res)


#     # mbase.trainable = False
#     # new_in_base, new_in_res = deepcopy(model_base.input), deepcopy(model_res.input)
#     # input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(new_in_base), tf.keras.layers.Masking(mask_value=0.0)(new_in_res)
#     # model_tail = Model(inputs=[mbase.input, mres.input], 
#     #                  outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([mbase.layers[-2].output, mres.layers[-2].output])))

#     # out = model_tail([input_base_masked, input_res_masked])

#     return Model(inputs = [tf.keras.layers.Masking(0.0)(model_base.input), tf.keras.layers.Masking(0.0)(model_res.input)], 
#                 outputs = tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
#     # return Model(inputs = [new_in_base, new_in_res], outputs = tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([mbase.layers[-2].output, mres.layers[-2].output])))

    
#     # input_base, input_res = model_base.input, model_res.input
#     # input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(input_base), tf.keras.layers.Masking(mask_value=0.0)(input_res)

#     # model_tail = Model(inputs=[model_base.input, model_res.input], 
#     #                 outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
    
#     # OUTPUT = model_tail([input_base_masked, input_res_masked])

#     # # add masking right after input
#     # # disadvantage: model summary contains functional tf-model -> layers cannot simply be iterated over
#     # model = Model(inputs = [input_base, input_res], outputs= OUTPUT )

#     # # add masking at the end
#     # # return Model(inputs=[model_base.input, model_res.input], 
#     # #                 outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()(
#     # #                     [tf.keras.layers.Masking(mask_value=0.0)(model_base.layers[-2].output), 
#     # #                     tf.keras.layers.Masking(mask_value=0.0)(model_res.layers[-2].output)])))
#     # return model

#---------------------------------------------------------------------------------------------------------------------------

# def combine_models(model_base, model_res):
#     '''
#     DEPRECIATED! Masking layer shows no speed-up. In fact, model evaluation ca. 50% faster when not inserting embedding layer!
#         -> Note: As we also zero-pad the target-values y (CFs), y naturally provide a mask since they are multiplied with the predictions in custom tf-loss function compute_loss_mae!!
#     New version of original combine_base_and_res_model. Add masking of zero-padded input-sequence values to model.
#     Note: slicing of input, e.g. via a lambda layer is impractical -> we stick with two separate input heads for model_base and model_res

#     Inputs:
#     -------
#         model_base: baseline_mortality
#         model_res:  residual mortality, e.g. for specific type of customer
        

#     Outputs:
#     --------
#         model   tf.model.Model() which combines the two input model
#     '''

#     assert True, 'Function depreciated, read documentation for more information.'

#     #INPUT = tf.keras.layers.Input(input_shape)
    
#     input_base, input_res = model_base.input, model_res.input
#     # assert(len(id_feat_base)==input_base.shape[-1])
#     # assert(len(id_feat_res)==input_res.shape[-1])
#     # # explicitely set no. of timesteps for recurrent network
#     # #input_base = Input(shape=(input_res.get_shape()[1], len(id_feat_base)))
#     input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(input_base), tf.keras.layers.Masking(mask_value=0.0)(input_res)

#     # #input_base = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Lambda(lambda x: x[:,:,feat])(input_masked) for feat in id_feat_base]) 
#     # #input_res = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Lambda(lambda x: x[:,:,feat])(input_masked) for feat in id_feat_res]) 

#     model_tail = Model(inputs=[model_base.input, model_res.input], 
#                     outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
    
#     OUTPUT = model_tail([input_base_masked, input_res_masked])

#     # add masking right after input
#     # disadvantage: model summary contains functional tf-model -> layers cannot simply be iterated over
#     model = Model(inputs = [input_base, input_res], outputs= OUTPUT )

#     # add masking at the end
#     # return Model(inputs=[model_base.input, model_res.input], 
#     #                 outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()(
#     #                     [tf.keras.layers.Masking(mask_value=0.0)(model_base.layers[-2].output), 
#     #                     tf.keras.layers.Masking(mask_value=0.0)(model_res.layers[-2].output)])))
#     return model