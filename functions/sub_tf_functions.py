
# all functions have been moved in more granular modules, e.g. tf_model_res.py

# import numpy as np
# from numba import njit, prange
# import time
# import matplotlib.pyplot as plt


# from mortality import T_MAX
# import tensorflow as tf
# from tensorflow.keras.layers import Dense,GRU, Input, SimpleRNN
# from tensorflow.keras.models import  Model
# from tensorflow.keras.regularizers import L1, L2



# @tf.function#(experimental_relax_shapes = True)
# def compute_loss_test(y_true, y_pred):
#     '''
#     test functionality of custom syntax for tf-loss-functions.
#     '''

#     return tf.reduce_mean(y_true*y_pred)



# # address retracing as different # of steps will results in different shapes and therefore different tf Graphs
# @tf.function(experimental_relax_shapes = True) 
# def compute_loss_raw(y_true, y_pred):
#     '''
#     Custom loss for training of tensorflow-model.
#     Note:   new version which addresses issue of overestimating 1/m_q_x
#             before: CF_t accidentally weighted by (1/m+t)_p_x and (1/m+t)_q_m
#             now: CF_x weighted by (1/m+t)_p_x and (t-1/m)_p_x*1/m_q_(x+t-1/m) 

#     Inputs:
#     -------
#         y_true:     target values, i.e. cash-flows in time-series format for respective contracts
#                     shape: (None, steps, 2), where 2 equals the number of states, i.e. alive, dead
#         y_pred:     predictions, i.e. survival probs as a time-series of equal length than y_true for respective contracts
#                     shape: (None, steps, 2)

#     Outputs:
#     --------
#         loss value: tensorflow scalar, corresponding to the mean (per provided contracts) expected value of cash flows, weighted by surv.-probs
    
#     Detail on implementation:
#         step 1: compute cumulative product of the 1-step survival probs in y_pred, which represents the probability of surviving up to a specific perid.
#         step 2: combine cum. prod. with the 1-step transition probabilities (given the ph survived up to the respective period)
    
#     '''
    
#     # form cumulative survival probabilities to weight paths (of CFs)
#     # cumprod along axis of steps, i.e. axis = 1
#     # Note: y_pred.shape: (N_batch, N_eff, 2)
#     cum_probs = tf.math.cumprod(y_pred[:,:,0:1], axis = 1)
#     # broadcast and drop last entry/ last time-step
#     cum_probs = tf.broadcast_to(cum_probs[:,0:-1], shape=tf.shape(y_true[:,0:-1,:]))
#     ones = tf.ones(tf.shape(y_true[:,0:1,:]))
#     cum_prob_concat = tf.concat([ones, cum_probs], axis=1)
#     # prob of reaching time t and subsequently transitioning to states described in y_pred
#     prob_eff = cum_prob_concat*y_pred
#     # CFs at given times weighted by reaching time and transitioning in the respective state
#     # Note: discounting factor a-priori included in CFs, i.e. y_true
#     values = prob_eff*y_true

#     if False:
#         # check shapes for debugging
#         # Note: for better debugging remove @tf.function decorator to disable graph mode and enable eager performance
#         print('shapes y_true, y_pred: ', tf.shape(y_true), tf.shape(y_pred))        
#         print('shapes y_true, y_pred: ', y_true.shape, y_pred.shape)   
#         print('cum_probs.shape', tf.shape(cum_probs), ' numpy.shape: ', cum_probs.shape)
#         print('ones shape:', tf.shape(ones), ones.shape)
#         print('cum_prob_concat.shape', tf.shape(cum_prob_concat), 'numpy.shape: ', cum_prob_concat.shape)
#         print('prob_eff: ', tf.shape(prob_eff), prob_eff.shape)
#         print('shape of values ', tf.shape(values), ' numpy.shapes: ', values.shape)

#     return tf.reduce_mean((tf.reduce_sum(values, axis = [1,2])), axis = 0)

# @tf.function(experimental_relax_shapes = True) 
# def compute_loss_mae(y_true, y_pred):
#     '''
#     Custom loss for training of tensorflow-model. 

#     Inputs:
#     -------
#         y_true:     target values, i.e. discounted cash-flows in time-series format for respective contracts
#                     shape: (None, steps, 2), where 2 equals the number of states, i.e. alive, dead
#         y_pred:     predictions, i.e. survival probs as a time-series of equal length than y_true for respective contracts
#                     shape: (None, steps, 2)

#     Outputs:
#     --------
#         loss value: tensorflow scalar, corresponding to the mean (per provided contracts) absolute value of cash flows, weighted by predicted surv.-probs
    
#     Detail on implementation:
#         step 1: compute cumulative product of the 1-step survival probs in y_pred, which represents the probability of surviving up to a specific perid.
#         step 2: combine cum. prod. with the 1-step transition probabilities (given the ph survived up to the respective period)
    
#     '''
    
#     # form cumulative survival probabilities to weight paths (of CFs)
#     # cumprod along axis of steps, i.e. axis = 1
#     # Note: y_pred.shape: (N_batch, N_eff, 2)
#     cum_probs = tf.math.cumprod(y_pred[:,:,0:1], axis = 1)
#     # broadcast and drop last entry/ last time-step
#     cum_probs = tf.broadcast_to(cum_probs[:,0:-1], shape=tf.shape(y_true[:,0:-1,:]))
#     ones = tf.ones(tf.shape(y_true[:,0:1,:]))
#     cum_prob_concat = tf.concat([ones, cum_probs], axis=1)
#     # prob of reaching time t and subsequently transitioning to states described in y_pred
#     prob_eff = cum_prob_concat*y_pred
#     # CFs at given times weighted by reaching time and transitioning in the respective state
#     # Note: discounting factor a-priori included in CFs, i.e. y_true
#     values = prob_eff*y_true

#     return tf.reduce_mean(tf.math.abs((tf.reduce_sum(values, axis = [1,2]))), axis = 0)


# def create_baseline_model_ffn(n_in = 2, n_out = 2, h_units = [40,40,20], h_actv = ['relu', 'relu', 'tanh'], tf_loss_function = tf.keras.losses.KLDivergence(), optimizer = 'adam'):
#     '''
#     Create a baseline for n_in transition probabilities.
#     Note: This is a classical feed-forward type model.

#     Inputs:
#     -------
#         n_in:   no. of (scaled) input features to the model
#         n_out:  no. of (unit-sum) outputs, i.e. transition probabilities
#         width_factor:   factor for how wide hidden layers will be chosen, given n_in

#     Outputs:
#     --------
#         tf.keras.models.Model   feed-forward model


#     Note: The introduction of more than 2 output probabilities, i.e. by considering states alive, dead and disabled (and their transition probs) 
#     might invoke the need for multiple (unit-sum) output-layers since the respective transition-matrix has unit-sum in its columns.
#     '''

#     assert(len(h_units)==len(h_actv))
#     assert(len(h_units)>0)

#     INPUT = Input(shape=(n_in,))
#     h = Dense(units = h_units[0], activation = h_actv[0])(INPUT)
#     for u_k, f_k in zip(h_units[1:], h_actv[1:]):
#         if u_k>0:
#             h = Dense(units = u_k, activation = f_k)(h)
#     # create activation of output as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
#     # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
#     # output: 1-step trans.-probs: (1/m)_p_age, (1/m)_p_age 
#     #OUTPUT = Dense(units = n_out, activation= 'softmax')(h)
#     h = Dense(units = 2, activation= 'linear')(h)
#     OUTPUT = tf.keras.layers.Activation('softmax')(h)

#     model = Model(INPUT, OUTPUT)
#     model.compile(loss = tf_loss_function, metrics=['mae', 'mape'], optimizer=optimizer)

#     return model

# def create_baseline_model_rnn(input_shape = (None, 2), n_out = 2, hidden_layers = [40,40, 20], mask = False):
#     '''
#     Create a baseline for n_in transition probabilities.
#     Note: This is a sequence-to-sequence type model, given the time-series-character of our modeling objective.

#     Inputs:
#     -------
#         n_in:   no. of (scaled) input features to the model
#         n_out:  no. of (unit-sum) outputs, i.e. transition probabilities
#         hidden_layers:  widths (and implicitely count) of hidden layers

#     Outputs:
#     --------
#         tf.keras.models.Model   sequence-to-sequence model


#     Note: The introduction of more than 2 output probabilities, i.e. by considering states alive, dead and disabled (and their transition probs) 
#     might invoke the need for multiple (unit-sum) output-layers since the respective transition-matrix has unit-sum in its columns.
#     '''
    
#     ## Note: same activation-function as in respective non-rnn-baseline-model have to be used
#     # -> improve code-design

#     INPUT = Input(shape= input_shape, name='rnn_input')
#     h = Dense(units = hidden_layers[0], activation = 'relu', name = 'layer_0')(INPUT)
#     for num, l in enumerate(hidden_layers[1:-1]):
#         h = Dense(units = l, activation = 'relu', name = 'layer_{}'.format(num+1))(h)
#     h = SimpleRNN(units = hidden_layers[-1], activation = 'tanh', return_sequences = True, name = 'layer_{}'.format(len(hidden_layers)+1))(h)
#     # h = Dense(units = 2*width_factor, activation = 'relu')(INPUT)
#     # h = Dense(units = 2*width_factor, activation = 'relu')(h)
#     # h = SimpleRNN(units = width_factor, activation = 'tanh', return_sequences = True)(h)
#     # create activation as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
#     # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
#     h = Dense(units = n_out, activation = 'linear', name = 'layer_{}'.format(len(hidden_layers)+2))(h)
#     OUTPUT = tf.keras.layers.Activation('softmax')(h)

#     model = Model(INPUT, OUTPUT)
#     # weights will be transfered from feed-forward reference model; no training required
#     model.trainable = False 
#     model.compile(loss = tf.keras.losses.KLDivergence(), metrics=['mae'], optimizer='adam') 

#     return model

# def transfer_weights_dense2simpleRNN(dense_model, rnn_model):
#     '''
#     Transfer weights from a simple, pre-trained dense tf-model to a rnn tf-model with the same layers execpt a SimpleRNN (instead of Dense-layer) as its last hidden layer.
#     Note:   SimpleRNN-layer (in comparison to Dense-layer) introduces solely one additional matrix for memory processing.
#             We disable the recurrent memory processing in the baseline model by setting weights equal to zero.

#     Inputs:
#     -------
#         dense_model:    classical feed-forward tf-model, pretrained on 1-step transition probabilities
#         rnn_model:      tf-model, very similar to dense_model (see description above), but with a single SimpleRNN-layer

#     Outputs:
#     --------
#         rnn_model       now with trained weights
#     '''

#     assert(len(dense_model.layers)==len(rnn_model.layers))

#     for l_new, l_trained in zip(rnn_model.layers, dense_model.layers):
#         # only second-last hidden layer different between ffw and rnn
#         # Note: last hidden layer is the explicit activation layer
#         if l_new != rnn_model.layers[-3]:
#             print('l_new: ', l_new)
#             l_new.set_weights(l_trained.get_weights())
#         else:
#             # SimpleRNN with additional memory-related matrix at position 1
#             weights = l_trained.get_weights() # Dense layer: [weights, biases]
#             # TO DO: SimpleRNN layer: [weights, memory-weights (inactive), biases]
#             weights.insert(1, np.zeros((l_new.get_weights()[1].shape)))
#             l_new.set_weights(weights)

#     return rnn_model # no return required; weights are updated in-place




# def create_mortality_res_net(hidden_layers = [40,40,20], param_l2_penalty = 0.1, input_shape=(None, 4), n_out=2):
#     '''
#     Create neural net (seq2seq-type) that predicts residual, 1-step transition probabilities. 
#     Model architecture: Dense, .. , Dense, GRU, Dense-Output Layer
#     Note: To smoothen the descrepancies we apply l2-regularization
#     We assume  1-step-trans-prob(x) = baseline_mortality(x)+ mortality_res(x).

#     Inputs:
#     -------
#         hidden_layers:  widths (and implicitely count) of hidden layers
#         n_in:           number of inputs units
#         n_out:          number of output units

#     Outputs:
#     --------
#         model       tf.model.Model() architecture, no trained weights yet
#     '''

#     assert(len(hidden_layers)>2)

#     INPUT = Input(shape=input_shape)
#     h = Dense(units = hidden_layers[0], activation = 'relu', kernel_regularizer=L2(param_l2_penalty))(INPUT)
#     for k in hidden_layers[1:-1]:
#         h = Dense(units = k, activation = 'relu', kernel_regularizer=L2(param_l2_penalty))(h)
#     h = GRU(units = hidden_layers[-1], activation = 'tanh', return_sequences = True, kernel_regularizer=L2(param_l2_penalty))(h)
#     # create activation as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
#     # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
#     h = Dense(units = n_out, activation = 'linear', kernel_regularizer=L2(param_l2_penalty))(h)
#     OUTPUT = tf.keras.layers.Activation('softmax')(h)

#     model = Model(INPUT, OUTPUT)
#     model.compile(loss = tf.keras.losses.KLDivergence(), metrics=['mae'], optimizer='adam') 

#     return model

# def combine_models(model_base, model_res, id_feat_base, id_feat_res, input_shape):
#     '''
#     Advanced version of original combine_base_and_res_model.
#     New:    use slicing to create the two (hidden) inputs to the base and res model from a single input layer
#             include a masking layer to work with equal-length, padded sequences

#     Inputs:
#     -------
#         model_base: baseline_mortality
#         model_res:  residual mortality, e.g. for specific type of customer
#         input_shape:    shape of input layer including all features, i.e. (None, timesteps, features)
#         id_feat_base:   indices (for slicing), which input features of the full list of features is included in the base model
#         id_feat_res:    indices (for slicing), to include in the res model          

#     Outputs:
#     --------
#         model   tf.model.Model() which combines the two input model
#     '''

#     #INPUT = tf.keras.layers.Input(input_shape)
    
#     input_base, input_res = model_base.input, model_res.input
#     assert(len(id_feat_base)==input_base.shape[-1])
#     assert(len(id_feat_res)==input_res.shape[-1])
#     # explicitely set no. of timesteps for recurrent network
#     input_base = Input(shape=(input_res.get_shape()[1], len(id_feat_base)))
#     input_base_masked, input_res_masked = tf.keras.layers.Masking(mask_value=0.0)(input_base), tf.keras.layers.Masking(mask_value=0.0)(input_res)

#     #input_base = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Lambda(lambda x: x[:,:,feat])(input_masked) for feat in id_feat_base]) 
#     #input_res = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Lambda(lambda x: x[:,:,feat])(input_masked) for feat in id_feat_res]) 

#     model_tail = Model(inputs=[model_base.input, model_res.input], 
#                     outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
    
#     OUTPUT = model_tail([input_base_masked, input_res_masked])

#     model = Model(inputs = [input_base, input_res], outputs= OUTPUT )
#     return model



# def combine_base_and_res_model(model_base, model_res):
#     ''' 
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

#     model_base.trainable = False

#     input_base, input_res = model_base.input, model_res.input
#     #print(input_base)
#     #print(input_res)
#     #print(model_base.layers[-2](input_base))
#     #print(model_res.layers[-2](input_res))
#     model = Model(inputs=[input_base, input_res], 
#                     outputs= tf.keras.layers.Activation('softmax')(tf.keras.layers.Add()([model_base.layers[-2].output, model_res.layers[-2].output])))
#                     #[model_base.layers[-2](input_base), model_res.layers[-2](input_res)])))
#     return model


# def train_combined_model(pmodel, x, y, iterations_per_epoch, epochs, base_feat_in, res_feat_in):
#     '''
#     Helper function to modulize training.

#     Inputs:
#     -------
#         pmodel: tf-model to be trained, combining baseline_mortality(x)+ mortality_res(x)
#                 where baseline_mortality(x) should be fixed
#                 Note: the two components might require different slices of x as input
#         x:      List with one batch of contracts (of equal no. of iterations) per entry
#         y:      List of (discounted) cash-flows for contracts in x
#         iterations_per_epoch:   no. of iterations each list entry in x should be trained per epoch
#         epochs: no. of epochs
#         base_feat_in:   list of which columns of x are input to pmodel_base, i.e. x[:,:,base_feat_in]
#         res_feat_in:   list of which columns of x are input to pmodel_res, i.e. x[:,:,res_feat_in]

#     Outputs:
#     --------
#         pmodel (now trained)
#         history list with history of training losses (np.array) as list elements
#     '''

#     N_batches = len(x)
#     history = []

#     for it in range(epochs):
#         # store training history
        
#         for k, (x_val, y_val) in enumerate(zip(x,y)):
            
#             x_val_base, x_val_res = x_val[:,:,base_feat_in], x_val[:,:,res_feat_in]

#             print('Training batch {}/{}, epoch {}.'.format(k+1, N_batches, it+1))
#             tic = time.time()
#             # Note: dynamic whole-batch training -> avoid overfitting locally as single (low-duration) contracts only access small manifold on feature space
#             pmodel.fit(x=[x_val_base,x_val_res], y=y_val, epochs= iterations_per_epoch, verbose=0, batch_size= len(x_val_base))
#             print('\t training complete after {} sec.'.format(np.round_(time.time()-tic,2)))
#             # history of batch k
#             history += pmodel.history.history['loss']

#     return history


# def train_combined_model_on_padded_data(pmodel, tf_data=None, x = None, y = None, epochs = 1, callbacks = None, batch_sz = 64, verbose = 0):
#     '''
#     Helper function to modulize training.

#     Inputs:
#     -------
#         pmodel: tf-model to be trained, combining baseline_mortality(x)+ mortality_res(x)
#                 where baseline_mortality(x) should be fixed
#                 Note: the two components might require different slices of x as input
#         tf_data: tf.Data.Dataset to be fed for trainng to pmodel
#         x,y:    Training data, alternative to tf_data

#     Outputs:
#     --------
#         pmodel (now trained)
#         history list with history of training losses (np.array) as list elements
#     '''

#     history = [] # store training history

#     tic = time.time()
#     # Note: dynamic whole-batch training -> avoid overfitting locally as single (low-duration) contracts only access small manifold on feature space
#     if type(tf_data)!= type(None):
#         # note: batching handled by dataset
#         pmodel.fit(tf_data, epochs= epochs, callbacks = callbacks, verbose=verbose)
#     else:
#         pmodel.fit(x,y,epochs=epochs, callbacks = callbacks, verbose=verbose, batch_size=batch_sz)
#     print('\t training complete after {} sec.'.format(np.round_(time.time()-tic,2)))
#     # history of batch k
#     history += pmodel.history.history['loss']

#     return history
   

