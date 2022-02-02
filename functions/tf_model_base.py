
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import  Model, load_model


def create_baseline_model_ffn(n_in = 2, n_out = 2, h_units = [40,40,20], h_actv = ['relu', 'relu', 'tanh'], tf_loss_function = tf.keras.losses.KLDivergence(), optimizer = 'adam'):
    '''
    Create a baseline for n_in transition probabilities.
    Note: This is a classical feed-forward type model.

    Inputs:
    -------
        n_in:   no. of (scaled) input features to the model
        n_out:  no. of (unit-sum) outputs, i.e. transition probabilities
        width_factor:   factor for how wide hidden layers will be chosen, given n_in

    Outputs:
    --------
        tf.keras.models.Model   feed-forward model


    Note: The introduction of more than 2 output probabilities, i.e. by considering states alive, dead and disabled (and their transition probs) 
    might invoke the need for multiple (unit-sum) output-layers since the respective transition-matrix has unit-sum in its columns.
    '''

    assert(len(h_units)==len(h_actv))
    assert(len(h_units)>0)

    INPUT = Input(shape=(n_in,))
    h = Dense(units = h_units[0], activation = h_actv[0])(INPUT)
    for u_k, f_k in zip(h_units[1:], h_actv[1:]):
        if u_k>0:
            h = Dense(units = u_k, activation = f_k)(h)
    # create activation of output as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
    # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
    # output: 1-step trans.-probs: (1/m)_p_age, (1/m)_p_age 
    #OUTPUT = Dense(units = n_out, activation= 'softmax')(h)
    h = Dense(units = 2, activation= 'linear')(h)
    OUTPUT = tf.keras.layers.Activation('softmax')(h)

    model = Model(INPUT, OUTPUT)
    model.compile(loss = tf_loss_function, metrics=['mae', 'mape'], optimizer=optimizer)

    return model


def create_train_save_model_base(X, Y, h_units, learning_rate, epochs, batch_sz, path_save, bool_train, act_func = ['relu','relu', 'relu'], 
                                loss_function = tf.keras.losses.KLDivergence(),
                                callbacks = None, n_in = 2, n_out = 2, verbose = 0):
    '''
    Create baseline-mortality-model, train it and save it (including its history) for later comparison of its hyperparameters.

    Inputs:
    -------
        X, Y:       numpy arrays with training data
        tf_data:    tf.data.Dataset object; alternative to providing X, Y and will overwright X, Y
        h_units:    list with hidden units per (hidden) layer
        learning_rate:  learning rate used with AdamOptimizer
        epochs:     number of epochs for training
        path_save:  path for saving (and optionally loading -> see bool_train) models
        bool_train: boolean whether models should be loaded (if available) or trained regardless
        n_in:       number of input parameters
        n_out:      number of output parameters

    Outputs:
    --------
        model:  tf-Model with trained weights; Note: if weights have been loaded the attribute model.history.history will be empty
        history:    dictionary of training history of model, potentially loaded from path_save
    '''

    # create model
    model = create_baseline_model_ffn(n_in=n_in, n_out=n_out, h_units=h_units, h_actv= act_func, tf_loss_function = loss_function, 
                                    optimizer=tf.keras.optimizers.Adam(lr=learning_rate))

    if (bool_train==False) and (type(path_save)==type(None)):
        raise ValueError('Model cannot be loaded or trained!')

    if type(loss_function) == type(tf.keras.losses.KLDivergence()):
        tag = 'KLDiv'
    elif loss_function == 'mae':
        tag = 'mae'
    else:
        raise ValueError('Unknown loss_function!')

    print('Training with h_units: {}, batch_sz: {}, learning rate: {} for {} epochs with {} loss.'.format(h_units, batch_sz,learning_rate, epochs, tag))
    if (bool_train == False) and os.path.exists(os.path.join(path_save, r'model_widths_{}_bz_{}_lr_{}_loss_{}.h5'.format(h_units, batch_sz,learning_rate, tag))):
        # load model & hist
        model = load_model(os.path.join(path_save, r'model_widths_{}_bz_{}_lr_{}_loss_{}.h5'.format(h_units, batch_sz,learning_rate, tag)))
        history = np.load(os.path.join(path_save, r'hist_widths_{}_bz_{}_lr_{}_loss_{}.npy'.format(h_units, batch_sz,learning_rate, tag)), allow_pickle= True)#.item()
        print('\t ... pretrained model loaded.')
    else:
        print( '\t ... starting training')
        tic = time.time()
        model.fit(X, Y, batch_size=batch_sz, epochs=epochs, verbose=verbose, callbacks=callbacks)
        history = np.stack([np.array(model.history.history['loss']), np.array(model.history.history['mae']), np.array(model.history.history['mape'])], axis = 0)
        if type(path_save)!= type(None):
            model.save(os.path.join(path_save, r'model_widths_{}_bz_{}_lr_{}_loss_{}.h5'.format(h_units, batch_sz,learning_rate, tag)))
            np.save(os.path.join(path_save, r'hist_widths_{}_bz_{}_lr_{}_loss_{}.npy'.format(h_units, batch_sz,learning_rate, tag)), history)
        print(' \t ... completed after {} sec.'.format(np.round_(time.time()-tic,2)))

    return model, history



def create_baseline_model_rnn(input_shape = (None, 2), n_out = 2, hidden_layers = [40,40, 20], mask = False):
    '''
    Create a baseline for n_in transition probabilities.
    Note: This is a sequence-to-sequence type model, given the time-series-character of our modeling objective.

    Inputs:
    -------
        n_in:   no. of (scaled) input features to the model
        n_out:  no. of (unit-sum) outputs, i.e. transition probabilities
        hidden_layers:  widths (and implicitely count) of hidden layers

    Outputs:
    --------
        tf.keras.models.Model   sequence-to-sequence model


    Note: The introduction of more than 2 output probabilities, i.e. by considering states alive, dead and disabled (and their transition probs) 
    might invoke the need for multiple (unit-sum) output-layers since the respective transition-matrix has unit-sum in its columns.
    '''
    
    ## Note: same activation-function as in respective non-rnn-baseline-model have to be used
    # -> improve code-design

    INPUT = Input(shape= input_shape, name='rnn_input')
    h = Dense(units = hidden_layers[0], activation = 'relu', name = 'layer_0')(INPUT)
    for num, l in enumerate(hidden_layers[1:-1]):
        h = Dense(units = l, activation = 'relu', name = 'layer_{}'.format(num+1))(h)
    h = SimpleRNN(units = hidden_layers[-1], activation = 'tanh', return_sequences = True, name = 'layer_{}'.format(len(hidden_layers)+1))(h)

    # create activation as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
    # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
    h = Dense(units = n_out, activation = 'linear', name = 'layer_{}'.format(len(hidden_layers)+2))(h)
    OUTPUT = tf.keras.layers.Activation('softmax')(h)

    model = Model(INPUT, OUTPUT)
    # weights will be transfered from feed-forward reference model; no training required
    model.trainable = False 
    model.compile(loss = tf.keras.losses.KLDivergence(), metrics=['mae'], optimizer='adam') 

    return model

def transfer_weights_dense2simpleRNN(dense_model, rnn_model):
    '''
    Transfer weights from a simple, pre-trained dense tf-model to a rnn tf-model with the same layers execpt a SimpleRNN (instead of Dense-layer) as its last hidden layer.
    Note:   SimpleRNN-layer (in comparison to Dense-layer) introduces solely one additional matrix for memory processing.
            We disable the recurrent memory processing in the baseline model by setting weights equal to zero.

    Inputs:
    -------
        dense_model:    classical feed-forward tf-model, pretrained on 1-step transition probabilities
        rnn_model:      tf-model, very similar to dense_model (see description above), but with a single SimpleRNN-layer

    Outputs:
    --------
        rnn_model       now with trained weights
    '''

    assert(len(dense_model.layers)==len(rnn_model.layers))

    for l_new, l_trained in zip(rnn_model.layers, dense_model.layers):
        # only second-last hidden layer different between ffw and rnn
        # Note: last hidden layer is the explicit activation layer
        if l_new != rnn_model.layers[-3]:
            l_new.set_weights(l_trained.get_weights())
        else:
            # SimpleRNN with additional memory-related matrix at position 1
            weights = l_trained.get_weights() # Dense layer: [weights, biases]
            # TO DO: SimpleRNN layer: [weights, memory-weights (inactive), biases]
            weights.insert(1, np.zeros((l_new.get_weights()[1].shape)))
            l_new.set_weights(weights)

    #return rnn_model # no return required; weights are updated in-place
