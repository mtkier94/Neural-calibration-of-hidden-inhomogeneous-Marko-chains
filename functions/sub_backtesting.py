import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt

#import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.core import Masking

from functions.sub_actuarial import get_CFs_vectorized
from functions.tf_model_res import create_mortality_res_net, combine_models
from functions.tf_loss_custom import compute_loss_mae


def check_exploded_gradients(model):
    '''
    Check if there are nan-values in the weight-parameters of a given tf-model. NaN values are most likely the cause of exploded gradients during training.

    Inputs:
    -------
        model: tf model

    Outputs:
    --------
        boolean: True (NaN values present) or False
    '''

    for l in  model.layers:
        # multiple weight and/or bias parameters per layer
        if type(l) == type(Input) or type(l) == Masking or type(l) == InputLayer:
            pass
        else:
            for p in l.get_weights():
                if np.isnan(p).any() == True:
                    return True
                else:
                    pass
    return False


def check_model_mask_vs_no_mask(x_base, x_res, y, model_base, iterations = 3, len_res_input = 4):
    '''
    Check whether including a masking layer of zero-padded input-sequence data x has an effect on performance or speed.
    Expected:   Loss values should equal (as zero-padded y provides a natural mask, since we multiply it with the predictions in the custom tf-loss compute_loss_mae)
                masking should provide some speed-up, as it indicates which time-steps can be skipped in evaluation

    Inputs:
    -------
        x:  input data as to be passed to the combined model, i.e. [x[:,:,base_features], x[:,:,res_features]]
        y:  target values
        model_base: pretrained, baseline-tf-model to be included in larger model

    Outputs:
    --------
        None; loss values and times will be printed
    '''

    for i in range(iterations):

        model_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, len_res_input), n_out=2)
        #model_mask = combine_models_with_masking(model_base = model_base, model_res = model_res)
        #model_mask.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

        model_nomask = combine_models(model_base = model_base, model_res = model_res)
        model_nomask.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

        if i == 0:
            #model_mask.summary()
            model_nomask.summary()

        pred_mask = model_nomask.predict([Masking(0.0)(x_base), Masking(0.0)(x_res)])
        pred_nomask = model_nomask.predict([x_base, x_res])
        #plt.plot((pred_mask-pred_nomask).flatten())
        #plt.show()
        assert(np.allclose(pred_mask, pred_nomask))

        tic = time.time()
        model_nomask.evaluate(x = [Masking(0.0)(x_base), Masking(0.0)(x_res)], y= y, batch_size = 1024)
        print('\t time with masking: ', time.time()-tic)
        print('\t tf_computation: ', compute_loss_mae(y_true = y, y_pred = pred_mask))


        #pmodel_test.summary()
        tic = time.time()
        model_nomask.evaluate(x = [x_base, x_res], y= y, batch_size = 1024)
        print('\t time wo masking: ', time.time()-tic)
        print('\t tf_computation: ', compute_loss_mae(y_true = y, y_pred = pred_nomask))


def check_padding(model, x_nopad, y_nopad, base_feat, res_feat, n_pad = 576):
    '''
    Check if the padding has any effect on the loss value of the model.

    Inputs:
    -------
        model:  tf-model, consisting of two input head which take indeces base_feat and res_feat from x_nopad
        data_raw: np.array
        data_padded: np.array, padded version of x_nopad

    Outputs:
    --------
        None; loss-values are printed for information
    '''

    n_batch, n_steps, n_feat = x_nopad.shape
    assert(n_pad > n_steps)
    delta = n_pad-n_steps

    if x_nopad.dtype != np.float32:
        x_nopad = x_nopad.astype(np.float32)
    if y_nopad.dtype != np.float32:
        y_nopad = y_nopad.astype(np.float32)

    x_pad = np.zeros((n_batch, n_pad, n_feat), dtype=np.float32)
    y_pad = np.zeros((n_batch, n_pad, y_nopad.shape[-1]), dtype=np.float32)
    x_pad[:,0:n_steps, :] = x_nopad
    y_pad[:,0:n_steps, :] = y_nopad

    loss_nopad, _ = model.evaluate([x_nopad[:,:,base_feat], x_nopad[:,:, res_feat]], y_nopad)
    print(compute_loss_mae(y_nopad, model.predict([x_nopad[:,:,base_feat], x_nopad[:,:, res_feat]])))
    loss_pad, _ = model.evaluate([x_pad[:,:,base_feat], x_pad[:,:, res_feat]], y_pad)
    print(compute_loss_mae(y_pad, model.predict([x_pad[:,:,base_feat], x_pad[:,:, res_feat]])))

    assert(np.isclose(loss_nopad, loss_pad))
    print('loss (no_pad): ', loss_nopad, '  loss (pad): ', loss_pad)

    

def check_if_rnn_version(model_ffn, model_rnn):
    '''
    Check if model_rnn is the rnn-version of model_ff.
    By this we mean: model_ffn and model_rrn share the same layers and weights with the exception of a single SimpleRNN layer which replaces a Dense layer and has its memory loop deactivated.
    '''
    bool_lst = []

    N_layers = len(model_ffn.layers)
    assert(N_layers == len(model_rnn.layers))

    # loop over layers
    for l_ffn, l_rnn in zip(model_ffn.layers, model_rnn.layers):
        #print(type(l_ffn), type(l_rnn))
        if type(l_rnn) != type(SimpleRNN(1)):
            bool_lst += [(w_ffn == w_rnn).all() for w_ffn, w_rnn in zip(l_ffn.get_weights(), l_rnn.get_weights())]
        else:
            bool_lst += [(w_ffn == w_rnn).all() for w_ffn, w_rnn in zip(l_ffn.get_weights(), [l_rnn.get_weights()[i] for i in [0,2]])] 

    return np.array(bool_lst)      


def predict_contract_backtest(x_raw, x_ts, y_ts, pmodel_ffn, pmodel_rnn, discount, age_scale, bool_print_shapes = True):
    '''
        Compare the ffn-computation implemented in predict_ffn_contract_vectorized with predict_rnn_contract_vectorized.
    '''

    assert((check_if_rnn_version(model_ffn=pmodel_ffn, model_rnn=pmodel_rnn)== True).all())

    N_batch = x_raw.shape[0]
    assert(N_batch == x_ts.shape[0])

    # check if effective lengths of all contracts in the batch are equal
    N_eff = x_raw[:,1]/x_raw[:,3]
    assert(np.max(N_eff)==np.min(N_eff))
    N_eff = int(N_eff[0])
    assert(N_eff==x_ts.shape[1])

    # are relevant input data indentical? Recall that x_ts is scaled while x_raw isn't
    #print(x_raw[0:5,0]/age_scale)
    #print(x_ts[0:5,0,0])
    assert((x_raw[:,0]/age_scale==x_ts[:,0,0]).all())
    assert((x_raw[:,3]==x_ts[:,0,3]).all())
  
    # CF  shape: (N_batch, N_eff, 2)
    CFs = get_CFs_vectorized(x_raw) #.reshape((N_batch, N_eff, 2))

    # Are CF values identical?
    assert((CFs==y_ts).all())
    
    ##### desired shape: (N_batch, N_eff, 2) -> use np.swapaxes
    # FFN:
    probs_ffn = np.array([pmodel_ffn.predict(np.array([(x_raw[:,0]+k*x_raw[:,3])/age_scale, x_raw[:,3]], ndmin=2).T) for k in np.arange(N_eff)])
    probs_ffn = np.swapaxes(probs_ffn, axis1 = 0, axis2 = 1) 
    print('probs_ffn shape: ', probs_ffn.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))   

    # RNN:
    probs_rnn = pmodel_rnn.predict(x_ts[:,:,[0,3]])
    print('probs_rnn shape: ', probs_rnn.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))   

    # # -> exact equality not given due to limited precision with softmax-function (?!?!)
    #print(probs_ffn[0])
    #print(probs_rnn[0])

    assert(np.isclose(probs_rnn, probs_ffn).all())
    #assert((probs_rnn==probs_ffn).all())

    # preceed as

    # cum_probs shape: (N_batch, N_eff); path-wise, N_eff-times multiplication of 1-dim transition-probs
    cum_prod = np.cumprod(probs_rnn[:,:,0], axis=1)
    cum_init = np.ones(shape = (len(cum_prod), 1))
    cum_prod = np.concatenate([cum_init, cum_prod[:,0:-1]], axis = 1).reshape(N_batch,N_eff,1) # Note: final dim to allow broadcasting

    prob_eff = probs_rnn*cum_prod

    # combine CFs at N_eff - times with probabilities cum_prob to reach that state and discount by GAMMA
    # Note: CFs are solely for p_surv, p_die -> slice cum_prob along axis = -1 -> shape: (N_batch, N_eff, 2)
    # Disconting: broadcast of step-dependent discounting
    discount = discount**(x_ts[:,:,3]* np.arange(N_eff).reshape(1,-1)).reshape(N_batch,N_eff,1)
    
    # sum weighted&discounted cash-flows for the batch (along axis=0)
    vals = (prob_eff*CFs*discount)

    if bool_print_shapes:
        print('CFs shape: ', CFs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('cum_prod shape: ', cum_prod.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('prob_eff shape: ', prob_eff.shape, ' expected: ({},{}, 2)'.format(N_batch,N_eff))
        print('discounting shape: ', discount.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('vals shape: ', vals.shape, ' expected: ({},{},{})'.format(N_batch, N_eff, 2))

    return vals.reshape((N_batch,-1)).sum(axis=1).reshape((-1,1))


def predict_rnn_contract_vectorized(x, y, pmodel, discount, bool_print_shapes = False):
    '''
        Perform the Markov-computation of Cash-flows, based on probabilites contained in pmodel.
        Note:   new version which addresses issue of overestimating 1/m_q_x
                before: CF_t accidentally weighted by (1/m+t)_p_x and (1/m+t)_q_m
                now: CF_x weighted by (1/m+t)_p_x and (t-1/m)_p_x*1/m_q_(x+t-1/m) 

        Inputs:
        -------
            x:  scaled contracts, seqence-shape: (N_batch, N_steps, N_features) 
                important: equals lengths "n/freq" (data-prep) and potentially different step-sizes "freq", i.e. discounting of steps varies)
                form : x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
            y:  cash-flows for contract(s) x, sequence-shape: (N_batch, N_steps, 2)
            pmodel: tf model (RNN type), which predicts p_survive and p_die within the next year, given an age

        Outputs:
        --------
            sum-value of the probability-weighted and discounted cash-flows
    '''
    N_batch = x.shape[0]

    # check if effective lengths of all contracts in the batch are equal
    N_eff = x.shape[1]
  
    # CF  shape: (N_batch, N_eff, 2)
    #CFs = get_CFs_vectorized(x) #.reshape((N_batch, N_eff, 2))
    CFs = y
    assert(CFs.shape[1] == N_eff)
    
    # rrn: pmodel.predict(x) -> shape: (N_batch, N_eff, 2)
    probs = pmodel.predict(x[:,:,[0,3]])
    # cum_probs shape: (N_batch, N_eff); path-wise, N_eff-times multiplication of 1-dim transition-probs
    cum_prod = np.cumprod(probs[:,:,0], axis=1)
    cum_init = np.ones(shape = (len(cum_prod), 1))
    cum_prod = np.concatenate([cum_init, cum_prod[:,0:-1]], axis = 1).reshape(N_batch,N_eff,1) # Note: final dim to allow broadcasting

    prob_eff = probs*cum_prod

    # combine CFs at N_eff - times with probabilities cum_prob to reach that state and discount by GAMMA
    # Note: CFs are solely for p_surv, p_die -> slice cum_prob along axis = -1 -> shape: (N_batch, N_eff, 2)
    # Disconting: broadcast of step-dependent discounting
    discount = discount**(x[:,:,3]* np.arange(N_eff).reshape(1,-1)).reshape(N_batch,N_eff,1)
    
    # sum weighted&discounted cash-flows for the batch (along axis=0)
    vals = (prob_eff*CFs*discount)

    if bool_print_shapes:
        print('CFs shape: ', CFs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('probs shape: ', probs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('cum_prod shape: ', cum_prod.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('prob_eff shape: ', prob_eff.shape, ' expected: ({},{}, 2)'.format(N_batch,N_eff))
        print('discounting shape: ', discount.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('vals shape: ', vals.shape, ' expected: ({},{},{})'.format(N_batch, N_eff, 2))

    return vals.reshape((N_batch,-1)).sum(axis=1).reshape((-1,1))