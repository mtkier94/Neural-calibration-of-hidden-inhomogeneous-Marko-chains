import numpy as np
#from numba import njit, prange
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import mean_absolute_error
import multiprocessing
import signal, pickle
import os, time
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.core import Masking
from tensorflow.python.ops.gen_array_ops import deep_copy
import tensorflow_addons as tfa


#from functions.sub_actuarial import get_CFs_vectorized, predict_contract_vectorized, predict_rnn_contract_vectorized
from functions.tf_loss_custom import compute_loss_mae
from functions.tf_model_res import random_pretraining, combine_models, create_mortality_res_net, train_combined_model
from functions.sub_backtesting import check_exploded_gradients, check_model_mask_vs_no_mask, check_padding

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_with_padding_hpsearch_male, path_models_resnet_with_padding_hpsearch_female



if __name__ ==  '__main__':

    bool_train = True
    bool_mask = True # insert a masking layer into the model
    baseline_sex = 'female'

    if baseline_sex == 'male':
        path_model = path_models_resnet_with_padding_hpsearch_male
    elif baseline_sex == 'female':
        path_model = path_models_resnet_with_padding_hpsearch_female
    else:
        assert False, 'Unknown Option for baseline_sex.'


    #### load data
    with open(os.path.join(path_data, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_discounted.npy'), 'rb') as f:
        y_ts_discounted = np.load(f, allow_pickle=True)

    # zero-padded seq
    with open(os.path.join(path_data, 'x_ts_pad.npy'), 'rb') as f:
        x_ts_pad = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_pad_discounted.npy'), 'rb') as f:
        y_ts_pad_discounted = np.load(f, allow_pickle=True)


    N_batches = len(x_ts)
    N_contracts, N_seq_pad, N_features = x_ts_pad.shape


    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]


    # HP_PARAMS (grid-search)
    LR_RATES = [1e-2, 1e-3, 1e-4] #5e-2,  1e-5]

    loss = np.zeros((N_batches, len(LR_RATES)))
    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))
    pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

    pmodel_res = create_mortality_res_net(hidden_layers=[40, 40, 20], param_l2_penalty=0.1, input_shape=(None, len(res_features)), n_out=2)
    pmodel_res.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = False)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # Note: we do not train, but only evaluate -> specifics of optimizer irrelevant

    print('Checking the general padding mechanism of the model on sample x[10], y[10]')
    check_padding(model = pmodel, x_nopad = x_ts[10], y_nopad = y_ts_discounted[10], base_feat = base_features, res_feat = res_features, n_pad = x_ts_pad.shape[1])
    
    print('\n', 'checking the masked vs the non-masked model:')
    check_model_mask_vs_no_mask(x_base = x_ts_pad[:,:,base_features], x_res = x_ts_pad[:,:,res_features], y= y_ts_pad_discounted, model_base=pmodel_base, iterations=2)
    print('\n')

    # part 1: check masking in res_model
    if True:
        id = 500
        quant = 1 # must be one currently
        no_padding = Masking(0.0)(x_ts_pad[id:id+quant,:,res_features])._keras_mask.numpy().reshape((quant, -1, 1))

        # step 1: (optional Masking) -> Dense -> Dense -> GRU
        model_debug = Model(pmodel_res.input, pmodel_res.layers[3].output)
        pred_mask = model_debug(Masking(0.0)(x_ts_pad[id:id+quant,:,res_features]))
        pred_nomask = model_debug(x_ts_pad[id:id+quant,:,res_features])
        # print(pred_mask)
        # print(pred_nomask)
        assert(np.allclose(pred_mask*no_padding, pred_nomask*no_padding))

        # step 2: (optional Masking) -> Dense -> Dense -> GRU -> Dense -> Softmax
        model_debug = Model(pmodel_res.input, pmodel_res.output)
        model_debug.compile(loss = compute_loss_mae, optimizer='adam')
        #print('res-net')
        #print(model_debug.summary())
        pred_mask = model_debug(Masking(0.0)(x_ts_pad[id:id+quant,:,res_features]))
        pred_nomask = model_debug(x_ts_pad[id:id+quant,:,res_features])
        # print(pred_mask)
        # print(pred_nomask)
        # Note: Values can only be compared on non-zero-padded time-steps
        # For zero-padded time steps: 
        assert(np.allclose(pred_mask*no_padding, pred_nomask*no_padding))

        # step 3: check loss for both models with costum loss function
        print('Applying Masking as Preprocessing layer to input, outside of the tf.model:')
        print('custom loss (wo masking): ', compute_loss_mae(y_true = y_ts_pad_discounted[id:id+quant], y_pred=pred_nomask))
        print('\t evaluated loss: ', model_debug.evaluate(x=x_ts_pad[id:id+quant,:,res_features], y = y_ts_pad_discounted[id:id+quant], batch_size=1024))
        print('custom loss (with input-masking): ', compute_loss_mae(y_true = y_ts_pad_discounted[id:id+quant], y_pred=pred_mask))
        print('\t evaluated loss: ', model_debug.evaluate(x=Masking(0.0)(x_ts_pad[id:id+quant,:,res_features]), y = y_ts_pad_discounted[id:id+quant], batch_size=1024))
       

    ### Part 2: Check combined model
    if True:
        id = 500
        quant = 1
        # padding-mask; note: mask irrespective of no. of features, only with time-steps
        mask = Masking(0.0)(x_ts_pad[id:id+quant,:,res_features])._keras_mask.numpy().reshape((quant, -1, 1))

        # step 1: Combined model with masking and without masking
        model_mask = combine_models(pmodel_base, pmodel_res, bool_masking = True)
        model_mask.compile(loss = compute_loss_mae, optimizer='adam')
        #model_mask.summary()
        model_nomask = combine_models(pmodel_base, pmodel_res, bool_masking = False)
        model_nomask.compile(loss = compute_loss_mae, optimizer='adam')

        pred_mask = model_mask.predict(x = [x_ts_pad[id:id+quant,:,base_features], x_ts_pad[id:id+quant,:,res_features]])
        pred_nomask = model_nomask.predict(x = [x_ts_pad[id:id+quant,:,base_features], x_ts_pad[id:id+quant,:,res_features]])


        # print(pred_mask)
        # print(pred_nomask)
        assert(np.allclose(pred_mask*mask, pred_nomask*mask))
        print('prediction values nomask vs mask also equal within non-masked sequence-area if Masking-layer is included withing the model', '\n')

        # step 2: compare model_nomask with masked input (preprocessed) with model_mask
        pred_new = model_nomask.predict(x = [Masking(0.0)(x_ts_pad[id:id+quant,:,base_features]), Masking(0.0)(x_ts_pad[id:id+quant,:,res_features])])
        assert(np.allclose(pred_nomask, pred_new)) #-> preprocessed input (via masking) does not pass the mask to subsequent layers
        


        # step 3: compare losses; they should be equal, as the same weights are used and targets are zero-padded too -> hence provide an additional, natural masking for predictions at zero-padded time-steps
        print('masking (model-integrated) vs no masking on all data:')
        x, y = [x_ts_pad[id:id+quant,:,base_features],x_ts_pad[id:id+quant,:,res_features]], y_ts_pad_discounted[id:id+quant]
        print('custom loss (wo masking): ', compute_loss_mae(y_true = y, y_pred=pred_nomask))
        print('\t evaluated loss: ', model_nomask.evaluate(x= x, y = y))#, batch_size=1024))
        print('custom loss (with masking): ', compute_loss_mae(y_true = y, y_pred=pred_mask))
        print('\t evaluated loss: ', model_mask.evaluate(x=x, y = y))#, batch_size=1024))

        # padding should only affect times where also target values have vbeen zero-padded
        assert((y_ts_pad_discounted[id:id+quant] == y_ts_pad_discounted[id:id+quant]*mask).all())
        exit()
        # assumption 1: evaluate is the mean loss over all batches, whereas loss(y_true, y_pred) is the global loss 

        n_batches = np.round(np.ceil(len(x_ts_pad)/1024))
        eval_batch = [None]*n_batches
        for i in range(n_batches):
            eval_batch[i] = model_nomask.evaluate(x=[x_ts_pad[id:id+quant,:,base_features], x_ts_pad[id:id+quant,:,res_features]], y = y_ts_pad_discounted[id:id+quant], batch_size=1024)



        # step 3: plot different in evaluate and tf_loss -> no constant relative or absolute relation observable
        # cache = np.zeros((2,100))
        # for id in range(100):
        #     cache[0,id] = model_nomask.evaluate(x=[x_ts_pad[id:id+1,:,base_features], x_ts_pad[id:id+1,:,res_features]], y = y_ts_pad_discounted[id:id+1], batch_size=1024, verbose = 0)
        #     cache[1,id] = model_mask.evaluate(x=[x_ts_pad[id:id+1,:,base_features], x_ts_pad[id:id+1,:,res_features]], y = y_ts_pad_discounted[id:id+1], batch_size=1024, verbose = 0)

        # # plt.plot(cache[0] - cache[1], label = 'nomask - mask prediction')
        # plt.legend()
        # plt.show()
        # plt.plot(cache[0]/cache[1], label = 'nomask/ mask prediction')
        # plt.legend()
        # plt.show()