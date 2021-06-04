import numpy as np
#from numba import njit, prange
import pandas as pd
import multiprocessing
import signal
import os, time
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers.core import Masking
import tensorflow_addons as tfa


from functions.sub_actuarial import get_CFs_vectorized, predict_contract_vectorized, predict_rnn_contract_vectorized
from functions.tf_loss_custom import compute_loss_mae
from functions.tf_model_res import create_mortality_res_net, train_combined_model
from functions.tf_model_res import combine_models, train_combined_model_on_padded_data
from functions.sub_visualization import mortality_rnn_heatmap, plot_new_vs_init_loss, mortality_heatmap, mortality_heatmap_grid, plot_implied_survival_curve
from functions.sub_backtesting import check_exploded_gradients, check_model_mask_vs_no_mask, check_padding

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_with_padding_hpsearch



if __name__ ==  '__main__':

    bool_train = False
    path_model = path_models_resnet_with_padding_hpsearch

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


    def check_dense_masking(x):

        in_shape = x.shape
        Masking(0.0)(x)
        layer_d1 = Dense(units = 40, input_shape = (None, in_shape[-1]))
        layer_d2 = Dense(units = 40)
        layer_rr = GRU(20)

        output_with_mask = layer_rr(layer_d2(layer_d1(Masking(0.0)(x))))
        output_wo_mask = layer_rr(layer_d2(layer_d1(x)))

        print(output_with_mask.shape)
        print(output_wo_mask.shape)
        print(output_with_mask)
        print(output_wo_mask)

        assert (output_with_mask.numpy() == output_wo_mask.numpy()).all()


    check_dense_masking(x_ts_pad[200:202])
    exit()

    N_batches = len(x_ts)
    N_contracts, N_seq_pad, N_features = x_ts_pad.shape
    #print(N_contracts, N_seq_pad, N_features)


    # backtesting length of data
    data = np.concatenate([x[:,0,:] for x in x_ts_raw], axis = 0)
    assert(len(data) == N_contracts)


    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]


    # HP_PARAMS (grid-search)
    LR_RATES = [1e-2, 1e-3, 1e-4, 1e-5]

    loss = np.zeros((N_batches, len(LR_RATES)))
    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'survival_baseline_ts.h5'))
    pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    #pmodel_base.summary()

    pmodel_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, len(res_features)), n_out=2)
    pmodel = combine_models(model_base = pmodel_base, model_res = pmodel_res)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')



    for k, lrate in enumerate(LR_RATES):

        # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
        # use gradient clipping and/ or lr-decay to avoid this issue
        opt = tf.keras.optimizers.Adam(lr=lrate, clipnorm=100.0)
        #opt = tf.keras.optimizers.Adam(lr=lrate)
        opt_avg = tfa.optimizers.MovingAverage(opt)

        for optimizer in [opt, opt_avg]:

            
            if optimizer == opt:
                tag = 'adam'
            else:
                tag = 'avg'
            
            # loading only, no training of models in this test-script
            if os.path.exists(os.path.join(path_model, r'model_lr_{}_opt_{}_hist.npy'.format(lrate, tag))):
                pmodel = load_model(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(lrate, tag)), compile=False)
                pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)

                with open(os.path.join(path_model, r'model_lr_{}_opt_{}_hist.npy'.format(lrate, tag)), 'rb') as f:
                    history = np.load(f, allow_pickle=True)

                bool_nan_val = check_exploded_gradients(pmodel)
                if bool_nan_val == True:
                    print('Model with nan-paramter-values! ', r'model_lr_{}_opt_{}.h5'.format(lrate, tag))
            else:
                print('train-flag is off and model with lr {} and optimizer {} not yet trained.'.format(lrate, tag))
                

    check_padding(model = pmodel, x_nopad = x_ts[10], y_nopad = y_ts_discounted[10], base_feat = base_features, res_feat = res_features, n_pad = x_ts_pad.shape[1])

    check_model_mask_vs_no_mask(x_base = x_ts_pad[:,:,base_features], x_res = x_ts_pad[:,:,res_features], y= y_ts_pad_discounted, model_base=pmodel_base, iterations=2)
    
