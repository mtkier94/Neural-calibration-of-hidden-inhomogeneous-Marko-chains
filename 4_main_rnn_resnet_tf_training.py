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
import tensorflow_addons as tfa


from functions.sub_actuarial import get_CFs_vectorized, predict_contract_vectorized, predict_rnn_contract_vectorized
from functions.tf_loss_custom import compute_loss_mae
from functions.tf_model_res import create_mortality_res_net, combine_base_and_res_model, train_combined_model
from functions.sub_visualization import mortality_rnn_heatmap, plot_new_vs_init_loss

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_resnet_wo_padding, path_models_baseline_transfer


if __name__ ==  '__main__':

    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)

    #strategy = tf.distribute.MirroredStrategy()

    #### load data
    with open(os.path.join(path_data, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_discounted.npy'), 'rb') as f:
        y_ts_discounted = np.load(f, allow_pickle=True)

    bool_train = False

    N_batches = len(x_ts)
    N_features = x_ts[0].shape[2]

    N_contracts = 0
    for x in x_ts:
        N_contracts += len(x)

    data = np.concatenate([x[:,0,:] for x in x_ts_raw], axis = 0)
    assert(len(data) == N_contracts)


    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]


    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer,r'survival_baseline_ts.h5'))
    pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    pmodel_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, len(res_features)), n_out=2)
    pmodel = combine_base_and_res_model(model_base = pmodel_base, model_res = pmodel_res)
    # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
    # use gradient clipping and/ or lr-decay to avoid this issue
    #tf.keras.optimizers.Adam(clipnorm=1)
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    opt_avg = tfa.optimizers.MovingAverage(opt)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = opt_avg)


    ###################################
    ITERATIONS_PER_EPOCH = 5#00 # Note: Not too high too avoid overfitting on low-duration contracts
    EPOCHS = 10#00 # Note: relatively high too fit transition-probs smoothly on the whole feature space, iteratively per round for all batches
    #batch_sz = 32 # not tuned yet (!!) -> set to 
    ###################################
    
    # visualize difference in random init of pmodel_res
    #plot_new_vs_init_loss(pmodel, pmodel_base, x_ts, y_ts_discounted, base_features, res_features)
    
    if bool_train:
        # train pmodel
        history = train_combined_model(pmodel, x = x_ts, y=y_ts_discounted, base_feat_in= base_features, res_feat_in=res_features,
                                    iterations_per_epoch=ITERATIONS_PER_EPOCH, epochs=EPOCHS)
        # include save_format as we use a custom loss (see load_model below for futher syntax)
        pmodel.save(os.path.join(path_models_resnet_wo_padding, r'model.h5'), save_format = 'tf')
        np.save(os.path.join(path_models_resnet_wo_padding, r'model_hist.npy'), history)
    else:
        pmodel = load_model(os.path.join(path_models_resnet_wo_padding, r'model.h5'), compile=False)
        pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = opt_avg)
        with open(os.path.join(path_models_resnet_wo_padding, r'model_hist.npy'), 'rb') as f:
            history = np.load(f, allow_pickle=True)

    # visualize progress
    plot_new_vs_init_loss(pmodel, pmodel_base, x_ts, y_ts_discounted, base_features, res_features)


    history_flattened = np.array(history).flatten()
    plt.plot(np.arange(len(history_flattened))/N_batches, history_flattened)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()

    #### NOTE: custom loss computation
    # https://stackoverflow.com/questions/41132633/can-numba-be-used-with-tensorflow

    
