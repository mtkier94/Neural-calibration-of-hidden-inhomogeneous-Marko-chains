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
from matplotlib.colors import LogNorm

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
from functions.tf_model_res import train_combined_model_on_padded_data
from functions.sub_visualization import mortality_rnn_heatmap, plot_new_vs_init_loss, mortality_heatmap, mortality_heatmap_grid, plot_implied_survival_curve
from functions.sub_backtesting import check_exploded_gradients

from global_vars import T_MAX, GAMMA, AGE_RANGE
from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_with_padding_hpsearch



if __name__ ==  '__main__':


    tf.keras.backend.set_floatx('float64')
    bool_train = False
    bool_mask = True # insert a masking layer into the model


    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
    # ALternative: For numeric stability, set the default floating-point dtype to float64
    #tf.keras.backend.set_floatx('float64')

    path_model = path_models_resnet_with_padding_hpsearch
    #strategy = tf.distribute.MirroredStrategy()

    # option for processing data during training
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    #### load data
    with open(os.path.join(path_data, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_discounted.npy'), 'rb') as f:
        y_ts_discounted = np.load(f, allow_pickle=True)

    # zero-padded seq
    with open(os.path.join(path_data, 'x_ts_pad.npy'), 'rb') as f:
        x_ts_pad = np.load(f, allow_pickle=True)#.astype(np.float64)
    with open(os.path.join(path_data, 'y_ts_pad_discounted.npy'), 'rb') as f:
        y_ts_pad_discounted = np.load(f, allow_pickle=True)#.astype(np.float64)

    # load training times
    with open(os.path.join(path_model, 'training_times.p'), 'rb') as fp:
            times_dict = pickle.load(fp)

    N_batches = len(x_ts)
    N_contracts, N_seq_pad, N_features = x_ts_pad.shape

    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]


    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'survival_baseline_ts.h5'))
    pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

   
    # look at one individual model and visualize progress
    pmodel = load_model(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(1e-2, 'adam')), compile=False)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
    p_survive = pd.read_csv(os.path.join(path_data, r'DAV2008Tmale.csv'),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))

    pmodel.summary()


    bool_grad = check_exploded_gradients(pmodel)
    if bool_grad:
        print('-------------------')
        print('NaN-parameter-values in model!')
        print('-------------------')
        ValueError


    # where did the training alter the underlying mortality-table?
    loss, _ = pmodel.evaluate([x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted,  batch_size = 1024, verbose = 0)
    #loss_classic = compute_loss_mae(y_true = y_ts_pad_discounted, y_pred = pmodel.predict([x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]])).numpy()
    
    #print('tf.eval vs. tf.loss: ', loss, ' vs. ', np.round_(loss_classic,2))
    #str_loss = f'(loss(: {str(np.round_(loss_classic,2))})'
    str_loss = ''
    val_mort_loss = plot_implied_survival_curve(pmodel, dav_table = p_survive, age_max = T_MAX, m=1, path_save = path_model, str_loss = str_loss, age_range= AGE_RANGE  )

    # plt.plot(val_mort_loss['male']['nonsmoker']-val_mort_loss['male']['smoker'], label = 'male')
    # plt.plot(val_mort_loss['female']['nonsmoker']-val_mort_loss['female']['smoker'], label = 'female')
    # plt.hlines(y=0.02,xmin=0, xmax=121)
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    # exit()

    val_dict, true_vals = mortality_heatmap_grid(pmodel, p_survive, m=1, age_max=T_MAX, rnn_seq_len = 20, save_path= path_model, age_range=AGE_RANGE)

    # print(true_vals[0:11, 0:11])
    # sns.heatmap(true_vals)
    # plt.show()


    # n, k = true_vals.shape
    # for i in range(k):
    #     plt.plot(np.arange(i, i+n), 1-true_vals[:,i])
    # plt.yscale('log')
    # plt.show()
    # exit()


    # for sex in ['male', 'female']:
    #     for status in ['nonsmoker', 'smoker']:
    #         # print('by implied loss (shape): ', val_mort_loss[sex][status].shape)
    #         # print('by heatmap (shape): ', val_dict[sex][status].shape)

    #         # print('by implied loss: ', val_mort_loss[sex][status])
    #         # print('by heatmap: ', val_dict[sex][status][:,0]) # grab only 1st time-step
    #         assert(np.allclose(val_mort_loss[sex][status], val_dict[sex][status][:,0]))
    #         plt.plot(1- val_dict[sex][status][:,0], label=sex+status)
    
    # plt.yscale('log')
    # plt.legend()
    # plt.show()

    # n,m = val_dict['male']['nonsmoker'].shape
    # for i in range(m):
    #     for sex in ['male', 'female']:
    #         if sex == 'male':
    #             tag = 'blue'
    #         else:
    #             tag = 'green'
    #         plt.plot(np.arange(i, i+n), val_dict[sex]['nonsmoker'][:,i]-val_dict[sex]['smoker'][:,i], color = tag)#, label=sex)
    
    # #plt.yscale('log')
    # #plt.legend()
    # plt.show()

    # sns.heatmap(val_dict['male']['nonsmoker'] -val_dict['female']['smoker'])
    # plt.show()
    # exit()
    
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    sns.heatmap(val_dict['male']['nonsmoker'] -val_dict['female']['nonsmoker'], ax=ax[0,0])
    ax[0,0].set_title('male (non-smoker) - female (non smoker)')
    
    sns.heatmap(val_dict['male']['nonsmoker'] -val_dict['male']['smoker'], ax=ax[0,1])
    ax[0,1].set_title('male (non-smoker) - male (smoker)')
    sns.heatmap(val_dict['male']['smoker'] -val_dict['female']['smoker'], ax=ax[1,0])
    ax[1,0].set_title('male (smoker) - female (smoker)')
    sns.heatmap(val_dict['female']['nonsmoker'] -val_dict['female']['smoker'], ax=ax[1,1])
    ax[1,1].set_title('female (non-smoker) - female (smoker)')
    fig.suptitle(r'differences of survival probs. $p_{x+k}$ of the model')
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xlabel('k')
            ax[i,j].set_ylabel('x')
    plt.savefig(os.path.join(path_model, 'heatmap_grid_model_differences.png'))
    plt.close()


    # zoom in
    m = 1
    age_low, age_up = AGE_RANGE
    index_low, index_up = age_low*m, age_up*m+1
    hm_yticker = [(t if int(t) != t else int(t)) if t%5==0 else '' for t in np.arange(age_low, age_up, 1/m)]

    fig, ax = plt.subplots(2,2, figsize=(12,10))
    sns.heatmap(val_dict['male']['nonsmoker'][index_low:index_up] -val_dict['female']['nonsmoker'][index_low:index_up], ax=ax[0,0], yticklabels=hm_yticker)
    ax[0,0].set_title('male (non-smoker) - female (non smoker)')
    sns.heatmap(val_dict['male']['nonsmoker'][index_low:index_up] -val_dict['male']['smoker'][index_low:index_up], ax=ax[0,1], yticklabels=hm_yticker, norm=LogNorm())
    ax[0,1].set_title('male (non-smoker) - male (smoker)')
    sns.heatmap(val_dict['male']['smoker'][index_low:index_up] -val_dict['female']['smoker'][index_low:index_up], ax=ax[1,0], yticklabels=hm_yticker)
    ax[1,0].set_title('male (smoker) - female (smoker)')
    sns.heatmap(val_dict['female']['nonsmoker'][index_low:index_up] -val_dict['female']['smoker'][index_low:index_up], ax=ax[1,1], yticklabels=hm_yticker, norm=LogNorm())
    ax[1,1].set_title('female (non-smoker) - female (smoker)')
    fig.suptitle(r'differences of survival probs. $p_{x+k}$ of the model')
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xlabel('k')
            ax[i,j].set_ylabel('x')
    plt.savefig(os.path.join(path_model, 'heatmap_grid_model_differences_zoom.png'))
    plt.close()


    # check relative difference for smoker status
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    sns.heatmap(val_dict['male']['nonsmoker'][index_low:index_up] -val_dict['female']['nonsmoker'][index_low:index_up], ax=ax[0,0], yticklabels=hm_yticker)
    ax[0,0].set_title('male (non-smoker) - female (non smoker)')
    sns.heatmap(np.log(val_dict['male']['nonsmoker'][index_low:index_up]/val_dict['male']['smoker'][index_low:index_up]), ax=ax[0,1], yticklabels=hm_yticker)
    ax[0,1].set_title('log(male (non-smoker)/male (smoker))')
    sns.heatmap(val_dict['male']['smoker'][index_low:index_up] -val_dict['female']['smoker'][index_low:index_up], ax=ax[1,0], yticklabels=hm_yticker)
    ax[1,0].set_title('male (smoker) - female (smoker)')
    sns.heatmap(np.log(val_dict['female']['nonsmoker'][index_low:index_up]/val_dict['female']['smoker'][index_low:index_up]), ax=ax[1,1], yticklabels=hm_yticker)
    ax[1,1].set_title('log(female (non-smoker)/female (smoker))')
    fig.suptitle(r'differences of survival probs. $p_{x+k}$ of the model')
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xlabel('k')
            ax[i,j].set_ylabel('x')
    plt.show()


    print('mean, rel. smoker buffer: ', np.mean(val_dict['male']['nonsmoker'][index_low:index_up]/val_dict['male']['smoker'][index_low:index_up]))
    print('\t vals: ')
    print((val_dict['male']['nonsmoker'][index_low:index_up]/val_dict['male']['smoker'][index_low:index_up])[0:5, 0:5])
    print('\t vals: ')
    print((val_dict['female']['nonsmoker'][index_low:index_up]/val_dict['female']['smoker'][index_low:index_up])[0:5, 0:5])


    # print('male: risk buffer for smoking')
    # print('\t mean: ', np.mean(val_dict['male']['nonsmoker'][index_low:index_up] -val_dict['male']['smoker'][index_low:index_up]))
    # print(val_dict['male']['nonsmoker'][index_low:index_up][0:10] -val_dict['male']['smoker'][index_low:index_up][0:10])
    # print('female: risk buffer for smoking')
    # print('\t mean: ', np.mean(val_dict['female']['nonsmoker'][index_low:index_up] -val_dict['female']['smoker'][index_low:index_up]))
    # print(val_dict['female']['nonsmoker'][index_low:index_up][0:10] -val_dict['female']['smoker'][index_low:index_up][0:10])


    # loss per no. of HMC iterations
    plot_new_vs_init_loss(pmodel, pmodel_base, x_ts, y_ts_discounted, base_features, res_features, path_save = path_model)

    
