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
from functions.sub_visualization import mortality_rnn_heatmap, plot_new_vs_init_loss, mortality_heatmap, mortality_heatmap_grid, plot_implied_survival_curve


from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_wo_padding_hpsearch

def exp_decay_scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


if __name__ ==  '__main__':
    
    bool_train = False


    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)

    path_model = path_models_resnet_wo_padding_hpsearch
    strategy = tf.distribute.MirroredStrategy()

    # option for processing data during training
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    #### load data
    with open(os.path.join(path_data,'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data,'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data,'y_ts_discounted.npy'), 'rb') as f:
        y_ts_discounted = np.load(f, allow_pickle=True)

    

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


    # HP_PARAMS (fixed)
    ITERATIONS_PER_EPOCH = 25#00 # Note: Not too high too avoid overfitting on low-duration contracts
    EPOCHS = 100#0 # Note: relatively high too fit transition-probs smoothly on the whole feature space, iteratively per round for all batches

    # HP_PARAMS (grid-search)
    LR_RATES = [1e-2, 1e-3, 1e-4, 1e-5]

    loss = np.zeros((N_batches, len(LR_RATES)))


    for k, lrate in enumerate(LR_RATES):

        #with strategy.scope():

        pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer,r'survival_baseline_ts.h5'))
        pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
        pmodel_res = create_mortality_res_net(hidden_layers = [40,40,20], input_shape= (None, len(res_features)), n_out=2)
        pmodel = combine_base_and_res_model(model_base = pmodel_base, model_res = pmodel_res)
        # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
        # use gradient clipping and/ or lr-decay to avoid this issue
        #tf.keras.optimizers.Adam(clipnorm=1)
        opt = tf.keras.optimizers.Adam(lr=lrate)
        opt_avg = tfa.optimizers.MovingAverage(opt)
        for optimizer in [opt, opt_avg]:
            
            if optimizer == opt:
                tag = 'adam'
            else:
                tag = 'avg'
            
            if bool_train:
                # train pmodel
                pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)
                history = train_combined_model(pmodel, x = x_ts, y=y_ts_discounted, base_feat_in= base_features, res_feat_in=res_features,
                                            iterations_per_epoch=ITERATIONS_PER_EPOCH, epochs=EPOCHS)
                # include save_format as we use a custom loss (see load_model below for futher syntax)
                pmodel.save(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(lrate, tag)), save_format = 'tf')
                np.save(os.path.join(path_model, r'model_lr_{}_opt_{}_hist.npy'.format(lrate, tag)), history)
            elif os.path.exists(os.path.join(path_model, r'model_lr_{}_opt_{}_hist.npy'.format(lrate, tag))):
                pmodel = load_model(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(lrate, tag)), compile=False)
                pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = opt_avg)
                with open(os.path.join(path_model, r'model_lr_{}_opt_{}_hist.npy'.format(lrate, tag)), 'rb') as f:
                    history = np.load(f, allow_pickle=True)
            
                # plot training history
                plt.plot(np.arange(len(np.array(history).flatten()))/(ITERATIONS_PER_EPOCH*N_batches), np.array(history).flatten(), alpha = 0.3, label='lr: {}, opt: {}'.format(lrate, tag))
            
            else:
                print('train-flag is off and model with lr {} and optimizer {} not yet trained.'.format(lrate, tag))
                
            
    # display collective plot for all training-settings loaded in loop
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path_model, 'HP_seach_lrates.png'), bbox_inches='tight')
    plt.close()

    # look at one individual model and visualize progress
    pmodel = load_model(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(1e-3, 'adam')), compile=False)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = opt_avg)
    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008Tmale.csv'),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))



    # where did the training alter the underlying mortality-table?

    plot_implied_survival_curve(pmodel, dav_table = p_survive, age_max = T_MAX, m=1 )



    val_dict, true_vals = mortality_heatmap_grid(pmodel, p_survive, m=1, age_max=T_MAX, rnn_seq_len = 20, save_path = path_model)



    fig, ax = plt.subplots(2,2, figsize=(12,10))
    sns.heatmap(val_dict['male']['nonsmoker'] -val_dict['female']['nonsmoker'], ax=ax[0,0])
    ax[0,0].set_title('male (non-smoker) - female (non smoker)')
    sns.heatmap(val_dict['male']['nonsmoker'] -val_dict['male']['smoker'], ax=ax[0,1])
    ax[0,1].set_title('male (non-smoker) - male (smoker)')
    sns.heatmap(val_dict['male']['smoker'] -val_dict['female']['smoker'], ax=ax[1,0])
    ax[1,0].set_title('male (smoker) - female (smoker)')
    sns.heatmap(val_dict['female']['nonsmoker'] -val_dict['female']['smoker'], ax=ax[1,1])
    ax[1,1].set_title('female (non-smoker) - female (smoker)')
    fig.suptitle(r'survival prob. $p_{x+m}$: dav_table - model_prediction')
    plt.savefig(os.path.join(path_model, 'heatmap_grid_model_differences.png'))
    plt.close()


    # loss per no. of HMC iterations
    plot_new_vs_init_loss(pmodel, pmodel_base, x_ts, y_ts_discounted, base_features, res_features)

    
