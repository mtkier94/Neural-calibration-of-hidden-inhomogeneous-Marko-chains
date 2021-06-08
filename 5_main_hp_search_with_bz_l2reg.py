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
from functions.tf_model_res import train_combined_model_on_padded_data
from functions.sub_visualization import mortality_rnn_heatmap, plot_new_vs_init_loss, mortality_heatmap, mortality_heatmap_grid, plot_implied_survival_curve
from functions.sub_backtesting import check_exploded_gradients

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_with_padding_hpsearch_male, path_models_resnet_with_padding_hpsearch_female


def exp_decay_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch%2==0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr



if __name__ ==  '__main__':


    print('----------------------------')
    print('Adjust saving process to new list of HPARAMS!')
    print('----------------------------')

    baseline_sex = 'male'
    bool_train = False
    bool_mask = True # insert a masking layer into the model
    # HP_PARAMS (fixed)
    EPOCHS = 150
    callbacks = [tf.keras.callbacks.LearningRateScheduler(exp_decay_scheduler)]

    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
    # ALternative: For numeric stability, set the default floating-point dtype to float64
    #tf.keras.backend.set_floatx('float64')

    if baseline_sex == 'male':
        path_model = path_models_resnet_with_padding_hpsearch_male
    elif baseline_sex == 'female':
        path_model = path_models_resnet_with_padding_hpsearch_female
    else:
        assert False, 'Unknown Option for baseline_sex.'
    strategy = tf.distribute.MirroredStrategy()

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
        x_ts_pad = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_pad_discounted.npy'), 'rb') as f:
        y_ts_pad_discounted = np.load(f, allow_pickle=True)


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

    #train_dataset = tf.data.Dataset.from_tensor_slices(({'input_1': x_ts_pad[base_features],'input_2':  x_ts_pad[res_features]}, y_ts_pad_discounted))
    # Shuffle and slice the dataset.
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64).prefetch().cache()


    # HP_PARAMS (grid-search)
    LR_RATES = [5e-2, 1e-2, 1e-3]#, 1e-4] #5e-2,  1e-5]

    loss = np.zeros((N_batches, len(LR_RATES)))
    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))
    pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    # pmodel_base.summary()

    pmodel_res = create_mortality_res_net(hidden_layers=[40, 40, 20], param_l2_penalty=0.1, input_shape=(None, len(res_features)), n_out=2)
    pmodel_res.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    # pmodel_res.summary()
    
    # keep random initialization for later use (optional resetting for different hyperparams)
    w_init = deepcopy(pmodel_res.get_weights())

    pmodel_transfer = combine_models(pmodel_base, pmodel_res, bool_masking = True)
    

    # pmodel, w_init = random_pretraining(masking = bool_mask, iterations = 1, model_base = pmodel_base, x = [x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted)
    # model_test = clone_model(combine_models(pmodel_base, create_mortality_res_net()))
    # model_test.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

    times_dict = {}

    for k, lrate in enumerate(LR_RATES):

        # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
        # use gradient clipping and/ or lr-decay to avoid this issue
        opt = tf.keras.optimizers.Adam(lr=lrate, clipnorm=100.0)
        #opt = tf.keras.optimizers.Adam(lr=lrate)
        opt_avg = tfa.optimizers.MovingAverage(opt)

        for optimizer in [opt]: #, opt_avg]:
            for hp_batchsize in [32, 64]:#[64, 128]:

                EPOCHS_ADJ = EPOCHS*max(1,hp_batchsize//64)
            
                if optimizer == opt:
                    tag = 'adam'
                else:
                    tag = 'avg'
                
                if bool_train:
                    
                    print('Running with lr {}, optimizer {} and bz {} and training for {} epochs.'.format(lrate, tag, hp_batchsize, EPOCHS_ADJ))
                    # train pmodel
                    # initiate new res_new and reset weights -> same pseudo-random start for different optimizers
                    pmodel_res = create_mortality_res_net(hidden_layers=[40, 40, 20], param_l2_penalty=0.1, input_shape=(None, len(res_features)), n_out=2)
                    pmodel_res.set_weights(w_init)
                    pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = True)
                    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)
                    #check if weight-initialization was sucessfull
                    #print('init model-eval: ', pmodel.evaluate(x=[x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted, batch_size = 1024, verbose= 1))

                    tic = time.time()
                    pmodel.fit(x=[x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted, epochs=EPOCHS_ADJ, batch_size = hp_batchsize, callbacks=callbacks, verbose= 1)
                    times_dict['lr_{}_bz_{}'.format(lrate, hp_batchsize)] = np.round((time.time()-tic)/60)
                    

                    history = pmodel.history.history['loss']
                    # include save_format as we use a custom loss (see load_model below for futher syntax)
                    pmodel.save(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize)), save_format = 'tf')
                    np.save(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize)), history)

                elif os.path.exists(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize))):
                    pmodel = load_model(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize)), compile=False)
                    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)
                    
                    with open(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize)), 'rb') as f:
                        history = np.load(f, allow_pickle=True)

                    # pmodel_transfer.set_weights(pmodel.get_weights())
                    # pmodel_transfer.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)
                    # pmodel_transfer.save(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize)), save_format = 'tf')
                    # pmodel_transfer.summary()

                    # test = load_model(os.path.join(path_model, r'model_lr_{}_opt_{}.h5'.format(lrate, tag)), compile=False)
                    # test.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = optimizer)
                    # test.summary()
                    # exit()

                    # did exploding gradients occur during training? check loaded model
                    bool_nan_val = check_exploded_gradients(pmodel)
                    if bool_nan_val == True:
                        print('Model with nan-paramter-values! ', r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize))
                else:
                    print('train-flag is off and model with lr {} and bz {} not yet trained.'.format(lrate, hp_batchsize))
                    
                # plot training history
                plt.plot(np.arange(len(np.array(history).flatten())), np.array(history).flatten(), label='lr: {}, bz: {}'.format(lrate, hp_batchsize))
    
    # save training times
    if bool_train == True:
        with open(os.path.join(path_model, 'training_times.p'), 'wb') as fp:
            pickle.dump(times_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(path_model, 'training_times.p'), 'rb') as fp:
            times_dict = pickle.load(fp)


    # display collective plot for all training-settings loaded in loop
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path_model, 'HP_seach_lrates.png'), bbox_inches='tight')
    plt.close()
    
    # look at one individual model and visualize progress
    pmodel = load_model(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(1e-2, 64)), compile=False)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
    p_survive = pd.read_csv(os.path.join(path_data, r'DAV2008T{}}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))

    bool_grad = check_exploded_gradients(pmodel)
    if bool_grad:
        print('-------------------')
        print('NaN-parameter-values in model!')
        print('-------------------')
        ValueError


    # where did the training alter the underlying mortality-table?
    loss, _ = pmodel.evaluate([x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted,  batch_size = 1024, verbose = 0)
    plot_implied_survival_curve(pmodel, dav_table = p_survive, age_max = T_MAX, m=1, path_save = path_model, str_loss = '(loss: {})'.format(np.round_(loss,2))  )



    val_dict, true_vals = mortality_heatmap_grid(pmodel, p_survive, m=1, age_max=T_MAX, rnn_seq_len = 20, save_path= path_model)
    
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

    
