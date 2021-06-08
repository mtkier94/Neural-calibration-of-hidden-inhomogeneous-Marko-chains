import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import product as iter_prod
from time import time

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorboard.plugins.hparams import api as hp # hyperparameter-tuning

from functions.sub_data_prep import create_trainingdata_baseline
from functions.tf_model_base import create_train_save_model_base, create_baseline_model_ffn
from functions.tf_tools import LRFind

from global_vars import T_MAX # max age in baseline survival table; used for scaling

from global_vars import path_models_baseline_transfer, path_data


def lr_schedule(epoch, lr):
        '''
        Custom learning rate schedule.
        '''
        if (epoch>=1000) and (epoch % 250==0):
            return lr*0.9
        else:
            return lr

def ES():
    return tf.keras.callbacks.EarlyStopping(monitor='mape', patience=10000, restore_best_weights=True)




if __name__ == '__main__':

    bool_train = True
    baseline_sex = 'female'
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule)]

    if bool_train:
        print('------------------------------------------------')
        print('\t NOTE: baseline surv-model will be trained!')
    print('------------------------------------------------')
    print('\t tensorflow-version: ', tf.__version__)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\t tensorflow-warnings: off')
    #print("\t GPU Available: ", tf.test.is_gpu_available()) # <--- enable if GPU devices are (expected to be) available; disabled for local CPU pre-training
    print('------------------------------------------------')

    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    assert(T_MAX==len(p_survive)-1) # table starts at age 0

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    # data generation modulized by create_trainingdata_baseline()
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    x, y = shuffle(x, y)

    BATCH = 8 # 1024 #min(1024, len(x))    
    LRATE = 0.005
    EPOCHS = 40000
    WIDTHS = [40,40,20]#[20,20,20] #
    n_in = x.shape[1]
    n_out = 2    

    if bool_train:
        model = create_baseline_model_ffn(n_in=n_in, n_out=n_out, h_units=WIDTHS, h_actv= ['relu']*(len(WIDTHS)-1)+['tanh'], tf_loss_function = tf.keras.losses.KLDivergence(),#'mae', #
                                    optimizer=tf.keras.optimizers.Adam(lr=0.005))
        model.fit(x, y, batch_size=BATCH, epochs= EPOCHS, verbose=1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule), ES()])
        history = np.stack([np.array(model.history.history['loss']), np.array(model.history.history['mae'])], axis = 0) #np.array(model.history.history, ndmin=2)
        
        model.save(os.path.join(path_models_baseline_transfer, r'ffn_davT{}.h5'.format(baseline_sex)))    
        np.save(os.path.join(path_models_baseline_transfer  , r'hist_{}.npy'.format(baseline_sex)), history)                                  
    else:
        # model = load_model(path_models_baseline_transfer, r'survival_baseline_{}.h5'.format(baseline_sex))
        model = load_model(os.path.join(path_models_baseline_transfer,  r'ffn_davT{}.h5'.format(baseline_sex)))
        history = np.load(os.path.join(path_models_baseline_transfer  , r'hist_{}.npy'.format(baseline_sex)), allow_pickle= True)#.item()
    
    print(history.shape)

    _, ax = plt.subplots(1,2,figsize=(10,4))
    if type(history) == type({}):
        ax[0].plot(history['loss'])
        ax[1].plot(history['mae'])
    else:
        ax[0].plot(history[0])
        ax[1].plot(history[1])
    ax[0].set_ylabel('loss')
    ax[1].set_ylabel('mae')
    ax[0].set_xlabel('iteration')
    ax[1].set_xlabel('iteration')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    plt.tight_layout()
    plt.show()


    #x_range = np.arange(len(p_survive)).reshape((-1,1))
    # plt.plot(x[:,0]*T_MAX, y[:,0], 'xr')
    # plt.plot(x[:,0]*T_MAX, model.predict(x)[:,0], 'ob', alpha = .2)
    # plt.ylabel('log(survival prob)')
    # plt.yscale('log')
    # plt.show()
    plt.plot(x[:,0]*T_MAX, y[:,1], 'xr')
    plt.plot(x[:,0]*T_MAX, model.predict(x)[:,1], 'om', alpha = .2)
    plt.ylabel('log(death prob)')
    plt.yscale('log')
    plt.show()    