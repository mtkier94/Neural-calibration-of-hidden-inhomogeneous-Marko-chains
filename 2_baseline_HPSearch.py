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
from functions.tf_model_base import create_train_save_model_base

from global_vars import T_MAX # max age in baseline survival table; used for scaling
from global_vars import path_models_baseline_hpsearch, path_data, path_models_baseline_plots

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

    ###-----------------------------------------------------------------------------------------------------------------
    # Optional HPSearch - Standard choice in 2a_main seems to be sufficient
    # Fine-tuning to respect customer segments will be performed with joint model either way
    ###-----------------------------------------------------------------------------------------------------------------
    #assert False, 'HPSearch turned off for now'

    baseline_sex = 'male'
    bool_train = False

    if bool_train:
        print('------------------------------------------------')
        print('\t NOTE: baseline surv-model will be trained!')
    print('------------------------------------------------')
    print('\t tensorflow-version: ', tf.__version__)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\t tensorflow-warnings: off')
    #print("\t GPU Available: ", tf.test.is_gpu_available()) # <--- enable if GPU devices are (expected to be) available; disabled for local CPU pre-training
    print('------------------------------------------------')

    #cwd = os.path.dirname(os.path.realpath(__file__)) 
    #path_models_base = os.path.join(cwd, 'models_base')

    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    assert(T_MAX==len(p_survive)-1) # table starts at age 0

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    # data generation modulized by create_trainingdata_baseline()
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    x, y = shuffle(x, y)


    BATCH = 8 #min(1024, len(x))    
    LRATE = 0.001
    EPOCHS = 20000
    WIDTHS = [40,40,20]
    n_in = x.shape[1]
    n_out = 2
     

    HP_UNITS = [[40,40,20]]#, [20,20,10] [60,60,40], [80,80,60], [80,80,60, 60], [80,80,60, 20]]
    HP_LR = [5e-2, 1e-2, 5e-3, 1e-3]
    HP_BATCH_SZ = [2**k for k in [3,4,6,10]]
    HP_LOSS = ['mae', tf.keras.losses.KLDivergence()]

    models_dict = {}
    histories_dict = {}

    for hidden_units in HP_UNITS:
        for learning_rate in HP_LR:
            for batch_sz in HP_BATCH_SZ:
                # train model
                # Note: bool_train = False -> load trained model, if available on path_models_base
                if len(hidden_units)==3:
                    hidden_act = ['relu', 'relu', 'tanh']
                elif len(hidden_units)==4:
                    hidden_act = ['relu', 'relu', 'relu', 'tanh']
                else:
                    raise ValueError
                model, hist = create_train_save_model_base(X=x, Y = y, h_units = hidden_units, learning_rate = learning_rate, epochs = EPOCHS, batch_sz = batch_sz, 
                                                        path_save = path_models_baseline_hpsearch, bool_train = bool_train, act_func= hidden_act, n_in = 2, n_out = 2)
                models_dict['hu: {}, lr: {}, ba {}'.format(hidden_units,learning_rate,batch_sz)], histories_dict['hu: {}, lr: {}, ba {}'.format(hidden_units,learning_rate,batch_sz)] = model, hist


    # visualize difference in HPs
    default_hu = HP_UNITS[0]
    default_lr = HP_LR[2]
    default_ba = HP_BATCH_SZ[2]

    # plot learning rates
    _, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for lr in HP_LR: 
        ax[0].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, default_ba)]['loss'], label= 'lr: {}'.format(lr))
        ax[1].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, default_ba)]['mae'], label= 'lr: {}'.format(lr))
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].set_ylabel('mae')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.title('hu: {}, ba: {}'.format(default_hu, default_ba))
        plt.legend()
    plt.savefig(os.path.join(path_models_baseline_plots, 'HP_seach_learning_rate_ba_{}.png'.format(default_ba)), bbox_inches='tight')
    plt.show()

    # plot learning rates
    _, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for lr in HP_LR: 
        ax[0].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, HP_BATCH_SZ[3])]['loss'], label= 'lr: {}'.format(lr))
        ax[1].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, HP_BATCH_SZ[3])]['mae'], label= 'lr: {}'.format(lr))
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].set_ylabel('mae')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.title('hu: {}, ba: {}'.format(default_hu, HP_BATCH_SZ[3]))
        plt.legend()
    plt.savefig(os.path.join(path_models_baseline_plots, 'HP_seach_learning_rate_ba_{}.png'.format(HP_BATCH_SZ[3])), bbox_inches='tight')
    plt.show()

    # plot batch-sizes
    _, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for ba in HP_BATCH_SZ:
        ax[0].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,default_lr, ba)]['loss'], label= 'ba: {}'.format(ba))
        ax[1].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,default_lr, ba)]['mae'], label= 'ba: {}'.format(ba))
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].set_ylabel('mae')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.legend()
        plt.title('hu: {}, lr: {}'.format(default_hu, default_lr))
    plt.savefig(os.path.join(path_models_baseline_plots, 'HP_seach_batch_size.png'), bbox_inches='tight')
    plt.show()

    # plot hidden-units
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for hu in HP_UNITS:
        ax[0].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(hu, default_lr, default_ba)]['loss'], label= 'ba: {}'.format(hu))
        ax[1].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(hu, default_lr, default_ba)]['mae'], label= 'ba: {}'.format(hu))
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].set_ylabel('mae')
        ax[1].set_yscale('log')
        ax[0].set_yscale('log')
        plt.legend()
        plt.title('lr: {}, ba: {}'.format(default_lr, default_ba))
    plt.savefig(os.path.join(path_models_baseline_plots, 'HP_seach_hidden_units.png'), bbox_inches='tight')
    plt.show()   