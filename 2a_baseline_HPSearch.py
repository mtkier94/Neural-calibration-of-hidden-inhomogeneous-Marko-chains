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
from global_vars import path_data, path_models_baseline_plots, path_models_baseline_hpsearch_male, path_models_baseline_hpsearch_female 

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

    baseline_sex = 'female'
    bool_train = False
    if baseline_sex == 'male':
        path_model = path_models_baseline_hpsearch_male
    elif baseline_sex == 'female':
        path_model = path_models_baseline_hpsearch_female
    else:
        assert False, 'Unknown Option for baseline_sex.'

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


    EPOCHS = 20000
    n_in = x.shape[1]
    n_out = 2
     

    HP_UNITS = [[40,40,20]]#, [20,20,10] [60,60,40], [80,80,60], [80,80,60, 60], [80,80,60, 20]]
    HP_LR = [5e-2, 1e-2, 1e-3]
    HP_BATCH_SZ = [2**k for k in [3,4,6]]
    HP_LOSS = ['mae', tf.keras.losses.KLDivergence()]

    models_dict = {}
    histories_dict = {}


    for loss in HP_LOSS:
        if type(loss) == type(tf.keras.losses.KLDivergence()):
            tag = 'KLDiv'
        elif loss == 'mae':
            tag = 'mae'
        else:
            raise ValueError('Unknown loss_function!')
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
                    model, hist = create_train_save_model_base(X=x, Y = y, h_units = hidden_units, learning_rate = learning_rate, epochs = EPOCHS, 
                                                        batch_sz = batch_sz, path_save = path_model, bool_train = bool_train, loss_function= loss,
                                                        act_func= hidden_act, callbacks= [tf.keras.callbacks.LearningRateScheduler(lr_schedule), ES()], 
                                                        n_in = 2, n_out = 2)
                    models_dict['hu {}, lr {}, bz {}, loss {}'.format(hidden_units,learning_rate,batch_sz, tag)], histories_dict['hu {}, lr {}, bz {}, loss {}'.format(hidden_units,learning_rate,batch_sz, tag)] = model, hist

                    plt.plot(x[:,0]*T_MAX, y[:,1], 'xr')
                    plt.plot(x[:,0]*T_MAX, model.predict(x)[:,1], 'om', alpha = .2)
                    plt.ylabel('log(death prob)')
                    plt.yscale('log')
                    plt.title('hu {}, lr {}, bz {}, loss {}'.format(hidden_units,learning_rate,batch_sz, tag))
                    plt.show() 


    # visualize results
    _, ax = plt.subplots(2,2,figsize=(12,8))
    for lr in HP_LR:
        for bz in HP_BATCH_SZ:
            for hu in HP_UNITS:
                for k, loss in enumerate(['KLDiv', 'mae']):
                    # Note row of histories: 1) loss, 2) mae, 3) mape
                    ax[k,0].plot(histories_dict['hu {}, lr {}, bz {}, loss {}'.format(hu, lr, bz, loss)][1], label = 'lr {}, bz {}'.format(lr, bz)) 
                    ax[k,0].set_ylabel('mae'), ax[k, 0].set_xlabel('epoch')
                    ax[k,0].set_yscale('log')
                    ax[k,0].set_title(f'training with {loss}')
                    ax[k,1].plot(histories_dict['hu {}, lr {}, bz {}, loss {}'.format(hu, lr, bz, loss)][2], label = 'lr {}, bz {}'.format(lr, bz)) 
                    ax[k,1].set_ylabel('KL-Div.'), ax[k, 1].set_xlabel('epoch')
                    ax[k,1].set_yscale('log')
                    ax[k,1].set_title(f'training with {loss}')

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(path_model, 'HPSearch.png'))
    plt.show()

    # save model and hist


    exit()

    # visualize difference in HPs
    default_hu = HP_UNITS[0]
    default_lr = HP_LR[2]
    default_ba = HP_BATCH_SZ[2]
    default_loss = HP_LOSS[0]

    # plot learning rates
    _, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for lr in HP_LR: 
        ax[0].plot(histories_dict['hu {}, lr {}, bz {}, loss {}'.format(default_hu,lr, default_ba, default_loss)][0], label= 'lr: {}'.format(lr))
        ax[1].plot(histories_dict['hu {}, lr {}, bz {}, loss {}'.format(default_hu,lr, default_ba, default_loss)][1], label= 'lr: {}'.format(lr))
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].set_ylabel('mae')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        plt.title('hu: {}, ba: {}, loss: {}'.format(default_hu, default_ba, default_loss))
        plt.legend()
    plt.savefig(os.path.join(path_models_baseline_plots, 'HP_seach_learning_rate_ba_{}.png'.format(default_ba)), bbox_inches='tight')
    plt.show()

    # plot learning rates
    _, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
    for lr in HP_LR: 
        ax[0].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, HP_BATCH_SZ[3])][0], label= 'lr: {}'.format(lr))
        ax[1].plot(histories_dict['hu: {}, lr: {}, ba {}'.format(default_hu,lr, HP_BATCH_SZ[3])][1], label= 'lr: {}'.format(lr))
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