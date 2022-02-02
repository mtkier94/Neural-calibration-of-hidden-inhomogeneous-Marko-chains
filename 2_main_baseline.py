import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# set some plotting parameters globally
parameters = {'axes.labelsize': 16, 'xtick.labelsize':14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 16, 'figure.titlesize': 18}
plt.rcParams.update(parameters)


from sklearn.utils import shuffle
from itertools import product as iter_prod
from time import time

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorboard.plugins.hparams import api as hp # hyperparameter-tuning

from functions.sub_data_prep import create_trainingdata_baseline
from functions.tf_model_base import create_baseline_model_ffn, create_baseline_model_rnn, transfer_weights_dense2simpleRNN
from functions.sub_backtesting import check_if_rnn_version

from global_vars import T_MAX # max age in baseline survival table; used for scaling

from global_vars import path_models_baseline_transfer, path_data


def lr_schedule(epoch, lr):
        '''
        Custom learning rate schedule.
        Note: Too rapids decay has shown to harm the quality of prediction, 
        particularly for low ages where we see high, relative but low 
        absolute differences in death probability
        '''
        if (epoch>=10000) and (epoch % 500==0):
            return lr*0.9
        else:
            return lr

def ES():
    return tf.keras.callbacks.EarlyStopping(monitor='mape', patience=10000, restore_best_weights=True)



def run_main(baseline_sex, bool_train):

    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule), ES()]
    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    assert(T_MAX==len(p_survive)-1) # table starts at age 0

    if baseline_sex == 'female':
        p_other_sex = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format('male')),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))
        tag_other_sex = 'DAVT2008male'
    elif baseline_sex == 'male':
        p_other_sex = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format('female')),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))
        tag_other_sex = 'DAVT2008female'
    else:
        raise ValueError('Unknown baseline_sex')
    
            
    print('\t tensorflow-version: ', tf.__version__)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\t tensorflow-warnings: off')
    #print("\t GPU Available: ", tf.test.is_gpu_available()) # <--- enable if GPU devices are (expected to be) available; disabled for local CPU pre-training
    print('------------------------------------------------')

    

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    # data generation modulized by create_trainingdata_baseline()
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    x, y = shuffle(x, y)

    BATCH = 32 # 1024 #min(1024, len(x))    
    LRATE = 0.001
    EPOCHS = 30000
    WIDTHS = [40,40,20]#[20,20,20] #
    n_in = x.shape[1]
    n_out = 2    

    if bool_train:

        print('------------------------------------------------')
        print('\t NOTE: baseline surv-model will be trained!')
        print('\t This is currently manually disabled, as it will require a full recallibration of the subsequent, residual network.')
        print('\t To activate a retraining of the baseline model, uncomment ValueError in subsequent line.')
        raise ValueError('Training of baseline model deactivated!')
        print('------------------------------------------------')


        model = create_baseline_model_ffn(n_in=n_in, n_out=n_out, h_units=WIDTHS, h_actv= ['relu']*(len(WIDTHS)-1)+['tanh'], 
                                    tf_loss_function = tf.keras.losses.KLDivergence(),#'mae', #'mean_squared_logarithmic_error', #
                                    optimizer=tf.keras.optimizers.Adam(lr=LRATE))
        model.fit(x, y, batch_size=BATCH, epochs= EPOCHS, verbose=1, callbacks=callbacks)
        history = np.stack([np.array(model.history.history['loss']), np.array(model.history.history['mae'])], axis = 0) #np.array(model.history.history, ndmin=2)
        
        model.save(os.path.join(path_models_baseline_transfer, r'ffn_davT{}.h5'.format(baseline_sex)))    
        np.save(os.path.join(path_models_baseline_transfer  , r'hist_{}.npy'.format(baseline_sex)), history) 

        # transfer weights to rnn-type model (for later use)
        # sequential character benefitial to non-Markovian HMC-objective
        model_rnn = create_baseline_model_rnn(input_shape=(None, n_in), n_out= n_out, hidden_layers=[40,40,20])
        transfer_weights_dense2simpleRNN(dense_model= model, rnn_model = model_rnn)
        model_rnn.save(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))    
        print('Weights transferred from ffn to rnn!')

        assert (check_if_rnn_version(model_ffn=model, model_rnn=model_rnn)==True).all()
        print('FFN evaluated: ', model.evaluate(x,y, batch_size=1024, verbose = 0))
        print('RNN evaluated: ', model_rnn.evaluate(x.reshape(1,-1,n_in),y.reshape(1,-1,n_out), batch_size=1024, verbose = 0))
                               
    else:
        # model = load_model(path_models_baseline_transfer, r'survival_baseline_{}.h5'.format(baseline_sex))
        model = load_model(os.path.join(path_models_baseline_transfer,  r'ffn_davT{}.h5'.format(baseline_sex)))
        history = np.load(os.path.join(path_models_baseline_transfer  , r'hist_{}.npy'.format(baseline_sex)), allow_pickle= True)#.item()
        model_rnn = load_model(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))


        print(model.summary())
        print(model.evaluate(x,y, batch_size=1024, verbose = 0))

    # visualize training process of FNN-model
    if False:
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

    # visualize fit of FFN vs baseline-DAV-table    
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    # do not shuffle to preserve order for ploting
    for k in range(len(freqs)):
        # note: invert the iter_prod of age and freq
        plt.plot(x[k:-1:len(freqs),0]*T_MAX, y[k:-1:len(freqs),1], linestyle = '-', color = 'black')
        # plt.plot(x[k:-1:len(freqs),0]*T_MAX, y[k:-1:len(freqs),1], linestyle = '-', color = 'gray')
        plt.plot(x[k:-1:len(freqs),0]*T_MAX, model.predict(x[k:-1:len(freqs),:])[:,1], linestyle = '--', color = 'orange')
        # plt.plot(x[k:-1:len(freqs),0]*T_MAX, model.predict(x[k:-1:len(freqs),:])[:,1], linestyle = '--', color = 'green')
    plt.yscale('log')
    plt.xlabel('age') #, fontsize = 'x-large')

    # create labels
    plt.plot(x[k:-1:len(freqs),0]*T_MAX, y[k:-1:len(freqs),1], linestyle = '-', color = 'black', label = r'$\mathcal{D}_{DAV}($' + baseline_sex+r')')
    plt.plot(x[k:-1:len(freqs),0]*T_MAX, model.predict(x[k:-1:len(freqs),:])[:,1], linestyle = '--', color = 'orange', label = r'$\hat{\pi}_{base}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path_models_baseline_transfer, 'baseline_fit_{}.eps'.format(baseline_sex)), format = 'eps', dpi = 400)
    # plt.show()
    plt.close()
    
    if False:
        # visualize fit of FFN and RNN vs baseline-DAV-table
        plt.plot(x[:,0]*T_MAX, model.predict(x)[:,0], 'xg', alpha = .5, label='ffn')
        plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,0].flatten(), 'ob', alpha = .2, label='rnn')
        
        plt.plot(x[:,0]*T_MAX, y[:,0], linestyle = 'None', marker = '_', color = 'red', label='DAV')
        plt.plot(x[:,0]*T_MAX, model.predict(x)[:,1], 'xg', alpha = .5)
        plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,1].flatten(), 'ob', alpha = .2)
        
        plt.plot(x[:,0]*T_MAX, y[:,1], linestyle = 'None', marker = '_', color = 'red',)
        plt.yscale('log')
        plt.title('Fit - FFN vs. RNN vs DAVT2008{}'.format(baseline_sex))
        plt.legend()
        plt.show()


if __name__ == '__main__':

    bool_train = False
    # baseline_sex = 'male'

    for baseline_sex in ['male', 'female']: #, 'female']:

        run_main(baseline_sex, bool_train)