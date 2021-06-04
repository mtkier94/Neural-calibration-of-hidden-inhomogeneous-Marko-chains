import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import product as iter_prod

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorboard.plugins.hparams import api as hp # hyperparameter-tuning

T_MAX = 121 # max age in baseline survival table; used for scaling


def create_baseline_model(n_in = 2, n_out = 2, h_units = [40,40,20], h_actv = ['relu', 'relu', 'relu', 'tanh'], tf_loss_function = tf.keras.losses.KLDivergence(), optimizer = 'adam'):
    '''
    Create a baseline for n_in transition probabilities.
    Note: This is a classical feed-forward type model.

    Inputs:
    -------
        n_in:   no. of (scaled) input features to the model
        n_out:  no. of (unit-sum) outputs, i.e. transition probabilities
        width_factor:   factor for how wide hidden layers will be chosen, given n_in

    Outputs:
    --------
        tf.keras.models.Model   feed-forward model


    Note: The introduction of more than 2 output probabilities, i.e. by considering states alive, dead and disabled (and their transition probs) 
    might invoke the need for multiple (unit-sum) output-layers since the respective transition-matrix has unit-sum in its columns.
    '''

    assert(len(h_units)==len(h_actv))
    assert(len(h_units)>0)

    INPUT = Input(shape=(n_in,))
    h = Dense(units = h_units[0], activation = h_actv[0])(INPUT)
    for u_k, f_k in zip(h_units[1:], h_actv[1:]):
        if u_k>0:
            h = Dense(units = u_k, activation = f_k)(h)
    # create activation of output as explicit layer -> transfer learning for when model_base and model_res for transition-probabilities will be combined
    # i.e. eventually pred(x) = softmax(pred_base_prior_softmax(x) + pred_res_prior_softmax(x))
    # output: 1-step trans.-probs: (1/m)_p_age, (1/m)_p_age 
    #OUTPUT = Dense(units = n_out, activation= 'softmax')(h)
    h = Dense(units = 2, activation= 'linear')(h)
    OUTPUT = tf.keras.layers.Activation('softmax')(h)

    model = Model(INPUT, OUTPUT)
    model.compile(loss = tf_loss_function, metrics=['mae', 'mape'], optimizer=optimizer)

    return model

def init_HP_PARAMS():
    '''
    Initialize hyperparameters for hyperparameter-search.
    '''

    HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([40, 60, 80, 100]))
    HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([40, 60, 80, 100]))
    HP_NUM_UNITS3 = hp.HParam('num_units3', hp.Discrete([20, 40, 60, 80]))
    HP_NUM_UNITS4 = hp.HParam('num_units4', hp.Discrete([0, 20, 40, 60]))
    #HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_LR = hp.HParam('learning_rate', hp.Discrete([10**(-i) for i in range(1,6)]))
    #HP_ACTV1 = hp.HParam('activation1', hp.Discrete(['relu', 'tanh']))
    #HP_ACTV2 = hp.HParam('activation2', hp.Discrete(['relu', 'tanh']))
    #HP_ACTV3 = hp.HParam('activation3', hp.Discrete(['relu', 'tanh']))
    #HP_ACTV4 = hp.HParam('activation4', hp.Discrete(['relu', 'tanh']))
    ##HP_LOSS = hp.HParam('loss', hp.Discrete([tf.keras.losses.KLDivergence(), tf.keras.losses.MeanAbsolutePercentageError()]))
    HP_BATCH_SZ = hp.HParam('batch_size', hp.Discrete([2**i for i in [0,4, 6,7, 10]]))

    METRIC_MAE = 'mae'
    
    return HP_NUM_UNITS1, HP_NUM_UNITS2, HP_NUM_UNITS3, HP_NUM_UNITS4, HP_LR, HP_BATCH_SZ, METRIC_MAE



def train_test_model(hparams, X, Y, number_epochs= 10, n_in = 2, n_out = 2):

    '''
    Initialize, compile and train a tensorflow model.
    User this mask for a hyperparameter-search.

    Inputs:
    -------
        hparams:    Dictionary with tensorboard.plugins.hparams hyperparameters
                    keys: 
                    HP_NUMS_UNITS1, HP_NUMS_UNITS2, HP_NUMS_UNITS3,HP_NUMS_UNITS4
                    HP_ACTV1, HP_ACTV2, HP_ACTV3, HP_ACTV4 
                    HP_LOSS, , HP_LR, (HP_OPTIMIZER fixed to Adam)
                    HP_BATCH_SZ

    Outputs:
    --------
        metric:   scalar value to compare final model configurations  
    '''


    # build model
    model = create_baseline_model(n_in = n_in, n_out = n_out, h_units = [hparams[HP_NUM_UNITS1], hparams[HP_NUM_UNITS2], hparams[HP_NUM_UNITS3], hparams[HP_NUM_UNITS4]], 
                                #h_actv = [hparams[HP_ACTV1], hparams[HP_ACTV2], hparams[HP_ACTV3], hparams[HP_ACTV4]], 
                                #tf_loss_function = hparams[HP_LOSS], 
                                optimizer = tf.keras.optimizers.Adam(lr=hparams[HP_LR]))

    model.fit(X,Y, batch_size= hparams[HP_BATCH_SZ], epochs = number_epochs)

    _, mae, _ = model.evaluate(X,Y) # (loss, mae, mape)

    return mae


def run(run_dir, hparams, X, Y):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    mae = train_test_model(hparams, X=X, Y=Y)
    tf.summary.scalar(METRIC_MAE, mae, step=1)


def create_trainingdata_baseline(frequencies, surv_probs):
    '''
    Create training data (x,y) where contracts x contain the current (scaled) age in combination with payment-frequency 1/m, m in mathbb{N} and 
    y the 1/m-step transition-probabilities. We assume two states, namely alive and dead. 
    Target quantities y stem from the DAV2008Tmale survival table. Sub-annual dead-probabilities are scaled linearly, 
    i.e. {}_{1/m}q_{age} = 1/m*q_{age} and {}_{1/m}p_{age}=1-{}_{1/m}q_{age}.

    Inputs:
    -------
        frequencies:    payment frequencies 1/m, i.e. 1 (annual), 1/2 (semi-annual), etc.
        surv_table:     np.array of shape (max_age+1, 1) with annual survival probabilities, age starting at 0.
                        Note: maximum age of survival, after which survival probabilities 1/m*q_{age} are floored to zero, inferred by len(surv_table)-1

    Outputs:
    --------
        Data x,y        x,y both np.array with x.shape and y.shape (len(surv_table) x len(frequencies), 2)
    '''

    age_max = len(surv_probs)-1

    ages = np.arange(age_max+1)/T_MAX # starting from age 0
    x = np.asarray(list(iter_prod(ages,frequencies)), dtype = 'float32')
    y = np.asarray(list(iter_prod(surv_probs,frequencies)), dtype = 'float32') # adapt to x format
    print('y.shape: ', y.shape)
    y[:,1] = (1-y[:,0])*y[:,1] # death-prob: adjust for subannual steps
    y[:,0] = 1-y[:,1] # fill in surv.-prob

    return x,y


if __name__ == '__main__':

    bool_train = True

    if bool_train:
        print('------------------------------------------------')
        print('\t NOTE: baseline surv-model will be trained!')
    print('------------------------------------------------')
    print("\t GPU Available: ", tf.test.is_gpu_available())
    print('------------------------------------------------')

    cwd = os.path.dirname(os.path.realpath(__file__)) 

    p_survive = pd.read_csv(os.path.join(cwd,r'DAV2008Tmale.csv'),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    assert(T_MAX==len(p_survive)-1) # table starts at age 0

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    # data generation modulized by create_trainingdata_baseline()
    ages = np.arange(len(p_survive))/T_MAX
    x = np.asarray(list(iter_prod(ages,freqs)), dtype = 'float32')
    y = np.asarray(list(iter_prod(p_survive,freqs)), dtype = 'float32') # adapt to x format
    y[:,1] = (1-y[:,0])*y[:,1] # death-prob: adjust for subannual steps
    y[:,0] = 1-y[:,1] # fill in surv.-prob
    #x,y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive)
    x, y = shuffle(x, y)

    print('x:', x.shape, type(x))
    print('y:', y.shape, type(y))

    BATCH = 1024
    EPOCHS = 10000
    WIDTHS = [40,40,20]
    n_in = x.shape[1]
    n_out = 2


    HP_NUM_UNITS1, HP_NUM_UNITS2, HP_NUM_UNITS3, HP_NUM_UNITS4, HP_LR, HP_BATCH_SZ, METRIC_MAE = init_HP_PARAMS()

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_NUM_UNITS3, HP_NUM_UNITS4, HP_LR, #HP_ACTV1, HP_ACTV2, HP_ACTV3, HP_ACTV4, HP_LOSS, 
                    HP_BATCH_SZ],
            metrics=[hp.Metric(METRIC_MAE, display_name='mae')],
        ) 
    
    session_num = 0
    for num1 in HP_NUM_UNITS1.domain.values:
        for num2 in HP_NUM_UNITS2.domain.values:
            for num3 in HP_NUM_UNITS3.domain.values:
                for num4 in HP_NUM_UNITS4.domain.values:
                    for lr in HP_LR.domain.values:
                        #for loss in HP_LOSS.domain.values:
                        for batch_sz in HP_BATCH_SZ.domain.values:
                            hparams = {
                                HP_NUM_UNITS1: num1,
                                HP_NUM_UNITS2: num2,
                                HP_NUM_UNITS3: num3,
                                HP_NUM_UNITS4: num4,
                                HP_LR: lr,
                                #HP_LOSS: loss,
                                HP_BATCH_SZ: batch_sz
                            }

                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run('logs/hparam_tuning/' + run_name, hparams, X=x, Y=y) # <--- run model with set of hyperparameters
                            session_num += 1







    # #x_range = np.arange(len(p_survive)).reshape((-1,1))
    # plt.plot(x[:,0]*T_MAX, y[:,0], 'xr')
    # plt.plot(x[:,0]*T_MAX, model.predict(x)[:,0], 'ob', alpha = .2)
    # plt.ylabel('log(survival prob)')
    # plt.yscale('log')
    # plt.show()
    # plt.plot(x[:,0]*T_MAX, y[:,1], 'xr')
    # plt.plot(x[:,0]*T_MAX, model.predict(x)[:,1], 'om', alpha = .2)
    # plt.ylabel('log(death prob)')
    # plt.yscale('log')
    # plt.show()    
    
    