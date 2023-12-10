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
from tensorflow.keras.models import Model


from functions.sub_actuarial import predict_rnn_contract_vectorized
from functions.tf_loss_custom import compute_loss_raw



from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_data_backtesting, path_models_baseline_transfer


if __name__ ==  '__main__':

    baseline_sex = 'female'
    # speed-up by setting mixed-precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    cwd = os.path.dirname(os.path.realpath(__file__))
    #strategy = tf.distribute.MirroredStrategy()

    #### load data
    with open(os.path.join(path_data_backtesting, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'y_ts.npy'), 'rb') as f:
        y_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'y_ts_discounted.npy'), 'rb') as f:
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


    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer,r'rnn_davT{}.h5'.format(baseline_sex)))
    pmodel_base.compile(loss = compute_loss_raw, metrics=['mae'], optimizer = 'adam')
    # Note: To simplify debugging (optional), add run_eagerly = True flag to compiler
    # Downside of run_eagerly = True: Disabling graph-mode results in much slower tf-operations

    p_survive = pd.read_csv(os.path.join(path_data, r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))


    # display baseline loss - for individual contracts
    # note: evaluate_loss with tf.reduce_mean(...,axis=0), i.e. vector-valued output of size batch_size
    loss_init = np.zeros((N_batches,1))
    loss_classical = np.zeros((N_batches,1))
    mae_init = np.zeros((N_batches,1))
    it_steps = np.zeros((N_batches,1))


    ###############################################################################
    #
    # Important: Classical computation takes discounting GAMMA as input
    #            tf.loss only takes y_pred and y_true -> include GAMMA in y_true   
    #
    ###############################################################################

    # computation via tf-loss-function
    tic = time.time()
    for k, (x_val, y_val) in enumerate(zip(x_ts,y_ts)):

        len_batch = x_val.shape[0]
        number_steps = x_val.shape[1]
        print('Batch {}/{} with shape: '.format(k+1, N_batches), x_val.shape)
        # include discounting as a hyperparameter affecting the target values (CFs) only
        discount = GAMMA**(x_val[:,:,3:4]*np.arange(number_steps).reshape(1,-1, 1))
        assert(discount.shape == (len_batch, number_steps, 1))
        target = y_val*discount

        assert((target==y_ts_discounted[k]).all())

        loss, _ = pmodel_base.evaluate(x_val[:,:,base_features], target, verbose = 0)
        print('\t tf-loss: ', loss)

        val_classical = predict_rnn_contract_vectorized(x_val, y_val, pmodel=pmodel_base, discount = GAMMA, bool_print_shapes=False).mean()
        print('\t classical value: ', val_classical)
        loss_init[k], loss_classical[k] = loss, val_classical
        it_steps[k] = number_steps

    print('Overall computation time: ', np.round(time.time()-tic,2), ' sec.')
    plt.plot(it_steps, loss_init, label = 'tf.loss') # aggregated avg. loss per batch
    plt.plot(it_steps, loss_classical,linestyle = "None",  marker='x', label = 'numpy')
    plt.xlabel('#iterations for full (hidden) Markov Chain computation.')
    plt.ylabel('mean loss - discounted, weighted CFs')
    plt.legend()
    #plt.yscale('log')
    plt.show()