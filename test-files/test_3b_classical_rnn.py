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
# from functions.tf_model_res import create_mortality_res_net, combine_base_and_res_model
# from functions.sub_visualization import mortality_rnn_heatmap


from global_vars import GAMMA
from global_vars import path_data, path_data_backtesting, path_models_baseline_transfer


if __name__ ==  '__main__':

    # speed-up by setting mixed-precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    #strategy = tf.distribute.MirroredStrategy()

    #### load data
    with open(os.path.join(path_data_backtesting, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'y_ts.npy'), 'rb') as f:
        y_ts = np.load(f, allow_pickle=True)

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
    pmodel_base_rnn = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'survival_baseline_ts.h5'))
    pmodel_base_rnn.compile(loss = compute_loss_raw, metrics=['mae'], optimizer = 'adam')

    p_survive = pd.read_csv(os.path.join(path_data, r'DAV2008Tmale.csv'),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))

    # computation via classical approach
    # batch wise computation
    vals = np.zeros((N_contracts,1))
    index = 0
    for it in range(len(x_ts)):
        print('Iteration {} of {}'.format(it, len(x_ts)))
        print('\t Size of batch: ', x_ts[it].shape)
        len_seq = x_ts[it].shape[0]
        tic = time.time()
        # Note: predict_contract_vectorized() uses raw/non-scaled data x (!!!)
        vals[index:index+len_seq] = predict_rnn_contract_vectorized(x_ts[it], y_ts[it], pmodel=pmodel_base_rnn, discount = GAMMA, bool_print_shapes=False)
        toc = time.time()
        print('\t Computation time: ', np.round(toc-tic,2), ' sec.')
        index+= len_seq
    

    # illustrate results
    # note: contracts are clustering w.r.t. their "effective number of iterations"
    #       hence, contracts are ordered differently than in the original data and displayed differently than in MAIN.py

    # Plot 1: raw (discounted & mortality-weighted) values, aggregated at time 0
    plt.scatter(range(N_contracts), vals[:,:])
    plt.xlabel('data point')
    plt.ylabel('value')
    plt.show()

    # Plot 2: Create DataFrame with quantities of interest for simplified illustration using seaborn
    # format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    iter_vs_val = np.concatenate([data[:, 1:2]/data[:, 3:4], 
                                vals, vals/data[:,-2].reshape((-1,1)), 
                                vals/data[:,-1].reshape((-1,1))], axis=1)
    df = pd.DataFrame(data = iter_vs_val, columns = ['iterations', 'value', 'rel_sum', 'rel_prem'] )


    # PLot 2
    plt.scatter(iter_vs_val[:,0], iter_vs_val[:,1])
    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.show()

    # Plot 3
    sns.lineplot(data = df, x = 'iterations', y = 'value')
    plt.show()

    # Plot 4
    sns.lineplot(data = df, x = 'iterations', y = 'rel_sum')
    plt.show()

    # Plot 5
    sns.lineplot(data = df, x = 'iterations', y = 'rel_prem')
    plt.show()

    # Plot 6
    plt.scatter(range(N_contracts), vals/data[:,-2:-1])
    plt.xlabel('data point')
    plt.ylabel('value/sum_insured')
    plt.show()