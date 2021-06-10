import numpy as np
from numba import njit, prange
import pandas as pd
import multiprocessing
import signal
import os, time
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf


from functions.sub_actuarial import get_CFs_vectorized, predict_contract_vectorized
from functions.sub_visualization import mortality_ffn_heatmap

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_models_baseline_transfer

if __name__ ==  '__main__':

    strategy = tf.distribute.MirroredStrategy()
    baseline_sex = 'female'

    #### load data
    with open(os.path.join(path_data, 'Data.npy'), 'rb') as f:
        data = np.load(f)
    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    assert(T_MAX==len(p_survive)-1) # table starts at age 0

    #with strategy.scope():
    pmodel = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer,r'survival_baseline.h5'))

    #####################
    # Plot heatmap: mortalities implied by DAV-table - mortalities of pmodel
    if False:
        mortality_ffn_heatmap(pmodel, p_survive, m=1, age_max= T_MAX)
        exit()
    #####################

    # determine batches of equal length in data
    eff_lengths, counts = np.unique(data[:,1]/data[:,3], return_counts=True)
    N_unique_steps = len(eff_lengths)

    # value computation
    vals = np.zeros((len(data),1))
    index_computed = np.zeros((len(data),), dtype = 'bool')
    CFs = [None]*N_unique_steps

    # batch wise computation
    for it, (len_eff, count) in enumerate(zip(eff_lengths, counts)):
        # if it == 30:
        #     break
        
        index = (data[:,1]/data[:,3]==len_eff)
        index_computed[index]= True
        print('Iteration {} of {}'.format(it, N_unique_steps))
        print('\t Size of batch: ', count)
        x = data[index,:]
        tic = time.time()
        # Note: predict_contract_vectorized() uses raw/non-scaled data x (!!!)
        vals[index], CFs[it] = predict_contract_vectorized(x, pmodel, discount = GAMMA, bool_print_shapes=False, age_scale=T_MAX)
        toc = time.time()
        print('\t Computation time: ', np.round(toc-tic,2), ' sec.')
    
    #---------------------------
    # backtest: do CFs-values agree with saved data y_ts (-> to be used for rnn-model structures)
    with open(os.path.join(path_data, 'CFs_backtest.npy'), 'wb') as f:
        np.save(f, CFs)

    with open(os.path.join(path_data, 'y_ts.npy'), 'rb') as f:
        y_ts = np.load(f, allow_pickle=True)

    # check if provided data is consistent
    print('Do the computed cash-flows CFs match the saved data y_ts.npy?')
    # print('\t', sum([(CFs[k]==y_ts[k]).all() for k in range(N_unique_steps)]) == N_unique_steps)
    for k in range(N_unique_steps):
        print((CFs[k]==y_ts[k]).all()) # all elements equal?
    #---------------------------

    ################ illustrate results ######################

    # Plot 1: raw (discounted & mortality-weighted) values, aggregated at time 0
    plt.scatter(range(len(vals[index_computed,:])), vals[index_computed,:])
    plt.xlabel('data point')
    plt.ylabel('value')
    plt.show()


    #  create DataFrame with quantities of interest for simplified illustration using seaborn
    iter_vs_val = np.concatenate([data[index_computed, 1:2]/data[index_computed, 3:4], 
                                vals[index_computed,:], vals[index_computed,:]/data[index_computed,-2].reshape((-1,1)), 
                                vals[index_computed,:]/data[index_computed,-1].reshape((-1,1))], axis=1)
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
    plt.scatter(range(len(vals[index_computed,:])), vals[index_computed,:]/data[index_computed,-2:-1])
    plt.xlabel('data point')
    plt.ylabel('value/sum_insured')
    plt.show()



#### NOTE: custom loss computation
# https://stackoverflow.com/questions/41132633/can-numba-be-used-with-tensorflow

    
