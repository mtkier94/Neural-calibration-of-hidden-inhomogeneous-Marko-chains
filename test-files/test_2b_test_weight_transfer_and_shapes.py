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


from functions.tf_loss_custom import compute_loss_mae, compute_loss_raw
from functions.sub_backtesting import predict_contract_backtest, check_if_rnn_version

# discount factor
from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_data_backtesting, path_models_baseline_transfer


if __name__ == '__main__':

    baseline_sex = 'female'
    cwd = os.path.dirname(os.path.realpath(__file__))
    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))


    # load data
    # 1) contract data (raw)
    with open(os.path.join(path_data, 'Data.npy'), 'rb') as f:
        data = np.load(f)

    # 2) (x,y) -> processed, sequence data with contracts (x) and CFs (y)
    with open(os.path.join(path_data_backtesting, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data_backtesting, 'y_ts.npy'), 'rb') as f:
        y_ts = np.load(f, allow_pickle=True)


    # 3) mortality models - FFN and RNN versions
    pmodel_base_rnn = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))
    pmodel_base_rnn.compile(loss = compute_loss_raw, metrics=['mae'], optimizer = 'adam')

    pmodel_base_ffn = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'ffn_davT{}.h5'.format(baseline_sex)))
    pmodel_base_ffn.compile(loss = compute_loss_raw, metrics=['mae'], optimizer = 'adam')


    # are ffn and rnn models of equal training-versions?
    bool_lst = check_if_rnn_version(model_ffn=pmodel_base_ffn, model_rnn=pmodel_base_rnn)
    # print(bool_lst) # optional: show list to see - if False - which layers are different
    print('-----------------------------------------------------------------')
    print('The FFN and RNN models have equal weights: ', (bool_lst==True).all())
    print('-----------------------------------------------------------------')
    assert (bool_lst==True).all()


    ###### loop over data in batches with equal effective lengths
    # determine batches of equal length in data
    eff_lengths, counts = np.unique(data[:,1]/data[:,3], return_counts=True)
    N_unique_steps = len(eff_lengths)

    # value computation
    vals = np.zeros((len(data),1))
    index_computed = np.zeros((len(data),), dtype = 'bool')
    CFs = [None]*N_unique_steps

    # batch wise computation
    for it, (len_eff, count) in enumerate(zip(eff_lengths, counts)):

        index = (data[:,1]/data[:,3]==len_eff)
        index_computed[index]= True
        print('Iteration {} of {}'.format(it, N_unique_steps))
        print('\t Size of batch: ', count)
        x = data[index,:]

        tic = time.time()
        # compare CFs and predicted probabilities for vanilla FFN and (basically equal) RNN model
        # quantities, shapes, etc. should (approx. withing numpy precision) be equal -> predict_contract_backtest contains many assert( .. ) statements to test just that
        vals[index] = predict_contract_backtest(x, x_ts[it], y_ts[it], pmodel_ffn=pmodel_base_ffn, pmodel_rnn=pmodel_base_rnn, discount=GAMMA, age_scale = T_MAX, bool_print_shapes = False)
        toc = time.time()
        print('\t Computation time: ', np.round(toc-tic,2), ' sec.')

        if it == 20:
            break



    




    