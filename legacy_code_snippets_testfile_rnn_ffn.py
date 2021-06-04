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


from mortality import T_MAX
from sub_functions import get_CFs, get_CFs_vectorized, predict_contract, predict_contract_vectorized, predict_rnn_contract_vectorized
from sub_tf_functions import compute_loss, evaluate_loss_elementwise, compute_loss_new, create_mortality_res_net, combine_base_and_res_model
from sub_visualization import mortality_rnn_heatmap

GAMMA = 1/1.02


if __name__ ==  '__main__':

    # speed-up by setting mixed-precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # speed-up by XLA: fusing Graph-kernels
    # TF_XLA_FLAGS = -tf_xla_auto_jit = 1 # note applicable for variable input-shape models (!!)

    # TO-DO: Include Threading for speed-up
    # Nvidia recommendation: -> check applicability
    # import multiprocessing
    # config.intra_op_parallelism_threads = 1
    # config.intra_op_parallelism_threads = max(2, (multiprocessing.cpu_count()//hvd.size())-1)

    # # GPU Threads
    # TF_GPU_THREAD_MODE = gpu_private
    # GF_GPU_THREAD_COUNT = 1




    #print('Tensorflow version: ', tf.__version__)

    cwd = os.path.dirname(os.path.realpath(__file__))
    #strategy = tf.distribute.MirroredStrategy()

    #### load data
    with open('x_ts.npy', 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open('y_ts.npy', 'rb') as f:
        y_ts = np.load(f, allow_pickle=True)

    print('---------------------------------------------------')
    print('x_ts: ', x_ts.shape)
    print('y_ts: ', y_ts.shape)
    print('x_ts[0].shape: ', x_ts[0].shape)
    print('y_ts[0].shape: ', y_ts[0].shape)
    N_batches = len(x_ts)
    print('number of batches: ', N_batches)

    N_contracts = 0
    for k in range(N_batches):
        N_contracts += len(x_ts[k])

    print('number of contracts: ', N_contracts)
    print('---------------------------------------------------')


    # test shapes of input/ target data
    if False:
        for i in range(N_batches):
            print('Batch {} with x_ts/y_ts shapes of '.format(i), x_ts[i].shape, ' and ', y_ts[i].shape)

        exit()



    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]


    #with strategy.scope():
    pmodel_base = tf.keras.models.load_model(os.path.join(cwd,r'survival_baseline_ts.h5'))

    p_survive = pd.read_csv(os.path.join(cwd,r'DAV2008Tmale.csv'),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    
    if False:
        mortality_rnn_heatmap(pmodel_base, p_survive, m=1, age_max=T_MAX, rnn_seq_len=10, save_path=None)
    #exit()

    
    pmodel_res = create_mortality_res_net(hidden_layers = [40,40,20], n_in= len(res_features), n_out=2)

    # display baseline loss - for individual contracts
    # note: evaluate_loss with tf.reduce_mean(...,axis=0), i.e. vector-valued output of size batch_size
    pmodel_base.compile(loss = compute_loss_new, metrics=['mae'], optimizer = 'adam')
    loss_init = np.zeros((N_batches,1))
    mae_init = np.zeros((N_batches,1))
    it_batch = np.zeros((N_batches,1))
    # loss = np.zeros((N_contracts,1))
    # it_batch = np.zeros((N_contracts,1))

    ###########################################################
    #
    # Important: Plug in discounting in modeling procedure
    #
    ###########################################################

    # computation via classical approach
    # batch wise computation
    vals = np.zeros((N_contracts,1))
    #index_computed = np.zeros((len(data),), dtype = 'bool')
    index = 0
    for it in range(len(x_ts)):
        # if it == 50:
        #     break

        print('Iteration {} of {}'.format(it, len(x_ts)))
        print('\t Size of batch: ', x_ts[it].shape)
        len_seq = x_ts[it].shape[0]
        print(x_ts[it][0:5,:])
        exit()
        tic = time.time()
        # Note: predict_contract_vectorized() uses raw/non-scaled data x (!!!)
        vals[index:index+len_seq] = predict_rnn_contract_vectorized(x_ts[it], y_ts[it], pmodel=pmodel_base, discount = GAMMA, bool_print_shapes=True)
        toc = time.time()
        print('\t Computation time: ', np.round(toc-tic,2), ' sec.')
        index+= len_seq
    

    # illustrate results

    # Plot 1: raw (discounted & mortality-weighted) values, aggregated at time 0
    plt.scatter(range(len(vals[:,:])), vals[:,:])
    plt.xlabel('data point')
    plt.ylabel('value')
    plt.show()

    exit()



    # computation via tf-loss-function
    if True:
        for k, (x_val, y_val) in enumerate(zip(x_ts,y_ts)):
            # if k==0:
            #     pass
            # else:
            len_batch = x_val.shape[0]
            print('batch of shape: ', x_val.shape, y_val.shape)
            loss, _ = pmodel_base.evaluate(x_val[:,:,base_features], y_val, verbose = 0)
            print('batch {}, loss (overall/ average): '.format(k), loss, ' resp. ', loss/len_batch)
            #print('x_ts[k][:,0,:].shape: ', x_ts[k][:,0,:].shape)
            #print('Effective length: ', x_ts[k][0:10,0,1], x_ts[k][0:10,0,3])
            #print('Classical computation: ', predict_contract_vectorized(x_ts[k][:,0,:], pmodel_base, discount = GAMMA))
            #loss[index:index+len_batch] = cache
            loss_init[k] = loss
            mae_init[k] = loss/len_batch
            it_batch[k] = len_batch
            #index += len_batch
            # if k == 3:
            #     exit()
        plt.scatter(it_batch, loss_init, label='agg. loss') # aggregated loss (?) per batch
        plt.scatter(it_batch, mae_init, label = 'avg. loss') # average loss per batch
        plt.xlabel('#iterations for full Markov Chain computation.')
        plt.ylabel('mae - discounted, weighted CFs')
        plt.yscale('log')
        plt.legend()
        plt.show()

    exit()

    # display baseline loss - per batch
    # note: compute_loss with tf.reduce_mean across all axis, i.e. axis = None
    pmodel_base.compile(loss = compute_loss, metrics=['mae'], optimizer = 'adam')
    loss_init = np.zeros((N_batches,1))
    it_batch = np.zeros((N_batches,1))

    for k, (x_val, y_val) in enumerate(zip(x_ts,y_ts)):

        it_batch[k] = len(x_val)
        loss_init[k], _ = pmodel_base.evaluate(x_val[:,:,base_features], y_val, verbose = 0)
    
    plt.scatter(it_batch, loss_init)
    plt.xlabel('#iterations for full Markov Chain computation.')
    plt.ylabel('mae - discounted, weighted CFs')
    plt.yscale('log')
    plt.show()
    
    
    pmodel = combine_base_and_res_model(model_base = pmodel_base, model_res = pmodel_res)


    # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
    # use gradient clipping and/ or lr-decay to avoid this issue
    pmodel.compile(optimizer = tf.keras.optimizers.Adam(clipnorm=1), loss=compute_loss)
    pmodel.summary()
    exit()
    ###################################
    EPOCHS_PER_ROUND = 100 # Note: Not too high too avoid overfitting on low-duration contracts
    ROUNDS = 5#00 # Note: relatively high too fit transition-probs smoothly on the whole feature space, iteratively per round for all batches
    ###################################

    for it in range(ROUNDS):
        # store training history
        history = {}
        for k, (x_val, y_val) in enumerate(zip(x,y)):
            
            x_val_base, x_val_res = x_val[:,:,base_features], x_val[:,:,res_features]
            print('training batch {}/{}, round {}: '.format(k, N_batches, it))
            print('\t loss prior training: ', pmodel.evaluate(x=[x_val_base,x_val_res], y=y_val, verbose=0))
            print('\t starting training')
            tic = time.time()
            # Note: whole-batch training -> avoid overfitting locally as single (low-duration) contracts only access small manifold on feature space
            pmodel.fit(x=[x_val_base,x_val_res], y=y_val, epochs= EPOCHS_PER_ROUND, verbose=0, batch_size= len(x_val))
            print('\t training complete after {} sec.'.format(np.round_(time.time()-tic,2)))
            print('\t loss after training: ', pmodel.evaluate(x=[x_val_base,x_val_res], y=y_val, verbose=0))
            # history of batch k
            history[k] = pmodel.history.history
        # save one round of training all batches
        np.save(os.path.join(cwd, r'HMC_hist_iteration_{}.npy'.format(it)), history)
        if (it>0) and (it%2 == 0):
            pmodel.save(os.path.join(cwd, r'HMC_model_it_{}.h5'.format(it)))


    loss_new = np.zeros((N_batches,1))

    for k, (x_val, y_val) in enumerate(zip(x,y)):

        #it_batch[k] = len(x_val)
        loss_new[k], _ = pmodel_base.evaluate(x_val[:,:,base_features], y_val, verbose = 0)
    
    plt.scatter(it_batch, loss, alpha = 0.4, color='gray', label='pre-training',)
    plt.scatter(it_batch, loss_new, alpha = 0.4, color='green', label='post-training')
    plt.yscale('log')
    plt.show()
    
    # TO DO:    visualize baseline-loss vs base+res -loss
    #           visualize difference in mortality assumptions
    exit()
    for k, (x_val, y_val) in enumerate(zip(x,y)):
        # x_val.shape: (batch_size, steps, features)
        # y_val.shape: (batch_size, steps, 2), where 2 equal number of states (alive, dead)
        # print('\t x_val.shape: ', x_val.shape)
        #print('\t y_val.shape: ', y_val.shape)


        print('loss on batch {}: '.format(k))
        print('\t \t baseline-model:' , pmodel.evaluate(x_val_base, y_val,verbose=0))
        #print(pmodel.layers[1].get_weights())
        pmodel.fit(x_val[:,:,[0,3]],y_val,verbose=0)
        print('\t after training: ', pmodel.evaluate(x_val[:,:,[0,3]],y_val,verbose=0))
        #print(pmodel.layers[1].get_weights())


    #### NOTE: custom loss computation
    # https://stackoverflow.com/questions/41132633/can-numba-be-used-with-tensorflow

    
