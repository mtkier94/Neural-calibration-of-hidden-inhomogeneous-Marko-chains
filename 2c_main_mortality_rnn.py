import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from itertools import product as iter_prod

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, GRU, SimpleRNN

from global_vars import T_MAX
from functions.sub_data_prep import create_trainingdata_baseline
from functions.tf_model_base import create_baseline_model_rnn, transfer_weights_dense2simpleRNN
#from sub_tf_functions import create_baseline_model_rnn, transfer_weights_dense2simpleRNN

from global_vars import T_MAX
from global_vars import path_data, path_models_baseline_transfer


if __name__ == '__main__':

    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008Tmale.csv'),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))

    assert(T_MAX == len(p_survive)-1)

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    x, y = shuffle(x, y)

    #print('x:', x.shape, type(x))
    #print('y:', y.shape, type(y))
    
    BATCH = 1
    EPOCHS = 5000
    n_in = x.shape[1]
    n_out = 2

    # RNN inputs: A 3D tensor, with shape [batch, timesteps, feature]
    model_rnn = create_baseline_model_rnn(input_shape=(None, n_in), n_out= n_out, hidden_layers=[40,40,20])

    if False: # never train, only transfer FFN-baseline
        model_rnn.fit(x, y, batch_size=BATCH, epochs=EPOCHS)
        model_rnn.save(os.path.join(path_models_baseline_transfer, r'survival_baseline_ts.h5'))
    else:
        if os.path.exists(os.path.join(path_models_baseline_transfer,  r'survival_baseline.h5')):
            model_pretrained = load_model(os.path.join(path_models_baseline_transfer,  r'survival_baseline.h5'))
            #model_pretrained.summary()
            #print(model_pretrained.layers)

            rnn = transfer_weights_dense2simpleRNN(dense_model= model_pretrained, rnn_model = model_rnn)
            model_rnn.save(os.path.join(path_models_baseline_transfer, r'survival_baseline_ts.h5'))
            
        else:
            print('Model cannot be loaded or trained!')
            exit()

    model_rnn.summary()
    
    # print(x.shape)
    # print(x.reshape(1,-1,n_in).shape)
    # exit()

    pred = model_rnn.predict(x.reshape(1,-1,n_in))
    print('shape of predictions: ', pred.shape)

    plt.plot(x[:,0]*T_MAX, model_pretrained.predict(x)[:,0], 'og', alpha = .2)
    plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,0].flatten(), 'ob', alpha = .2, label='rnn')
    
    plt.plot(x[:,0]*T_MAX, y[:,0], 'xr')
    plt.plot(x[:,0]*T_MAX, model_pretrained.predict(x)[:,1], 'og', alpha = .2)
    plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,1].flatten(), 'ob', alpha = .2)
    
    plt.plot(x[:,0]*T_MAX, y[:,1], 'xr')
    plt.yscale('log')
    plt.show()

    # Additional sanity-check

    # n_test = 10
    # x_test_a = np.array([[(40+1/4*i)/T_MAX, 1/4] for i in range(n_test)]).reshape((1,n_test, n_in))
    # #pred_a = model_rnn.predict(x_test_a)
    # pred_a = model_rnn.predict(x.reshape(10,-1,n_in))
    # x_test_b = np.array([[(40+1/4*i)/T_MAX, 1/4] for i in range(n_test)])
    # pred_b = model_pretrained.predict(x)
    # #pred_b = model_pretrained.predict(x_test_b)

    # print('Option a): ', pred_a.shape, pred_a)
    # print('Option b): ', pred_b.shape, pred_b)
    
    # plt.scatter(pred_a.reshape(-1,2)[:,0], pred_b[:,0])
    # plt.show()

    # plt.scatter(pred_a.reshape(-1,2)[:,1], pred_b[:,1])
    # plt.show()