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
from functions.sub_backtesting import check_if_rnn_version
#from sub_tf_functions import create_baseline_model_rnn, transfer_weights_dense2simpleRNN

from global_vars import T_MAX
from global_vars import path_data, path_models_baseline_transfer


if __name__ == '__main__':

    raise ValueError('Content of this script has been moved/ appended to main_baseline.py')


    baseline_sex = 'female'
    p_survive = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))

    if baseline_sex == 'female':
        p_other_sex = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format('male')),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))
        tag_other_sex = 'DAVT2008male'
    elif baseline_sex == 'male':
        p_other_sex = pd.read_csv(os.path.join(path_data,r'DAV2008T{}.csv'.format('female')),  delimiter = ';', header=None ).loc[:,0].values.reshape((-1,1))
        tag_other_sex = 'DAVT2008female'
    else:
        raise ValueError('Unknown baseline_sex')

    assert(T_MAX == len(p_survive)-1)

    freqs = [1,1/2, 1/3, 1/6, 1/12]
    x, y = create_trainingdata_baseline(frequencies = freqs, surv_probs=p_survive, age_scale=T_MAX)
    x, y = shuffle(x, y)

    n_in = x.shape[1]
    n_out = 2

    # RNN inputs: A 3D tensor, with shape [batch, timesteps, feature]
    model_rnn = create_baseline_model_rnn(input_shape=(None, n_in), n_out= n_out, hidden_layers=[40,40,20])
    #model_rnn.summary()

    
    if os.path.exists(os.path.join(path_models_baseline_transfer,  r'ffn_davT{}.h5'.format(baseline_sex))):
        model_pretrained = load_model(os.path.join(path_models_baseline_transfer,  r'ffn_davT{}.h5'.format(baseline_sex)))
        model_pretrained.evaluate(x,y, batch_size=64)
        print('loss-type: ', model_pretrained.loss)

        transfer_weights_dense2simpleRNN(dense_model= model_pretrained, rnn_model = model_rnn)
        model_rnn.save(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))    
        print('Weights transferred from ffn to rnn!')

        assert (check_if_rnn_version(model_ffn=model_pretrained, model_rnn=model_rnn)==True).all()
    else:
        print('Model cannot be loaded or trained!')
        exit()



    pred = model_rnn.predict(x.reshape(1,-1,n_in))
    print('shape of predictions: ', pred.shape)

    plt.plot(x[:,0]*T_MAX, model_pretrained.predict(x)[:,0], 'xg', alpha = .5, label='ffn')
    plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,0].flatten(), 'ob', alpha = .2, label='rnn')
    
    plt.plot(x[:,0]*T_MAX, y[:,0], linestyle = 'None', marker = '_', color = 'red', label='DAV')
    plt.plot(x[:,0]*T_MAX, model_pretrained.predict(x)[:,1], 'xg', alpha = .5)
    plt.plot(x[:,0]*T_MAX, model_rnn.predict(x.reshape(1,-1,n_in))[0,:,1].flatten(), 'ob', alpha = .2)
    
    plt.plot(x[:,0]*T_MAX, y[:,1], linestyle = 'None', marker = '_', color = 'red',)
    plt.yscale('log')
    plt.title('Fit - FFN vs. RNN vs DAVT2008{}'.format(baseline_sex))
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()