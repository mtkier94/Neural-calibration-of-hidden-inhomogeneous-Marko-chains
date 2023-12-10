import numpy as np
import pandas as pd
import os
import pickle5 as pickle
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.keras.backend.set_floatx('float64')

from functions.tf_loss_custom import compute_loss_mae, eval_loss_raw
from functions.sub_visualization import plot_new_vs_init_loss, mortality_heatmap_grid, plot_implied_survival_curve

from global_vars import path_data, path_models_baseline_transfer, path_models_resnet_with_padding_hpsearch_male, path_models_resnet_with_padding_hpsearch_female
from global_vars import T_MAX, AGE_RANGE


def run_main():
    '''
    Compare models trained on different baselines.
    '''


    #### load data
    with open(os.path.join(path_data, 'x_ts_raw.npy'), 'rb') as f:
        x_ts_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_ts.npy'), 'rb') as f:
        x_ts = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_ts_discounted.npy'), 'rb') as f:
        y_ts_discounted = np.load(f, allow_pickle=True)

    # zero-padded seq
    with open(os.path.join(path_data, 'x_ts_pad.npy'), 'rb') as f:
        x_ts_pad = np.load(f, allow_pickle=True)#.astype(np.float64)
    with open(os.path.join(path_data, 'y_ts_pad_discounted.npy'), 'rb') as f:
        y_ts_pad_discounted = np.load(f, allow_pickle=True)#.astype(np.float64)



    N_batches = len(x_ts)
    N_contracts, N_seq_pad, N_features = x_ts_pad.shape

    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]

    model = {}
    dav_baseline = {}

    for sex in ['male', 'female']:     
        if sex == 'male':
            path_model = path_models_resnet_with_padding_hpsearch_male
        else:
            path_model = path_models_resnet_with_padding_hpsearch_female
        # look at one individual model and visualize progress
        model[sex] = load_model(os.path.join(path_model, r'model_best.h5'.format(1e-3, 32)), compile=False)
        model[sex].compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
        print(f'Neural network for {sex} dav-baseline loaded sucessfully.')
        dav_baseline[sex] = pd.read_csv(os.path.join(path_data, r'DAV2008T{}.csv'.format(sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))


    # compare implied survival curves
    surv_curve = {}
    for sex in ['male', 'female']: 
        surv_curve[sex] = plot_implied_survival_curve(model[sex], dav_table = dav_baseline[sex], age_max = T_MAX, m=1, path_save = None, 
                        str_loss = '', baseline_tag= sex, age_range= AGE_RANGE  )    


    # plot death curves
    plt.plot(1-dav_table, label = f'DAV2008T{baseline_tag}', color='black')
    for sex in ['male', 'female']:
        for status in ['nonsmoker', 'smoker']:
            if sex == 'male':
                marker = 'x'
                linestyle = '-'#"None"
            else:
                marker = None
                linestyle = '-'
            plt.plot(1-val_dict[sex][status], marker = marker, linestyle = linestyle, label='{}, {}'.format(sex, status))
    plt.plot(1-dav_table, color='black')

    #-----------------------------------------------------

    # optional: indicate range of training data
    if type(age_range) != type(None):
        plt.vlines(age_range[0], ymin = 0, ymax= 0.1, color = 'gray', alpha = .5, linestyles = 'dashed')
        plt.vlines(age_range[1], ymin = 0, ymax= 1, color = 'gray', alpha = .5, linestyles = 'dashed')





    # compare raw, discounted CF-values, i.e. loss-values without mae or mse 
    loss_raw = {}
    for sex in ['male', 'female']:     
        # check raw losses
        loss_raw[sex] = eval_loss_raw(y_ts_pad_discounted, model[sex].predict([x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]]))

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
    ax[0].plot(np.arange(N_contracts), loss_raw['male'], linestyle = ':', alpha = 0.4, label = 'male')
    ax[0].plot(np.arange(N_contracts), loss_raw['female'], linestyle = ':', alpha = 0.4, label = 'female')
    ax[0].legend()
    # ax[0,1].plot(np.arange(N_contracts), loss_raw['female'], label = 'female')
    ax[1].plot(np.arange(N_contracts), loss_raw['male']-loss_raw['female'], label = 'male-female')
    ax[1].legend()
    plt.show()

if __name__ == '__main__':

    run_main()