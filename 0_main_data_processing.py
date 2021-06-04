import pandas as pd 
import numpy as np 
import os, copy

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from functions.sub_actuarial import get_CFs_vectorized
from functions.sub_data_prep import prep_data, scale_timeseries, transform_to_timeseries

from global_vars import T_MAX, GAMMA
from global_vars import path_data



if __name__ == '__main__':

    #### load data
    data = pd.read_csv(os.path.join(path_data,r'Tarifierung_RI_2017.csv'),  delimiter=';'  )

    # Note: scaler fit, but not used to scale/ transform data yet (!!)
    data, scaler = prep_data(data, scale_age = (0,T_MAX))


    with open(os.path.join(path_data, 'Data.npy'), 'wb') as f:
        np.save(f, data)

    # transform to ts
    x_ts_raw, y_ts = transform_to_timeseries(data)
    # scale ts (x-data only)
    # Important: sklearn-learn scaler is not compatible with manually set feature-range, i.e. scaler.transform(x) should not be used -> custom function    
    # code snippet to show error if one were to use scaler.transform()
    # print(scaler.data_min_, scaler.data_max_)
    # print(data[0:2])
    # print(scaler.transform(data[0:2,:]))
    # print((data[0:2]-scaler.data_min_)/(scaler.data_max_-scaler.data_min_))
    x_ts = scale_timeseries(x_ts_raw, scaler)

    # prepare for saving objects
    x_ts, y_ts = np.array(x_ts, dtype='object'), np.array(y_ts, dtype='object')
    x_ts_raw = np.array(x_ts_raw, dtype='object')

    with open(os.path.join(path_data,'x_ts_raw.npy'), 'wb') as f:
        np.save(f, x_ts_raw)
    with open(os.path.join(path_data,'x_ts.npy'), 'wb') as f:
        np.save(f, x_ts)
    with open(os.path.join(path_data,'y_ts.npy'), 'wb') as f:
        np.save(f, y_ts)


    # create new target data y_ts_discounted, which include discounting straight away
    y_ts_discounted = [None]*len(y_ts)
    for k, (x_val, y_val) in enumerate(zip(x_ts, y_ts)):
        number_steps = x_val.shape[1]
        y_ts_discounted[k] = y_val*GAMMA**(x_val[:,:,3:4]*np.arange(number_steps).reshape(1,-1, 1))

    # prepare for saving objects
    y_ts_discounted = np.array(y_ts_discounted, dtype='object')

    with open(os.path.join(path_data,'y_ts_discounted.npy'), 'wb') as f:
        np.save(f, y_ts_discounted)
    


    # create zero paded data
    N_features_x = x_ts[0].shape[-1]
    N_features_y = y_ts[0].shape[-1]
    N_contracts = 0
    max_len = 0
    for el in x_ts:
        N_contracts += len(el)
        max_len = max(max_len, el.shape[1])

    #print(N_contracts, max_len)

    x_ts_pad = np.zeros((N_contracts, max_len, N_features_x), dtype=np.float32)
    y_ts_pad, y_ts_pad_discounted = np.zeros((N_contracts, max_len, N_features_y), dtype=np.float32), np.zeros((N_contracts, max_len, N_features_y), dtype=np.float32)

    pointer = 0
    for el_x, el_y, el_yd in zip(x_ts,y_ts, y_ts_discounted):
        batch_sz = len(el_x)
        steps = el_x.shape[1]
        x_ts_pad[pointer:pointer+batch_sz, 0:steps] = el_x
        y_ts_pad[pointer:pointer+batch_sz, 0:steps] = el_y
        y_ts_pad_discounted[pointer:pointer+batch_sz, 0:steps] = el_yd
        pointer += batch_sz

    
    with open(os.path.join(path_data,'x_ts_pad.npy'), 'wb') as f:
        np.save(f, x_ts_pad)
    with open(os.path.join(path_data,'y_ts_pad.npy'), 'wb') as f:
        np.save(f, y_ts_pad)
    with open(os.path.join(path_data,'y_ts_pad_discounted.npy'), 'wb') as f:
        np.save(f, y_ts_pad_discounted)


    # print(x_ts_pad.shape)
    # print(x_ts_pad[3,0:10])





