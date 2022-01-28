import pandas as pd 
import numpy as np 
import os

from functions.sub_data_prep import prep_data, scale_timeseries, transform_to_timeseries

from global_vars import T_MAX, GAMMA
from global_vars import path_data, path_data_backtesting


def run_main(mode = 'train'):
    '''
    Perform the following actions
        1) Fetch csv-data, i.e. a portfolio of contracts
        2) scale data
        3) obtain target data, i.e. discounted cash flows, which depend on 
            - the discounting factor GAMMA (HParam) and  
            - the implicit assumption on the cost structure in the transform_to_timeseries function
        4) apply zero-padding to target and contract data to speed up neural training lateron
        5) save the processed data

    Note: 
        For training (mode = train) we assume the premiums to be known.
        For economic testing (mode = test) of the final neural network architecture we will set the premium equal to zero and infer it from the trained transition probabilities.
        More detail for the testing-mode will be given in the respective python-script. However, we use this preprocessing-function for creating the scaled contract data and the cash-flow data without premium related costs.
    '''

    assert mode in ['train', 'test'] # sanity-check for user input

    #### load original data for processing
    data = pd.read_csv(os.path.join(path_data,r'Tarifierung_RI_2017.csv'),  delimiter=';'  )

    # Note: scaler fit, but not used to scale/ transform data yet (!!)
    data, scaler = prep_data(data, scale_age = (0, T_MAX))

    with open(os.path.join(path_data, 'Data.npy'), 'wb') as f:
        np.save(f, data)

    if mode == 'test':
        # set premium values to 0
        # Note from prep_data(): data = x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']].values
        data[:,-1] = 0

    # transform to ts; data-format: list of sequences of different lengths
    x_ts_raw, y_ts = transform_to_timeseries(data)

    # scale ts (x-data only)
    # Important: sklearn-learn scaler is not compatible with manually set feature-range, i.e. scaler.transform(x) should not be used -> custom function scale_timeseries()
    x_ts = scale_timeseries(x_ts_raw, scaler)

    # create new target data y_ts_discounted, which includes discounting straight away
    # Note: discount needs to respect the frequency of observations, e.g. monthly, semi-anually, anually
    y_ts_discounted = [None]*len(y_ts)
    for k, (x_val, y_val) in enumerate(zip(x_ts, y_ts)):
        number_steps = x_val.shape[1]
        y_ts_discounted[k] = y_val*GAMMA**(x_val[:,:,3:4]*np.arange(number_steps).reshape(1,-1, 1))

    # prepare for saving objects
    x_ts, y_ts = np.array(x_ts, dtype='object'), np.array(y_ts, dtype='object')
    x_ts_raw = np.array(x_ts_raw, dtype='object')
    y_ts_discounted = np.array(y_ts_discounted, dtype='object')

    if mode == 'train':
        with open(os.path.join(path_data_backtesting,r'x_ts_raw.npy'), 'wb') as f:
            np.save(f, x_ts_raw)
        with open(os.path.join(path_data_backtesting,r'x_ts.npy'), 'wb') as f:
            np.save(f, x_ts)
        with open(os.path.join(path_data_backtesting,r'y_ts.npy'), 'wb') as f:
            np.save(f, y_ts)
        with open(os.path.join(path_data_backtesting,r'y_ts_discounted.npy'), 'wb') as f:
                np.save(f, y_ts_discounted)

    
    # create zero paded data
    N_features_x = x_ts[0].shape[-1]
    N_features_y = y_ts[0].shape[-1]
    N_contracts = 0
    max_len = 0
    for el in x_ts:
        N_contracts += len(el)
        max_len = max(max_len, el.shape[1])


    x, x_raw = np.zeros((N_contracts, max_len, N_features_x), dtype=np.float32), np.zeros((N_contracts, max_len, N_features_x), dtype=np.float32)
    y = np.zeros((N_contracts, max_len, N_features_y), dtype=np.float32)

    pointer = 0
    for el_x, el_x_raw, el_yd in zip(x_ts, x_ts_raw, y_ts_discounted):
        batch_sz = len(el_x)
        steps = el_x.shape[1]
        x[pointer:pointer+batch_sz, 0:steps] = el_x
        x_raw[pointer:pointer+batch_sz, 0:steps] = el_x_raw
        y[pointer:pointer+batch_sz, 0:steps] = el_yd
        pointer += batch_sz

    
    with open(os.path.join(path_data,r'x_{}.npy'.format(mode)), 'wb') as f:
        np.save(f, x)
    with open(os.path.join(path_data,r'x_{}_raw.npy'.format(mode)), 'wb') as f:
        np.save(f, x_raw)
    with open(os.path.join(path_data,r'y_{}.npy'.format(mode)), 'wb') as f:
        np.save(f, y)


if __name__ == '__main__':

    # data and cash-flows with premium-values
    run_main(mode='train')
    
    # data and cash-flows without premium-values (resp. premium set to 0)
    run_main(mode='test')




