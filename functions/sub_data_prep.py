import pandas as pd 
import numpy as np 
import os, copy, sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from itertools import product as iter_prod

from functions.sub_actuarial import get_CFs_vectorized


def prep_data(x, scale_age = (0,120), scale_freq = (0,1)):

    '''
    Prepare the dataframe such that it represents the desired format of a state

    Inputs:
    -------
        x:  pd.DataFrame of expected format
        scale_age:  min & max of age for MinMaxScaler(); Note: This scaling has to be in line with pretrained baseline-mortality model  

    Outputs:
    --------
        prep: prepared data (e.g. with one-hot encoding) in the pd.DataFrame format
        scaler: MinMaxScaler object, fitted to the range of features in prep
    '''

    #x['Zahlweise2'] = x.ZahlweiseInkasso.map(lambda x: (x=='HALBJAEHRLICH'))
    #x['Zahlweise4'] = x.ZahlweiseInkasso.map(lambda x: (x=='VIERTELJAEHRLICH'))
    #x['Zahlweise12'] = x.ZahlweiseInkasso.map(lambda x: (x=='MONATLICH'))
    x['ZahlweiseNum'] = x.ZahlweiseInkasso.map(lambda x: (x=='JAEHRLICH')+(x=='HALBJAEHRLICH')/2+(x=='VIERTELJAEHRLICH')/4+(x=='MONATLICH')/12)

    x['GeschlechtNum'] = x.GeschlechtVP1.map(lambda x: int(x=='MAENNLICH'))
    x['RauchertypNum'] = x.RauchertypVP1.map(lambda x: int(x=='NICHTRAUCHEN'))

    data = x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']].values
    scaler = MinMaxScaler().fit(data)
    scaler.data_min_[0], scaler.data_max_[0] = scale_age
    scaler.data_min_[3], scaler.data_max_[3] = scale_freq
    #data_scaled = scaler.transform(prep)

    # Note: data not scaled; scaler simply fit to (training-)data
    return data, scaler


def transform_to_timeseries(x):
    '''
    Transform contracts x to a timeseries with stepsize 'frequency' up to maturity 'duration.
    Create targets y, which per contract represent a timeseries with respective cash-flows, conditional that the state is reached.

    Inputs:
    -------
        x:  pd.DataFrame of expected format, stemming from function prep_data( )
            format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]

    Outputs:
    --------    
        x:  transformed pd.DateFrame with list (timeseries) per contract
        y:  target values (time series with cash-flows) for x
    
    '''

    assert(len(x.shape) == 2)
    n_features = x.shape[1]

    # determine batches of equal length in data
    lengths_eff, counts = np.unique(x[:,1]/x[:,3], return_counts=True)
    # time-series - initialize batchwise (for batches of equal effective length)
    x_ts = [None]*len(counts)
    # target values - initialize batchwise
    y_ts = [None]*len(counts)

    # compute CF vector-valued in batches, i.e. for contracts with same no. of step maturity/frequency
    for k, (n_eff, count) in enumerate(zip(lengths_eff, counts)):

        assert(n_eff == int(n_eff))
        n_eff = int(n_eff)
        index= (x[:,1]/x[:,3]==n_eff)

        # Get cash-flows; shape: (N_batch, N_eff, 2)
        # Note: no discounting yet
        CFs = get_CFs_vectorized(x[index,:])
        # transform contracts to time-series; format/ shape: (batch_size, N_eff, N_features) where batch_size = counts
        # single batch, shape: (counts, n_features) -> after stacking, shape: (n_eff, counts, n_features) -> use swap axis to arrive at (counts, n_eff, n_features)
        contracts = np.stack([np.concatenate([x[index,0:1]+k*x[index,3:4],x[index,1:]], axis = 1) for k in range(n_eff)], axis=0)#.reshape((count,n_eff, n_features))
        #contracts = np.stack([scaler.transform(np.concatenate([x[index,0:1]+k*x[index,3:4],x[index,1:]], axis = 1)) for k in range(n_eff)], axis=0)#.reshape((count,n_eff, n_features))
        contracts = np.swapaxes(contracts, axis1=0, axis2=1)
        #print('Shapes: Batch {}, n_eff {}, n_feat {}'.format(count, n_eff, n_features))
        #print('contracts, shape: ', contracts.shape)
        x_ts[k], y_ts[k] = contracts, CFs
    return x_ts, y_ts

def apply_scaler(x, scaler):
    '''
    Two motivations for not using e.g. scaler.transform(x)
        1) sklearn does not seem to consistently support transforming x after manually adapting scaler attributes as scaler.data_min_, scaler.data_max_
        2) This custom implementation allows for vectorized scaling applied to time-series data (using broadcasting)

    Inputs:
    -------
        x:  data to be scaled either (n_examples, n_features) or (n_examples, n_steps, n_features)
        scaler: sklear-scaler object, from which we obtain scaling range scaler.data_min_, scaler.data_max_

    Outputs:
    --------
        x_scaled:   scaled data
    '''

    if len(x.shape) == 3:
        feat_min, feat_max = scaler.data_min_.reshape((1,1,-1)), scaler.data_max_.reshape((1,1,-1))
    elif len(x.shape) == 2:
        feat_min, feat_max = scaler.data_min_.reshape((1,-1)), scaler.data_max_.reshape((1,-1))
    else:
        ValueError

    return (x-feat_min)/(feat_max-feat_min)


def scale_timeseries(x, scaler):
    '''
    Given a fitted scaler object, apply scaling to features of a time-series for each step in time.

    Inputs:
    -------
        x:  list with np.arrays of different shapes as elements; shapes follow the logic of (N_batch, N_steps, N_features)
        scaler: fitted scaler object, suitable for N_features - elements

    Outputs:
    --------
        x_scaled: list with np.arrays of different shapes as elements; now, elements are scaled to the range of the scaler object (default: [0,1])
    '''
    # copy format of x
    x_scaled = copy.deepcopy(x)

    for k in range(len(x)):
        x_scaled[k] = apply_scaler(x[k], scaler)
        #for t in range(x[k].shape[1]):
            #x_scaled[k][:,t,:] = x[k][:,t,:]

    return x_scaled


def create_trainingdata_baseline(frequencies, surv_probs, age_scale):
    '''
    Create training data (x,y) where contracts x contain the current (scaled) age in combination with payment-frequency 1/m, m in mathbb{N} and 
    y the 1/m-step transition-probabilities. We assume two states, namely alive and dead. 
    Target quantities y stem from the DAV2008Tmale survival table. Sub-annual dead-probabilities are scaled linearly, 
    i.e. {}_{1/m}q_{age} = 1/m*q_{age} and {}_{1/m}p_{age}=1-{}_{1/m}q_{age}.
    -> data to be used for training a FFN-Net baseline to replicate whatever survival-table contains surv_probs, e.g. DAVT2008m.

    Inputs:
    -------
        frequencies:    payment frequencies 1/m, i.e. 1 (annual), 1/2 (semi-annual), etc.
        surv_table:     np.array of shape (max_age+1, 1) with annual survival probabilities, age starting at 0.
                        Note: maximum age of survival, after which survival probabilities 1/m*q_{age} are floored to zero, inferred by len(surv_table)-1

    Outputs:
    --------
        Data x,y        x,y both np.array with x.shape and y.shape (len(surv_table) x len(frequencies), 2)
    '''

    age_max = len(surv_probs)-1

    ages = np.arange(age_max+1)/age_scale # starting from age 0
    x = np.asarray(list(iter_prod(ages,frequencies)), dtype = 'float32')
    y = np.asarray(list(iter_prod(surv_probs,frequencies)), dtype = 'float32') # adapt to x format

    y[:,1] = (1-y[:,0])*y[:,1] # death-prob: adjust for subannual steps
    y[:,0] = 1-y[:,1] # fill in surv.-prob

    return x,y