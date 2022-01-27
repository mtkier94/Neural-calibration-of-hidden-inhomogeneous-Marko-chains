import numpy as np
from numpy.lib.function_base import quantile
import pandas as pd
import os, sys
# import pickle5 as pickle
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from functions.tf_loss_custom import compute_loss_mae, compute_loss_raw, eval_loss_raw
from functions.sub_visualization import mortality_heatmap_grid, plot_implied_survival_curve, mortality_heatmap_differences, heatmap_check_homogeneity, plot_economic_evaluation
from functions.sub_backtesting import check_exploded_gradients
from functions.sub_actuarial import neural_premium_zillmerisation
from functions.sub_backtesting import check_test_data

from global_vars import T_MAX, AGE_RANGE, INIT_AGE_RANGE, ALPHA, BETA, GAMMA
from global_vars import path_data, path_models_baseline_transfer, path_hyperopt_male, path_hyperopt_female
from global_vars import path_models_resnet_hpsearch_male, path_models_resnet_hpsearch_female, path_models_resnet_hpsearch_none

def run_visual_eval(baseline_sex = 'female', tuning_type = 'manual', path_tag = ''):
    '''
    Run analysis of results for given DAV-baseline, including
        1) Implied survival curve
        2) heatmap: fitted probs vs DAV baseline
        3) heatmap: fitted probs, difference between gender and smoker-status combos.

    Inputs:
    -------
        baseline_sex:   string, either "female" or "male"
        tuning_type:    sring, either "manual" or "auto"
        path_tag:       run experiment for paths in the style of [path_model]+path_tag, i.e. avoid changing the path_model manually to explore other experiments
    '''

    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
    # ALternative: For numeric stability, set the default floating-point dtype to float64
    # tf.keras.backend.set_floatx('float64')

    assert baseline_sex in ['male', 'female']
    assert tuning_type in ['manual', 'auto']

    if baseline_sex == 'male':
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_male+path_tag
        else:
            path_model = path_hyperopt_male
        sex2 = 'female' 
    elif baseline_sex == 'female':
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_female+path_tag
        else:
            path_model = path_hyperopt_female
        sex2 = 'male'
    else:
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_none+path_tag
        else:
            #! Not implemented
            raise ValueError('An automated HPSearch without the baseline model has not been implemented yet.')
            # path_model = path_hyperopt_none
        sex2 = None
        

    # zero-padded seq
    with open(os.path.join(path_data, 'x_train.npy'), 'rb') as f:
        x_train = np.load(f, allow_pickle=True)#.astype(np.float64)
    with open(os.path.join(path_data, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)#.astype(np.float64)

    # N_batches = len(x_ts)
    N_contracts, N_seq_pad, N_features = x_train.shape

    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    # res_features = [0,3,6,7]
    # base_features = [0,3]

    # look at one individual model and visualize progress
    # pmodel = load_model(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(1e-3, 32)), compile=False)
    pmodel = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
    p_survive = pd.read_csv(os.path.join(path_data, r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    p_survive2 = pd.read_csv(os.path.join(path_data, r'DAV2008T{}.csv'.format(sex2)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))

    # pmodel.layers[-1].summary()
    # exit()

    # check raw losses
    # if baseline_sex != 'none':
    #     loss_raw = eval_loss_raw(y_train, pmodel.predict([x_train[:,:,base_features], x_train[:,:,res_features]]))
    # else:
    #     loss_raw = eval_loss_raw(y_train, pmodel.predict(x_train[:,:,res_features]))
    # plt.plot(np.arange(N_contracts), loss_raw)
    # # plt.show()
    # plt.close()

    bool_grad = check_exploded_gradients(pmodel)
    if bool_grad:
        raise ValueError('NaN-parameter-values in model!')


    # where did the training alter the underlying mortality-table?
    #loss, _ = pmodel.evaluate([x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted,  batch_size = 1024, verbose = 0)
    # loss_classic = compute_loss_mae(y_true = y_train, y_pred = pmodel.predict([x_train[:,:,base_features], x_train[:,:,res_features]])).numpy()
    
    #print('tf.eval vs. tf.loss: ', loss, ' vs. ', np.round_(loss_classic,2))
    # str_loss = f'loss: {str(np.round_(loss_classic,2))}'
    #str_loss = ''

    _ = plot_implied_survival_curve(pmodel, dav_table = p_survive,  dav_table2= p_survive2,
                                        path_save = path_model, age_max = T_MAX,
                                        baseline_tag= baseline_sex, age_range= AGE_RANGE  )
    
    # plot a heatmap of the 
    val_dict, _ = mortality_heatmap_grid(pmodel, p_survive, baseline_tag=baseline_sex, m=1, age_max=T_MAX, rnn_seq_len = 20, save_path= path_model, age_range=INIT_AGE_RANGE)

    # try:
    heatmap_check_homogeneity(val_dict, baseline_tag= baseline_sex, save_path=path_model, age_range=INIT_AGE_RANGE)
    # except ValueError as e:
    #     print('heatmap could not be created due to log(0) values.')
    #     # print(e) # most likely 0-value that conflicts with log-scale of heatmap
    #     pass

    mortality_heatmap_differences(val_dict, baseline_tag=baseline_sex, save_path=path_model, age_range=INIT_AGE_RANGE)



def run_econom_eval(baseline_sex ='male', tuning_type = 'manual', path_tag = ''):
    '''
    Intrinsic backtesting. Use transition probabilities of calibrated neural network 'model_best' to compute premium values.
    Results are displayed as absolute and relative errors, arranged with respect to selected policy features, such as initial age.

    Inputs:
    -------
        baseline_sex:   string, either "female" or "male"
        tuning_type:    string, either "manual" or "auto"
        path_tag:       run experiment for paths in the style of [path_model]+path_tag, i.e. avoid changing the path_model manually to explore other experiments

    '''

    assert baseline_sex in ['male', 'female', 'none']
    assert tuning_type in ['manual', 'auto']

    if baseline_sex == 'male':
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_male+path_tag
        else:
            path_model = path_hyperopt_male
    elif baseline_sex == 'female':
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_female+path_tag
        else:
            path_model = path_hyperopt_female
    else:
        if tuning_type == 'manual':
            path_model = path_models_resnet_hpsearch_none+path_tag
        else:
            #! Not implemented
            raise ValueError('An automated HPSearch without the baseline model has not been implemented yet.')
            # Note: tests were run but did not ecourage a more detailed analysis due to poor performance
            # path_model = path_hyperopt_none


    #### load test data, i.e. with premium values set to zero and cash-flows w/o premium-related payments
    with open(os.path.join(path_data, r'x_test_raw.npy'), 'rb') as f:
        x_test_raw = np.load(f, allow_pickle=True)
        # print('\t .. x_test_raw loaded. ', type(x_test_raw), ' of shape ', x_test_raw.shape)
    with open(os.path.join(path_data, r'x_test.npy'), 'rb') as f:
        x_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        # print('\t .. x_test loaded.', type(x_test), ' of shape ', x_test.shape)
    # load cash-flow values (test-data w/o premium-related payments)
    with open(os.path.join(path_data, r'y_test.npy'), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        # print('\t .. y_test loaded.', type(y_test), ' of shape ', y_test.shape)

    # backtesting: check consistency of training and test data
    with open(os.path.join(path_data, 'x_train.npy'), 'rb') as f:
        x_train = np.load(f, allow_pickle=True)
        # print('\t .. x_train loaded for backtesting.')
    with open(os.path.join(path_data, 'x_train_raw.npy'), 'rb') as f:
        x_train_raw = np.load(f, allow_pickle=True)
        # print('\t .. x_train_raw loaded. Premium-values will be used for economic evaluation.')

    # sanity check for loaded data
    # check_test_data(data_train=x_train, data_test=x_test)

    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]
    
    # look at one individual model and visualize progress
    print('Loading model_best.h5 ...')
    pmodel = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)
    pmodel.compile(loss = compute_loss_mae, metrics=[compute_loss_raw], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant


    # lump sums for net-premiums P_0
    # Note: cash-flows y_test are of the from APV_Premium - APV_Sum - APV_Cost with the premium-related payments set to 0
    #       hence, we need a negative sign to obtain the P_0 quantity 
    if baseline_sex != 'none':
        y_pred = pmodel.predict([x_test[:,:,base_features], x_test[:,:,res_features]])
    else:
        y_pred = pmodel.predict(x_test[:,:,res_features])
    P_0 = -eval_loss_raw(y_true = y_test, y_pred = y_pred).numpy().reshape((-1,1))
    
    
    # premium duration = t/ZahlweiseNum (bspw. 3/(1/12) = 36 [iterations])
    # Note: we need to use raw, i.e. non-scaled values, here
    premium_duration = x_test_raw[:,0,2].reshape((-1,1))/x_test_raw[:,0,3].reshape((-1,1))
    P_true = x_train_raw[:,0,-1].reshape((-1,1))

    zill_factor = neural_premium_zillmerisation(y = y_pred, freq = x_test[:,0,3], v = GAMMA, 
                                                t_iter=premium_duration, alpha=ALPHA, beta=BETA) #model = pmodel,
    P_pred = P_0/zill_factor
    q = 0.995
 
    
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    name_lst, index_lst = [r'initial age $a_0$', r'premium value $P$', r'sum insured $S$', r'premium duration $t$', r'duration $n$', r'payment style $m$'], [0, -1, -2, 2, 1, 3]
    
    for e in ['absolute', 'relative']:
        plot_economic_evaluation(val_true=P_true, val_pred= P_pred, path_save=path_model, 
                                error_type= e, baseline_tag= baseline_sex,
                                features_data=x_train_raw, features_id_lst=index_lst, features_str_lst=name_lst, q=q)

    # create table with statistics
    e_rel = (P_true-P_pred)/P_true
    alphas = [0, 0.005, 0.1, 0.25, 0.5, 0.75, 0.9, 0.995, 1]

    quantiles = np.quantile(e_rel, alphas)
    stats = pd.DataFrame(data = None, columns = alphas)
    stats.loc[baseline_sex+r' $q_\alpha$ [%]']=np.round(quantiles*100,2)


    print(stats)

    stats.to_latex(os.path.join(path_model, r'{}_error_rel_stats.tex'.format(baseline_sex)))

    
if __name__ == '__main__':

    # ---------------------
    # load results of manual ('manual') or automated ('auto') HPTuning
    # optional: load load as a user input when running the current file
    try:
        mode = sys.argv[1]
        if mode not in ['manual', 'auto']:
            raise ValueError('User input not compatible.')
        print('HPTuning mode: ', mode)
    except:
        mode = 'manual' 
    # ---------------------

    for gender in ['male', 'female']:
        print('####################################################')
        # optional: loop over old layer-settings of manual HPSearch
        for tag in ['_50_50_50_50_50']: #'_40_40_20', '_50_50_50', '_50_50_50_50', '_50_50_50_50_50', '_50_50_50_50_50_50', '_18_11_2021_best']:
        
            import warnings
            warnings.filterwarnings("ignore")
            import logging
            logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL) # supress postscript latency warnings
        

            # create all qualitative plots
            run_visual_eval(baseline_sex = gender, tuning_type= mode, path_tag=tag)

            print('\t layer widths: ' + tag)
            # perform economic backtesting
            run_econom_eval(baseline_sex= gender, tuning_type= mode, path_tag=tag)