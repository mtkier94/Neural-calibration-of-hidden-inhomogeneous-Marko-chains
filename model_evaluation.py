import numpy as np
import pandas as pd
import os, sys

import pandas as pd
from tensorflow.keras.models import load_model

from functions.tf_loss_custom import compute_loss_mae, compute_loss_raw, eval_loss_raw
from functions.sub_visualization import mortality_heatmap_grid, plot_implied_survival_curve, mortality_heatmap_differences, heatmap_check_homogeneity, plot_economic_evaluation
from functions.sub_backtesting import check_exploded_gradients
from functions.sub_actuarial import neural_premium_zillmerisation

from global_vars import T_MAX, AGE_RANGE, INIT_AGE_RANGE, ALPHA, BETA, GAMMA
from global_vars import path_data, path_dav, path_hyperopt_male, path_hyperopt_female
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

    # Optional: uncomment upon demand
    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
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
        

    # look at one individual model and visualize progress
    try:
        pmodel = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)
    except:
        print('"model_best.h5" seems not to exist. Potentially, it has to manually be created first be copying and renaming the model of choice, which was found during HPTuning.')
        raise ValueError('Loading model_best.h5 failed. Path reference: ' +str(os.path.join(path_model, r'model_best.h5')))
    pmodel.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
    p_survive = pd.read_csv(os.path.join(path_dav, r'DAV2008T{}.csv'.format(baseline_sex)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))
    p_survive2 = pd.read_csv(os.path.join(path_dav, r'DAV2008T{}.csv'.format(sex2)),  delimiter=';', header=None ).loc[:,0].values.reshape((-1,1))


    bool_grad = check_exploded_gradients(pmodel)
    if bool_grad:
        raise ValueError('NaN-parameter-values in model!')


    # where did the training alter the underlying mortality-table?
    _ = plot_implied_survival_curve(pmodel, dav_table = p_survive,  dav_table2= p_survive2,
                                        path_save = path_model, age_max = T_MAX,
                                        baseline_tag= baseline_sex, age_range= AGE_RANGE  )
    
    # plot a heatmap of the 
    val_dict, _ = mortality_heatmap_grid(pmodel, p_survive, baseline_tag=baseline_sex, m=1, age_max=T_MAX, rnn_seq_len = 20, save_path= path_model, age_range=INIT_AGE_RANGE)

    heatmap_check_homogeneity(val_dict, baseline_tag= baseline_sex, save_path=path_model, age_range=INIT_AGE_RANGE)
    mortality_heatmap_differences(val_dict, baseline_tag=baseline_sex, save_path=path_model, age_range=INIT_AGE_RANGE)



def run_econom_eval(baseline_sex ='male', tuning_type = 'manual', path_tag = '', kfolds = 1):
    '''
    Intrinsic backtesting. Use transition probabilities of calibrated neural network 'model_best' to compute premium values.
    Results are displayed as absolute and relative errors, arranged with respect to selected policy features, such as initial age.

    Inputs:
    -------
        baseline_sex:   string, either "female" or "male"
        tuning_type:    string, either "manual" or "auto"
        path_tag:       run experiment for paths in the style of [path_model]+path_tag, i.e. avoid changing the path_model manually to explore other experiments
        cv_tag:         string. Helper tag for crossvalidation study. Ex.: cv_1 will 

    '''

    assert baseline_sex in ['male', 'female', 'none']
    assert tuning_type in ['manual', 'auto']
    assert kfolds in (1,2), 'only no crossvalidation (1) or 2-fold cv (2) supported.'

    for k in range(kfolds):
        # tag which which slice the model was trained on, e.g. [model_name]_cv_0[.h5] and [x_train]_cv_0[.npy]
        cv_train_tag = f'_cv_{k}' if kfolds>1 else ''
        # validation tag. Select hold-out data for validation, e.g. [x_test]_cv_1[.npy]
        cv_validation_tag = f'_cv_{(k+1)%2}' if kfolds>1 else ''
        
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
        with open(os.path.join(path_data, f'x_test_raw{cv_validation_tag}.npy'), 'rb') as f:
            x_test_raw = np.load(f, allow_pickle=True)
            # print('\t .. x_test_raw loaded. ', type(x_test_raw), ' of shape ', x_test_raw.shape)
        with open(os.path.join(path_data, f'x_test{cv_validation_tag}.npy'), 'rb') as f:
            x_test = np.load(f, allow_pickle=True)#.astype(np.float64)
            # print('\t .. x_test loaded.', type(x_test), ' of shape ', x_test.shape)
        # load cash-flow values (test-data w/o premium-related payments)
        with open(os.path.join(path_data, f'y_test{cv_validation_tag}.npy'), 'rb') as f:
            y_test = np.load(f, allow_pickle=True)#.astype(np.float64)
            # print('\t .. y_test loaded.', type(y_test), ' of shape ', y_test.shape)
    
        # # backtesting: check consistency of training and test data
        # with open(os.path.join(path_data, f'x_train{cv_validation_tag}.npy'), 'rb') as f:
        #     x_train = np.load(f, allow_pickle=True)
        #     # print('\t .. x_train loaded for backtesting.')

        # load x_values from holdout set
        with open(os.path.join(path_data, f'x_train_raw{cv_validation_tag}.npy'), 'rb') as f:
            contract_features_raw = np.load(f, allow_pickle=True)
            # print('\t .. contract_features_raw loaded. Premium-values will be used for economic evaluation.')
    
        # sanity check for loaded data
        # check_test_data(data_train=x_train, data_test=x_test)
    
        # select contract-features for res-net
        # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
        res_features = [0,3,6,7]
        base_features = [0,3]
        
        # look at one individual model and visualize progress
        path_local = os.path.join(path_model, f'model_best{cv_train_tag}.h5')
        print(f'Loading {path_local} ...')
        pmodel = load_model(path_local, compile=False)
        pmodel.compile(loss = compute_loss_mae, metrics=[compute_loss_raw], optimizer = 'adam') # as we don't train anymore, specifics of the optimizer are irrelevant
    
    
        # Markov transition probabilities of calibrated model
        if baseline_sex != 'none':
            y_pred = pmodel.predict([x_test[:,:,base_features], x_test[:,:,res_features]])
        else:
            y_pred = pmodel.predict(x_test[:,:,res_features])
        # lump sums for net-premiums P_0
        # Note: cash-flows y_test are of the from APV_Premium - APV_Sum - APV_Cost with the premium-related payments set to 0
        #       hence, we need a negative sign to obtain the P_0 quantity 
        P_0 = -eval_loss_raw(y_true = y_test, y_pred = y_pred).numpy().reshape((-1,1))
        
        
        # premium duration = t/ZahlweiseNum (bspw. 3/(1/12) = 36 [iterations])
        # Note: we need to use raw, i.e. non-scaled values, here
        premium_duration = x_test_raw[:,0,2].reshape((-1,1))/x_test_raw[:,0,3].reshape((-1,1))
        P_true = contract_features_raw[:,0,-1].reshape((-1,1))
    
        zill_factor = neural_premium_zillmerisation(y = y_pred, freq = x_test[:,0,3], v = GAMMA, 
                                                    t_iter=premium_duration, alpha=ALPHA, beta=BETA) #model = pmodel,
        P_pred = P_0/zill_factor
        q = 0.995
     
        
        # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
        name_lst, index_lst = [r'initial age $a_0$', r'premium value $P$', r'sum insured $S$', r'premium duration $t$', r'duration $n$', r'payment style $m$'], [0, -1, -2, 2, 1, 3]
        
        for e in ['absolute', 'relative']:
            plot_economic_evaluation(val_true=P_true, val_pred= P_pred, path_save=path_model, 
                                    error_type= e, baseline_tag= baseline_sex,
                                    features_data=contract_features_raw, 
                                    features_id_lst=index_lst, features_str_lst=name_lst, q=q,
                                    kfold_tag=cv_train_tag)
    
        # create table with statistics
        e_rel = (P_true-P_pred)/P_true
        alphas = [0, 0.005, 0.1, 0.25, 0.5, 0.75, 0.9, 0.995, 1]
    
        quantiles = np.quantile(e_rel, alphas)
        stats = pd.DataFrame(data = None, columns = alphas)
        stats.loc[baseline_sex+r' $q_\alpha$ [%]']=np.round(quantiles*100,2)
        stats.to_latex(os.path.join(path_model, r'{}{}_error_rel_stats.tex'.format(baseline_sex, cv_train_tag)))

    
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
        print('Using manual HPTuning by default ..')
    # ---------------------

    for gender in ['female', 'male']:
        print('####################################################')
        # optional: loop over old layer-settings of manual HPSearch
        # Note: for the automated hp-search the empty tag '' is required
        for tag in ['_50_50_50_50_50']: #'_40_40_20', '_50_50_50', '_50_50_50_50', '_50_50_50_50_50', '_50_50_50_50_50_50', '_18_11_2021_best']:
        
            import warnings
            warnings.filterwarnings("ignore")
            import logging
            logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL) # supress postscript latency warnings when saving images in an .eps-format
        

            # create all qualitative plots
            run_visual_eval(baseline_sex = gender, tuning_type= mode, path_tag=tag)

            print('\t layer widths: ' + tag)
            # perform economic backtesting
            run_econom_eval(baseline_sex= gender, tuning_type= mode, path_tag=tag)