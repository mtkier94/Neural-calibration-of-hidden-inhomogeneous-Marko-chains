import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


def mortality_ffn_heatmap(pmodel, dav_table, m = 1, age_max = 121, save_path = None):
    '''
    Given a mortality model pmodel and a reference-table (dav_table) display the differences as a heatmap.

    Inputs:
    -------
        pmodel: tf.model.Model() with (for now) two outputs that correspond to 1-step transition probs
                input.shape = (None, 2), output.shape = (None, 2)
        dav_table:  np.array with 1-year survival-probs from age 0 up to age_max
                    Note: more than 2 transition-probs might require additional tables as input
        m:      step width of transition-probs, i.e. m= 12 -> monthly
        age_max:    maximum age, used for scaling age input to pmodel
        save_path:  string or os.path-object which indicates - if provided - where the plot should be saved

    Ouputs:
    -------
        None; Heatmaps displayed
    '''

    assert(m==1) # adjust code for m!= 1

    ages = np.arange(0,age_max + 1/m, 1/m).reshape((-1,))
    freq = np.repeat(1/m, len(ages)).reshape((-1,))

    x_in = np.stack([ages/age_max,freq], axis=-1)
    #print('x_in.shape: ', x_in.shape)
    pred = pmodel.predict(x_in)[:, 0:1] # grab only survival prob
    #print('pred.shape: ', pred.shape)
    delta = dav_table - pred
    #print('delta.shape: ', delta.shape)

    plt.plot(ages, dav_table, 'r+')
    plt.plot(ages, pred, '-b')
    plt.show()

    sns.heatmap(data = delta)
    if type(save_path) == type(str):
        plt.savefig(os.path.join(save_path, 'heatmap_ffn.png'))
    plt.show()

    return

def mortality_rnn_heatmap(pmodel, dav_table, m = 1, age_max = 121, rnn_seq_len = 20, save_path = None):
    '''
    Given a mortality model pmodel and a reference-table (dav_table) display the differences as a heatmap.

    Inputs:
    -------
        pmodel: tf.model.Model() with (for now) two outputs that correspond to 1-step transition probs
                input.shape = (None, 2), output.shape = (None, 2)
        dav_table:  np.array with 1-year survival-probs from age 0 up to age_max
                    Note: more than 2 transition-probs might require additional tables as input
        m:      step width of transition-probs, i.e. m= 12 -> monthly
        age_max:    maximum age, used for scaling age input to pmodel
        bool_rnn:   boolean, indicating whether pmodel is a FFN oder RNN
        rnn_seq_len:    length of the sequence input to pmodel, in case that bool_rnn == True

    Ouputs:
    -------
        None; Heatmaps displayed
    '''

    assert(m==1) # adjust code for m!= 1

    ages = np.array([np.arange(k, k+rnn_seq_len+1/m, 1/m) for k in np.arange(0, age_max+1/m, 1/m)]).reshape((age_max+m, -1))
    freq = np.repeat(1/m, ages.shape[0]*ages.shape[1]).reshape((age_max+m, -1))
    x_in = np.stack([ages/age_max,freq], axis=-1)
    pred = pmodel.predict(x_in)[:,:, 0] # grab only survival prob
    
    # reshape table-values to rnn-format 
    cache = np.append(arr = dav_table, values= np.array([0]*rnn_seq_len).reshape(-1,1)).reshape((-1,1))
    true_vals = np.array([cache[k:k+int((rnn_seq_len+1)*m)] for k in np.arange(0, (age_max+1)*m)]).reshape((age_max+m, -1))
    delta = true_vals - pred
    
    if False: # shape debugging
        print('ages.shape: ', ages.shape)
        print('freq.shape: ', freq.shape)
        print('x_in.shape: ', x_in.shape)
        print('pred.shape: ', pred.shape)
        print('cache.shape: ', cache.shape)
        print('delta.shape: ', delta.shape)
    
    # sanity check for quality of model_pred vs DAV-table
    #plt.plot(ages, true_vals, 'r+')
    #plt.plot(ages, pred, '-b')
    #plt.show()

    sns.heatmap(data = delta)
    if type(save_path) == type(str):
        plt.savefig(os.path.join(save_path, 'heatmap_rnn.png'))
    plt.show()

    return

def mortality_heatmap(pmodel, dav_table, sex, nonsmoker, m =1, age_max = 121, rnn_seq_len = 20,  save_path = None):
    '''
    Create a heatmap to hightlight the difference in m-step survival probs between the (calibrated) pmodel and the baseline dav_table.
    Note: We assume pmodel to have only two additional feature-inputs, namely sex and smoker (in that order)

    Inputs:
    -------
        pmodel: rnn-type tf-model with two headed input of shapes (None, None, 2) and (None, None, 4)
        dav_table:  np.array with 1-year survival-probs from age 0 up to age_max
                    Note: more than 2 transition-probs might require additional tables as input
        sex:        'm'/ 'male' (eventually encoded as 1) or 'f' or 'female' (eventually encoded as 0)
        nonsmoker:     boolean with True or False
        save_path:  optional; path where to save heatmap

    Outputs:
    --------
        heatmap will be displayed (and optionally saved)
    '''

    assert(m==1) # adjust code for m!= 1
    


    ages = np.array([np.arange(k, k+rnn_seq_len+1/m, 1/m) for k in np.arange(0, age_max+1/m, 1/m)]).reshape((age_max+m, -1))
    freq = np.repeat(1/m, ages.shape[0]*ages.shape[1]).reshape((age_max+m, -1))

    if sex == 'm' or sex == 'male':
        arr_sex = np.ones(shape = freq.shape)
    elif sex == 'f' or sex == 'female':
        arr_sex = np.zeros(shape = freq.shape)
    else:
        raise ValueError('invalid sex')

    if nonsmoker == True:
        arr_nonsmoke = np.ones(shape = freq.shape)
    elif nonsmoker == False:
        arr_nonsmoke = np.zeros(shape = freq.shape)
    else:
        raise ValueError('Invalid non-smoker status')

    x_in_base = np.stack([ages/age_max,freq], axis=-1)
    x_in_res = np.stack([ages/age_max,freq, arr_sex, arr_nonsmoke], axis=-1)
    pred = pmodel.predict([x_in_base, x_in_res])[:,:, 0] # grab only survival prob
    
    # reshape table-values to rnn-format 
    cache = np.append(arr = dav_table, values= np.array([0]*rnn_seq_len).reshape(-1,1)).reshape((-1,1))
    true_vals = np.array([cache[k:k+int((rnn_seq_len+1)*m)] for k in np.arange(0, (age_max+1)*m)]).reshape((age_max+m, -1))
    delta = true_vals - pred

    sns.heatmap(data = delta)
    plt.ylabel('age x')
    plt.xlabel('value k')
    plt.title(r'surv. prob. $p_{x+k}$ for ph with sex: {}, nonsmoker: {}'.format(sex, nonsmoker))
    if type(save_path) == type(str):
        plt.savefig(os.path.join(save_path, 'heatmap_rnn.png'))
    plt.show()

    # return values to check whether male and female plot differ
    return delta

def mortality_heatmap_grid(pmodel, dav_table, m =1, age_max = 121, rnn_seq_len = 20,  save_path = None, age_range = None):
    '''
    Create a heatmap to hightlight the difference in m-step survival probs between the (calibrated) pmodel and the baseline dav_table.
    Note:   We assume pmodel to have only two additional feature-inputs, namely sex and smoker (in that order)
    Difference to mortality_heatmap():  Display the various heatmaps for all combinations of sex and smoker-status in one figure
    
    Inputs:
    -------
        pmodel: rnn-type tf-model with two headed input of shapes (None, None, 2) and (None, None, 4)
        dav_table:  np.array with 1-year survival-probs from age 0 up to age_max
                    Note: more than 2 transition-probs might require additional tables as input
        sex:        'm'/ 'male' (eventually encoded as 1) or 'f' or 'female' (eventually encoded as 0)
        nonsmoker:     boolean with True or False
        save_path:  optional; path where to save heatmap
        age_range:  optional; tupel which provides a range of the age too zoom in, e.g. (20,60)

    Outputs:
    --------
        heatmap will be displayed (and optionally saved)
        values:     dictionary with delta values (delta = dav_table - predicted_value)
    '''

    assert(m==1) # adjust code for m!= 1

    # initialize dictionary to save values
    val_dict = {'male': {'nonsmoker': None, 'smoker': None}, 'female': {'nonsmoker': None, 'smoker': None}}

    # dav_table: reshape table-values to rnn-format 
    # appended zeros: after max-age survival-prob equals 0
    surv = np.append(arr = dav_table, values= np.array([0]*rnn_seq_len).reshape(-1,1)).reshape((-1,1))
    true_vals = np.array([surv[k:k+int((rnn_seq_len+1)*m)] for k in np.arange(0, (age_max+1)*m)]).reshape((age_max+m, -1))    

    # step1: create base-input for pmodel
    ages = np.array([np.arange(k, k+rnn_seq_len+1/m, 1/m) for k in np.arange(0, age_max+1/m, 1/m)]).reshape((age_max+m, -1))
    # print('ages.shape: ', ages.shape)
    # print(ages)
    freq = np.repeat(1/m, ages.shape[0]*ages.shape[1]).reshape((age_max+m, -1))

    # step 2+3: create residual-input for pmodel and compute predicted surv.-probs
    # Note: 'male' and 'nonsmoker' are encoded by '1', 'female' and 'smoker' by '0'
    # This is done in the preprocessing in sub_data_prep.py
    for sex in ['male', 'female']:
        arr_sex = np.ones(shape = freq.shape)*(sex=='male')
        for status in ['nonsmoker', 'smoker']:
            arr_status = np.ones(shape = freq.shape)*(status=='nonsmoker')

            x_in_base = np.stack([ages/age_max,freq], axis=-1)
            # print('base_input.shape: ', x_in_base.shape)
            x_in_res = np.stack([ages/age_max,freq, arr_sex, arr_status], axis=-1)
            pred = pmodel.predict([x_in_base, x_in_res])[:,:, 0] # grab only survival prob
            
            val_dict[sex][status] = pred

    
    # create heatmaps
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    for i, sex in enumerate(['male', 'female']):
        for j, status in enumerate(['nonsmoker', 'smoker']):
            hm = sns.heatmap(data = true_vals - val_dict[sex][status], ax = ax[i,j])
            # hm.set_xticklabels(hm.get_xticklabels(), rotation=90) 
            # hm.set_yticklabels(hm.get_yticklabels(), rotation=90) 
            ax[i,j].set_title('ph: {} and {}'.format(sex, status))
            ax[i,j].set_xlabel('k')
            ax[i,j].set_ylabel('age x')
    fig.suptitle(r'survival prob. $p_{x+k}$: dav_table - model_prediction')#, fontsize=20)
    if type(save_path) != type(None):
        plt.savefig(os.path.join(save_path, 'heatmap_grid_dav_vs_pred.png'))
        plt.close()
    else:
        plt.show()


    # optional: zoom in for provided age_range:
    if type(age_range) != type(None):
        age_low, age_up = age_range
        index_low, index_up = age_low*m, age_up*m+1

        # create heatmaps
        fig, ax = plt.subplots(2,2, figsize=(12,10))
        for i, sex in enumerate(['male', 'female']):
            for j, status in enumerate(['nonsmoker', 'smoker']):
                assert m==1, 'visualization not yet adjusted for 1/m'
                
                hm = sns.heatmap(data = true_vals[index_low:index_up] - val_dict[sex][status][index_low:index_up], ax = ax[i,j], yticklabels= [t if t%5==0 else '' for t in np.arange(age_low, age_up)])
                #hm.set_xticklabels(hm.get_xticklabels(), rotation=90) 
                #hm.set_yticklabels([str(t) for t in np.arange(age_low, age_up, 1/m)], rotation=90) 
                ax[i,j].set_title('ph: {} and {}'.format(sex, status))
                ax[i,j].set_xlabel('m')
                ax[i,j].set_ylabel('age x')
        fig.suptitle(r'survival prob. $p_{x+m}$: dav_table - model_prediction')#, fontsize=20)
        if type(save_path) != type(None):
            plt.savefig(os.path.join(save_path, 'heatmap_grid_dav_vs_pred_zoom.png'))
            plt.close()
        else:
            plt.show()

    # return values to check whether male and female plot differ
    return val_dict, true_vals



    
def plot_new_vs_init_loss(pmodel, pmodel_base, x, y, base_feat_in, res_feat_in, path_save = None):
    '''
    Compare the loss (per batch) after training pmodel with the initial baseline pmodel_base
    
    Inputs:
    -------
        pmodel, pmodel_base:    optimized tf-models
                                Note: pmodel combines pmodel_base and pmodel_res
        x:    List with one batch of contracts (of equal no. of iterations) per entry
        y:    List of (discounted) cash-flows for contracts in x
        base_feat_in:   list of which columns of x are input to pmodel_base, i.e. x[:,:,base_feat_in]
        res_feat_in:   list of which columns of x are input to pmodel_res, i.e. x[:,:,res_feat_in]
    
    Outputs:
        plot of loss, i.e. weighted discounted CFs, comparing pmodel and pmodel_base
        -> ilustration of training progress
    '''

    N_batches = len(x)
    it_steps = np.zeros((N_batches, 1))
    loss_init = np.zeros((N_batches, 1))
    loss_new = np.zeros((N_batches, 1))

    for k, (x_val, y_val) in enumerate(zip(x,y)): 
        it_steps[k] = x_val.shape[1]

        x_val_base, x_val_res = x_val[:,:,base_feat_in], x_val[:,:,res_feat_in]
        loss_new[k], _ = pmodel.evaluate(x=[x_val_base,x_val_res], y=y_val, verbose=0, batch_size=1024)
        loss_init[k], _ = pmodel_base.evaluate(x=x_val_base, y=y_val, verbose = 0, batch_size=1024)

    plt.plot(it_steps, loss_new, label='new values (base + residual)')
    plt.plot(it_steps, loss_init, alpha=0.5, label='initial values (baseline)')
    plt.xlabel('no. of steps in HMC computation')
    plt.ylabel('mean loss - discounted, weighted CFs')
    plt.legend()
    #plt.yscale('log')
    if type(path_save) != type(None):
        plt.savefig(os.path.join(path_save, 'loss_new_vs_init.png'))
        plt.close()
    else:
        plt.show()

def plot_implied_survival_curve(pmodel, dav_table, age_max = 121, m=1, path_save = None, str_loss = '', age_range = None):

    '''
        Plot the 1-step transition probabilites implied by pmodel vs. stated by the dav_table.
        The visualization assumes that pmodel is a combined model with p_base and p_res where both model take age and frequency as input, but p_res additionally takes categoritcal features sex and smoker_status.
        We visualize the effect of 'switching on or off' these categorical features.

        Inputs:
        -------
            pmodel: tf.keras.models.Model
            dav_table:  survival probabilities (1-year) of some reference, e.g. a dav_table
            age_max:    maximum age for which to plot the curve
            m:      frequency at which death probabilities {}_{1/m} q_x should be considered
            save_path:  optional; path, where to save the respective figure
            str_loss:   optional; additional info (string) to add to the plot-title, e.g. the respective loss on the training data
            age_range:  optional; tupel in the form of (age_low, age_up) indicating which range the (training-) data provided


        Outputs:
        --------
            if save_path != None: figure will be saved under respective path
            else:   figure will be displayed

    '''
    assert(len(dav_table) == age_max +1)

    ages = np.arange(0,age_max+1, 1).reshape((-1,1))
    freq = np.ones(shape = ages.shape)/m

    val_dict = {'male': {'nonsmoker': None, 'smoker': None}, 'female': {'nonsmoker': None, 'smoker': None}}

    for sex in ['male', 'female']:
        arr_sex = np.ones(shape = freq.shape)*(sex=='male')
        for status in ['nonsmoker', 'smoker']:
            arr_status = np.ones(shape = freq.shape)*(status=='nonsmoker')

            x_in_base = np.stack([ages/age_max,freq], axis=-1)
            x_in_res = np.stack([ages/age_max,freq, arr_sex, arr_status], axis=-1)
            val_dict[sex][status] = pmodel.predict([x_in_base, x_in_res])[:,:, 0].flatten()

    # plot death curves
    plt.plot(1-dav_table, label = 'DAV2008T(male)', color='black')
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

    # optional: indicate range of training data
    if type(age_range) != type(None):
        plt.vlines(age_range[0], ymin = 0, ymax= 0.1, color = 'gray', alpha = .5, linestyles = 'dashed')
        plt.vlines(age_range[1], ymin = 0, ymax= 1, color = 'gray', alpha = .5, linestyles = 'dashed')
    
    plt.legend()
    plt.yscale('log')
    plt.xlabel('age x')
    plt.ylabel(r'${}_1q_x$')
    plt.title(str_loss)
    if type(path_save) != type(None):
        plt.savefig(os.path.join(path_save, 'implied_surv_curve.png'))
        plt.close()
    else:
        plt.show()


    return val_dict


def plot_implied_survival_curve_confidence(pmodel, dav_table, age_max = 121, m=1 ):

    '''
        Plot the 1-step transition probabilites implied by pmodel vs. stated by the dav_table.
        The visualization assumes that pmodel is a combined model with p_base and p_res where both model take age and frequency as input, but p_res additionally takes categoritcal features sex and smoker_status.
        We visualize the effect of 'switching on or off' these categorical features.

        Note: In addition to plot_implied_survival_curve() we visualize sequencial prediction of the rnn-type model pmodel.
    '''
    
    assert(len(dav_table) == age_max +1)

    ages = np.arange(0,age_max, 1).reshape((-1,1))
    freq = np.ones(shape = ages.shape)*m

    val_dict = {'male': {'nonsmoker': None, 'smoker': None}, 'female': {'nonsmoker': None, 'smoker': None}}

    for sex in ['male', 'female']:
        arr_sex = np.ones(shape = freq.shape)*(sex=='male')
        for status in ['nonsmoker', 'smoker']:
            arr_status = np.ones(shape = freq.shape)*(status=='nonsmoker')

            x_in_base = np.stack([ages/age_max,freq], axis=-1)
            x_in_res = np.stack([ages/age_max,freq, arr_sex, arr_status], axis=-1)
            val_dict[sex][status] = pmodel.predict([x_in_base, x_in_res])[:,:, 0].flatten()

    # plot survival curves
    plt.plot(1-dav_table, label = 'DAV2008T(male)', color='black')
    for sex in ['male', 'female']:
        for status in ['nonsmoker', 'smoker']:
            if sex == 'male':
                marker = 'x'
            else:
                marker = None
            plt.plot(1-val_dict[sex][status], marker = marker, label='{}, {}'.format(sex, status))
    plt.plot(1-dav_table, color='black')
    
    plt.legend()
    plt.yscale('log')
    plt.title('death probabilities')
    plt.show()