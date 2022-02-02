import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# set some plotting parameters globally
parameters = {'axes.labelsize': 16, 'xtick.labelsize':14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 16, 'figure.titlesize': 18}
plt.rcParams.update(parameters)

def plot_implied_survival_curve(pmodel, dav_table, dav_table2 = None, age_max = 121, path_save = None, baseline_tag ='male', age_range = None):

    '''
        Plot the 1-step transition probabilites implied by pmodel vs. stated by the dav_table.
        The visualization assumes that pmodel is a combined model with p_base and p_res where both model take age and frequency as input, but p_res additionally takes categoritcal features sex and smoker_status.
        We visualize the effect of 'switching on or off' these categorical features.
        Note: payment style m=1 is hard encoded. Visualization of e.g. m=1/12 possible, but not very insightful

        Inputs:
        -------
            pmodel: tf.keras.models.Model
            dav_table:  survival probabilities (1-year) of some reference, e.g. a dav_table
            dav_table2: second, potential baseline of survival probs., such as the oposite gender as in dav_table
            age_max:    maximum age for which to plot the curve
            save_path:  optional; path, where to save the respective figure
            str_loss:   optional; additional info (string) to add to the plot-title, e.g. the respective loss on the training data
            baseline_tag: string, indicating whether male, female DAV-baseline or 'none' i.e. no baseline was used
            age_range:  optional; tupel in the form of (age_low, age_up) indicating which range the (training-) data provided


        Outputs:
        --------
            if save_path != None: figure will be saved under respective path
            else:   figure will be displayed

    '''
    m=1 # hard encode annual payment style for this type of plot
    assert(len(dav_table) == age_max +1) # sanity check

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
    plt.plot(1-dav_table, label = f'DAV2008T {baseline_tag}', color='black')
    # plot 2nd baseline
    if type(dav_table2) != type(None):
        if baseline_tag == 'male':
            tag2 = 'female'
        else:
            tag2 = 'male'
        plt.plot(1-dav_table2, label = f'DAV2008T {tag2}', color='gray')

    # plot calibrated transition probs
    for sex in ['male', 'female']:
        for status in ['nonsmoker', 'smoker']:
            if sex == 'male':
                marker = '.'
                linestyle = 'None'#'-'#"None"
            else:
                marker = None
                linestyle = '--'
            plt.plot(1-val_dict[sex][status], marker = marker, linestyle = linestyle, label='{}, {}'.format(sex, status))
    # plt.plot(1-dav_table, color='black') # re-draw baseline


    # optional: indicate range of training data
    if type(age_range) != type(None):
        plt.vlines(age_range[0], ymin = 0, ymax= 1, color = 'gray', alpha = .5, linestyles = 'dashed')
        plt.vlines(age_range[1], ymin = 0, ymax= 1, color = 'gray', alpha = .5, linestyles = 'dashed')
    
    plt.legend(loc = 'lower right', fontsize = 11)
    plt.yscale('log')
    plt.xlabel(r'age $a_0$')
    plt.ylabel(r'1-year death prob.')
    if type(path_save) != type(None):
        plt.savefig(os.path.join(path_save, f'{baseline_tag}_implied_surv_curve.png'), dpi=400)
        plt.savefig(os.path.join(path_save, f'{baseline_tag}_implied_surv_curve.eps'), dpi=400)
        plt.close()
    else:
        plt.show()

    return val_dict



def mortality_heatmap_grid(pmodel, dav_table, baseline_tag, m =1, age_max = 121, rnn_seq_len = 20,  save_path = None, age_range = None):
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
    hm_values = val_dict.copy()
    min_cb, max_cb = 0.,0. # color bar scaling

    # dav_table: reshape table-values to rnn-format 
    # appended zeros: after max-age survival-prob equals 0
    surv = np.append(arr = dav_table, values= np.array([0]*rnn_seq_len).reshape(-1,1)).reshape((-1,1))
    true_vals = np.array([surv[k:k+int((rnn_seq_len+1)*m)] for k in np.arange(0, (age_max+1)*m)]).reshape((age_max+m, -1))    

    # step1: create base-input for pmodel
    ages = np.array([np.arange(k, k+rnn_seq_len+1/m, 1/m) for k in np.arange(0, age_max+1/m, 1/m)]).reshape((age_max+m, -1))
    freq = np.repeat(1/m, ages.shape[0]*ages.shape[1]).reshape((age_max+m, -1))

    # step 2+3: create residual-input for pmodel and compute predicted surv.-probs
    # Note: 'male' and 'nonsmoker' are encoded by '1', 'female' and 'smoker' by '0'
    # This is done in the preprocessing in sub_data_prep.py
    for sex in ['male', 'female']:
        arr_sex = np.ones(shape = freq.shape)*(sex=='male')
        for status in ['nonsmoker', 'smoker']:
            arr_status = np.ones(shape = freq.shape)*(status=='nonsmoker')

            x_in_base = np.stack([ages/age_max,freq], axis=-1)
            x_in_res = np.stack([ages/age_max,freq, arr_sex, arr_status], axis=-1)
            try:
                pred = pmodel.predict([x_in_base, x_in_res])[:,:, 0] # grab only survival prob
            except:
                print('Computation without baseline model.')
                pred = pmodel.predict(x_in_res)[:,:, 0] # grab only survival prob

            val_dict[sex][status] = pred
            hm_values[sex][status] = true_vals - val_dict[sex][status]
            # update colorbar range
            max_cb, min_cb = np.max([max_cb, np.max(hm_values[sex][status])]), np.min([min_cb, np.min(hm_values[sex][status])])

    
    # create heatmaps
    fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)

    for i, sex in enumerate(['male', 'female']):
        for j, status in enumerate(['nonsmoker', 'smoker']):
            # _ = sns.heatmap(data = true_vals - val_dict[sex][status], ax = ax[i,j], cmap= "Spectral")
            _ = sns.heatmap(data = hm_values[sex][status], ax = ax[i,j], cmap= "Spectral",
                            # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                            # cbar =i==0,
                            # vmin=min_cb, vmax=max_cb,
                            # cbar_ax=None if i else cbar_ax
                            )
            ax[i,j].set_title('{} {}'.format(sex, status)) 
            if j == 0:
                ax[i,j].set_ylabel(r'age $a_0$')
            if i == 1:
                ax[i,j].set_xlabel(r'$k$')
    fig.tight_layout()
    if type(save_path) != type(None):
        # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred.png'))
        plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred.eps')) # no dpi=400 due to image-size
        plt.close()
    else:
        plt.show()


    # optional: zoom in for provided age_range:
    if type(age_range) != type(None):
        age_low, age_up = age_range
        index_low, index_up = age_low*m, age_up*m+1

        # create heatmaps
        fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
        for i, sex in enumerate(['male', 'female']):
            for j, status in enumerate(['nonsmoker', 'smoker']):
                assert m==1, 'visualization not yet adjusted for 1/m'
                
                _ = sns.heatmap(data = hm_values[sex][status][index_low:index_up], cmap= "Spectral", ax = ax[i,j], 
                                # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                                # cbar =i==0,
                                # vmin=min_cb, vmax=max_cb,
                                # cbar_ax=None if i else cbar_ax,
                                yticklabels= [t if t%5==0 else '' for t in np.arange(age_low, age_up+1)])
                ax[i,j].set_title('{} {}'.format(sex, status)) 
                if j == 0:
                    ax[i,j].set_ylabel(r'age $a_0$')
                if i == 1:
                    ax[i,j].set_xlabel(r'$k$')
                
        if type(save_path) != type(None):
            fig.tight_layout()
            # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred_zoom.png'))
            plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred_zoom.eps')) # no dpi=400 due to image-size
            plt.close()
        else:
            plt.show()
    # return values to check whether male and female plot differ
    return val_dict, true_vals



def heatmap_check_homogeneity(val_dict, baseline_tag, save_path, age_range = None):
    '''
    Plot the mortality probabilities for all combinations of gender and smoker-status. 
    The resulting heatmaps, if constant on diagonals, show the homogeneity of the underlying Markov chain.

    Inputs:
    -------
        val_dict:   dictionary with keys representing gender and smoker-status combinations; values are mortalitiy probs.
        baseline_tag:   string-tag, indicating which gender the baseline of the model that produced values in val_dict was trained on
        save_path:  path where to save the plot
        age_range:  optional integer tuple-style input; if provided, we zoom into the respective range of ages for the heatmap
    '''
 
    # create heatmaps
    fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)

    for i, sex in enumerate(['male', 'female']):
        for j, status in enumerate(['nonsmoker', 'smoker']):

            _ = sns.heatmap(data = np.abs(val_dict[sex][status]), ax = ax[i,j], cmap= "Spectral", norm=LogNorm()
                            # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                            # cbar =i==0,
                            # vmin=min_cb, vmax=max_cb,
                            # cbar_ax=None if i else cbar_ax
                            )
            ax[i,j].set_title('{} {}'.format(sex, status))
            if j == 0:
                ax[i,j].set_ylabel(r'age $a_0$')
            if i == 1:
                ax[i,j].set_xlabel(r'$k$')
    fig.tight_layout()
    # fig.suptitle(r'survival prob. $p_{x+k}$: dav_table - model_prediction')#, fontsize=20)
    if type(save_path) != type(None):
        # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred.png'))
        plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_homogeneity.eps')) # no dpi=400 due to image-size
        plt.close()
    else:
        plt.show()

    # optional: zoom in for provided age_range:
    if type(age_range) != type(None):
        age_low, age_up = age_range
        m=1 # visualization restricted to annual payment style
        index_low, index_up = age_low*m, age_up*m+1

        # create heatmaps
        fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
        # cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, sex in enumerate(['male', 'female']):
            for j, status in enumerate(['nonsmoker', 'smoker']):                
                _ = sns.heatmap(data = np.abs(val_dict[sex][status][index_low:index_up]), cmap= "Spectral", 
                                ax = ax[i,j], norm=LogNorm(),
                                # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                                # cbar =i==0,
                                # vmin=min_cb, vmax=max_cb,
                                # cbar_ax=None if i else cbar_ax,
                                yticklabels= [t if t%5==0 else '' for t in np.arange(age_low, age_up+1)])
                ax[i,j].set_title('{} {}'.format(sex, status)) 
                if j == 0:
                    ax[i,j].set_ylabel(r'age $a_0$')
                if i == 1:
                    ax[i,j].set_xlabel(r'$k$') 
                
        # fig.suptitle(r'survival prob. $p_{x+m}$: dav_table - model_prediction')#, fontsize=20)
        if type(save_path) != type(None):
            fig.tight_layout()
            # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_dav_vs_pred_zoom.png'))
            plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_homogeneity_zoom.eps')) # no dpi=400 due to image-size
            plt.close()
        else:
            plt.show()




def mortality_heatmap_differences(val_dict, baseline_tag, save_path, age_range = None):
    '''
    Take the dictionary of values from mortality_heatmap_grid and look at the differences between gender and smoker-status combinations.
    '''

    # Note: implicitely assumes payment style m=1; Generalization possible but with little illustrative benefit 
    # -> intrinsic economic evaluation superior

    fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])

    min_cb, max_cb = 0., 0. # color bar range (to be updated)
    
    hm_values = {}
    hm_values[0] = val_dict['male']['nonsmoker'] -val_dict['female']['nonsmoker']
    min_cb, max_cb = np.min([min_cb, np.min(hm_values[0])]), np.max([max_cb, np.max(hm_values[0])])
    hm_values[1] = val_dict['male']['nonsmoker'] -val_dict['male']['smoker']
    min_cb, max_cb = np.min([min_cb, np.min(hm_values[1])]), np.max([max_cb, np.max(hm_values[1])])
    hm_values[2] = val_dict['male']['smoker'] -val_dict['female']['smoker']
    min_cb, max_cb = np.min([min_cb, np.min(hm_values[2])]), np.max([max_cb, np.max(hm_values[2])])
    hm_values[3] = val_dict['female']['nonsmoker'] -val_dict['female']['smoker']
    min_cb, max_cb = np.min([min_cb, np.min(hm_values[3])]), np.max([max_cb, np.max(hm_values[3])])

    ax = ax.flatten()
    for i in range(4):
        sns.heatmap(hm_values[i], cmap= "Spectral", ax=ax[i],
                    # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                    # cbar =i==0,
                    # vmin=min_cb, vmax=max_cb,
                    # cbar_ax=None if i else cbar_ax
                    )

    ax[0].set_title('male non-smoker - female non smoker') 
    ax[1].set_title('male non-smoker - male smoker') 
    ax[2].set_title('male smoker - female smoker')
    ax[3].set_title('female non-smoker - female smoker')
    for i in range(4):
        if i in [0,2]:
            ax[i].set_ylabel(r'age $a_0$')
        if i in [2,3]:
            ax[i].set_xlabel(r'$k$')
    fig.tight_layout()
    # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_delta.png'))
    plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_delta.eps')) # no dpi=400 due to image-size
    plt.close()


    if type(age_range) != type(None):

        # zoom in to age_range in data
        m = 1
        age_low, age_up = age_range
        index_low, index_up = age_low*m, age_up*m+1

        fig, ax = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
        # cbar_ax = fig.add_axes([.91, .3, .03, .4])

        # update color bar
        min_cb, max_cb = 0., 0.
        min_cb, max_cb = np.min([min_cb, np.min(hm_values[0][index_low:index_up])]), np.max([max_cb, np.max(hm_values[0][index_low:index_up])])
        min_cb, max_cb = np.min([min_cb, np.min(hm_values[1][index_low:index_up])]), np.max([max_cb, np.max(hm_values[1][index_low:index_up])])
        min_cb, max_cb = np.min([min_cb, np.min(hm_values[2][index_low:index_up])]), np.max([max_cb, np.max(hm_values[2][index_low:index_up])])
        min_cb, max_cb = np.min([min_cb, np.min(hm_values[3][index_low:index_up])]), np.max([max_cb, np.max(hm_values[3][index_low:index_up])])

        ax = ax.flatten()
        for i in range(4):
            sns.heatmap(hm_values[i][index_low:index_up], cmap= "Spectral", ax=ax[i],
                        # uncomment the next 3 lines and 'cbar_ax' to have all heatmaps share the same colorbar
                        # cbar =i==0,
                        # vmin=min_cb, vmax=max_cb,
                        # cbar_ax=None if i else cbar_ax,
                        yticklabels= [t if t%5==0 else '' for t in np.arange(age_low, age_up+1)])
        ax[0].set_title('male non-smoker - female non smoker')
        ax[1].set_title('male non-smoker - male smoker') 
        ax[2].set_title('male smoker - female smoker')
        ax[3].set_title('female non-smoker - female smoker') 
        for i in range(4):
            if i in [0,2]:
                ax[i].set_ylabel(r'age $a_0$') 
            if i in [2,3]:
                ax[i].set_xlabel(r'$k$') 
        # plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_delta_zoom.png'))
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'{baseline_tag}_heatmaps_delta_zoom.eps')) # no dpi=400 due to image-size
        plt.close()


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
    if type(path_save) != type(None):
        plt.savefig(os.path.join(path_save, 'loss_new_vs_init.png'))
        plt.close()
    else:
        plt.show()

def plot_economic_evaluation(val_true, val_pred, path_save, features_data, features_id_lst, 
                            features_str_lst, baseline_tag, error_type = 'relative', q = 0.995, ylim = (-0.11, 0.11)):

    '''
    Plot the results of the intrinsic, economic evaluation, i.e. backtesting true premium values by predicted premium values.

    Inputs:
    -------
        val_true:   true premium values
        val_pred:   predicted premium values, including zillmerisation of costs
        path_save:  path, where plots will be saved
        error_type:   string, either 'relative' or 'absolute' (errors)
        features_data:   data to analyze error; note: raw, i.e. unscaled data
                    format: 3d-tensor in the format (contract id, timestep, features)
        features_id_lst:    list of ids, w.r.t. which errors are to be analyze
        features_str_lst:   list of feature names, w.r.t. which errors are to be analyzed
        baseline_tag:       string-tag, which gender was used for training the baseline model
        
                    optional; default None
                    non-default: list of strings with features. the error will then be analyzed w.r.t. these features
        q:          quantile value, default 99.5%.

    Outputs:
    --------
        None; plots of errors will be saved.
    '''
    assert(error_type) in ['absolute', 'relative']

    # compute error
    error = val_true-val_pred
    if error_type == 'relative':
        error = error/val_true


    # (1-q)- and q-quantile
    q_low, q_up = np.quantile(error, 1-q), np.quantile(error, q)

    # single plot without decomposition w.r.t. individual features
    plt.scatter(range(1,len(error)+1), error, marker = '+', color = 'green', alpha = 0.4)
    plt.plot([0,len(error)], [q_low,q_low], linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
    plt.plot([0,len(error)], [q_up,q_up], linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
    plt.ylabel(error_type+' error') 
    if error_type == 'relative':
        # manual rescaling of axis to have approx. same scale for both genders
        plt.ylim((min(ylim[0], np.min(error)), max(ylim[1], np.max(error))))
    plt.xlabel('contracts')
    plt.tight_layout()
    # plot png format, as it captures the distribution implicitely via the transparency value alpha in the plot
    plt.savefig(os.path.join(path_save, f'{baseline_tag}_errors_{error_type}.png'), dpi=400)
    plt.savefig(os.path.join(path_save, f'{baseline_tag}_errors_{error_type}.eps'), dpi=400)
    plt.close()
    # print(error_type+' error plot wo. decomposition saved.')

    # check proper type
    assert(len(features_id_lst) == len(features_str_lst))
    assert(type(features_id_lst)== type([]))
    assert(type(features_str_lst)== type([]))
    assert(len(features_data.shape)==3)

    n = len(features_id_lst)
    n_rows = 2
    n_cols = int(np.ceil(n/n_rows))
    _, ax = plt.subplots(n_rows, n_cols, figsize = (12, 6))
    ax = ax.flatten()
    for k, i in enumerate(features_id_lst):
        if features_id_lst[k] != 3:
            ax[k].scatter(features_data[:,0,i], error, marker= '+', color = 'green',linewidth = 1, alpha = 0.4)
            ax[k].set_xlabel(features_str_lst[k])
            # indicate (1-q)- and q-quantile
            ax[k].plot([min(features_data[:,0,i]),max(features_data[:,0,i])], [q_low,q_low], 
                    linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
            ax[k].plot([min(features_data[:,0,i]),max(features_data[:,0,i])], [q_up,q_up], 
                    linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
        else:
            ax[k].scatter((1/features_data[:,0,i]).astype('int'), error, marker= '+', color = 'green',linewidth = 1, alpha = 0.4)
            ax[k].set_xlabel(features_str_lst[k]) 
            # indicate (1-q)- and q-quantile
            ax[k].plot([1,12], [q_low,q_low], 
                        linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
            ax[k].plot([1,12], [q_up,q_up], 
                        linestyle = (0, (5, 10)), linewidth = 1, color = 'black')
            ax[k].set_xticks(np.unique((1/features_data[:,0,i]).astype('int')))
        
        if k%n_cols == 0:
            ax[k].set_ylabel(error_type+' error')
        # ax[k].set_yticks(list(ax[k].get_yticks()) + [np.round_(err_low,2), np.round_(err_up,2)])

    # turn off display of axis that are not used
    for l in range(n,n_rows*n_cols):
        ax[l].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(path_save, f'{baseline_tag}_errors_{error_type}_decomposition.eps'), dpi=400)
    # plot png format, as it captures the distribution implicitely via the transparency value alpha in the plot
    plt.savefig(os.path.join(path_save, f'{baseline_tag}_errors_{error_type}_decomposition.png'), dpi=400) 
    plt.close()
    # print(error_type+' error plot wo. decomposition saved.')