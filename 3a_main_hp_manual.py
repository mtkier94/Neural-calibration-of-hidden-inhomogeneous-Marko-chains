import numpy as np
import pickle 
import os, time
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa


from functions.tf_loss_custom import compute_loss_mae, eval_loss_raw
from functions.tf_model_res import combine_models, create_mortality_res_net
from functions.sub_backtesting import check_exploded_gradients
from functions.sub_actuarial import neural_premium_zillmerisation

from global_vars import ALPHA, BETA, GAMMA, path_data, path_models_baseline_transfer
from global_vars import path_models_resnet_hpsearch_male, path_models_resnet_hpsearch_female, path_models_resnet_hpsearch_none


def exp_decay_scheduler(epoch, lr):
    if (epoch >= 50) and (epoch%15==0):
        return lr*0.9
    else:
        return lr


def exp_decay_scheduler_finetuning(epoch, lr):
    if (epoch >= 10) and (epoch%2==0):
        return lr*0.9
    else:
        return lr        

def ES():
    return tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)



def run_manual_HPS(baseline_sex, widths_lst = [40, 40, 20], bool_train = False):

    
    EPOCHS = 1000

    # speed-up by setting mixed-precision -> disable for now as it causes dtype issues in compute_loss, specifically when concatenating ones and cum_prob
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
    # ALternative: For numeric stability, set the default floating-point dtype to float64
    #tf.keras.backend.set_floatx('float64')

    if baseline_sex == 'male':
        path_model = path_models_resnet_hpsearch_male
    elif baseline_sex == 'female':
        path_model = path_models_resnet_hpsearch_female
    elif baseline_sex == 'none':
        path_model = path_models_resnet_hpsearch_none
    else:
        raise ValueError('Unknown Option for baseline_sex.')
    strategy = tf.distribute.MirroredStrategy()
    N_GPUs = strategy.num_replicas_in_sync

    # option for processing data during training
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    #### load data
    with open(os.path.join(path_data, 'x_train_raw.npy'), 'rb') as f:
        x_train_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_train.npy'), 'rb') as f:
        x_train = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)

    with open(os.path.join(path_data, r'x_test_raw.npy'), 'rb') as f:
        x_test_raw = np.load(f, allow_pickle=True)
        print('\t .. x_test_raw loaded. ', type(x_test_raw), ' of shape ', x_test_raw.shape)
    with open(os.path.join(path_data, r'x_test.npy'), 'rb') as f:
        x_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        print('\t .. x_test loaded.', type(x_test), ' of shape ', x_test.shape)
    # load cash-flow values (test-data w/o premium-related payments)
    with open(os.path.join(path_data, r'y_test.npy'), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        print('\t .. y_test loaded.', type(y_test), ' of shape ', y_test.shape)


    # N_contracts, N_seq_pad, N_features = x_train.shape


    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]



    # HP_PARAMS
    l2_penalty = 0.0
    callbacks = [tf.keras.callbacks.LearningRateScheduler(exp_decay_scheduler), ES()]
    LR_RATES = [1e-2,5e-3, 1e-3]
    HP_BZ = [32, 64, 128]

    if baseline_sex != 'none':
        pmodel_base = tf.keras.models.load_model(os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))
        pmodel_base.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')

    pmodel_res = create_mortality_res_net(hidden_layers=widths_lst, param_l2_penalty=l2_penalty, input_shape=(None, len(res_features)), n_out=2)
    pmodel_res.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = 'adam')
    
    # keep random initialization for later use (optional resetting for different hyperparams)
    w_init = deepcopy(pmodel_res.get_weights())

    # pmodel_transfer = combine_models(pmodel_base, pmodel_res, bool_masking = True)
    

    times_dict = {}

    for k, lrate in enumerate(LR_RATES):

        # Note for optimizer: Task seems to be susceptible for exploring gradients resulting in nan-loss caused by nan-weights
        # use gradient clipping and/ or lr-decay to avoid this issue
        opt = tf.keras.optimizers.Adam(lr=lrate, clipnorm=100.0)
        # opt_avg = tfa.optimizers.MovingAverage(opt) # showed little/ no benefit -> dropped from analysis

        for optimizer in [opt]: #, opt_avg]:
            for hp_batchsize in HP_BZ:
            
                if optimizer == opt:
                    tag = 'adam'
                else:
                    tag = 'avg'
                
                if bool_train:
                    
                    print('Running with lr {}, optimizer {} and bz {} and training for {} epochs.'.format(lrate, tag, hp_batchsize, EPOCHS))
                    # train pmodel
                    # initiate new res_new and reset weights -> same pseudo-random start for different optimizers
                    pmodel_res = create_mortality_res_net(hidden_layers=widths_lst, param_l2_penalty=l2_penalty, input_shape=(None, len(res_features)), n_out=2)
                    pmodel_res.set_weights(w_init)
                    if baseline_sex != 'none':
                        pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = True)
                    else:
                        pmodel = pmodel_res
                    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = optimizer)
                    #check if weight-initialization was sucessfull
                    #print('init model-eval: ', pmodel.evaluate(x=[x_ts_pad[:,:,base_features], x_ts_pad[:,:,res_features]], y = y_ts_pad_discounted, batch_size = 1024, verbose= 1))

                    tic = time.time()
                    if baseline_sex != 'none':
                        pmodel.fit(x=[x_train[:,:,base_features], x_train[:,:,res_features]], y = y_train, 
                                        epochs=EPOCHS, batch_size = hp_batchsize*N_GPUs, callbacks=callbacks, verbose= 1)
                    else:
                        pmodel.fit(x=x_train[:,:,res_features], y = y_train, 
                                        epochs=EPOCHS, batch_size = hp_batchsize*N_GPUs, callbacks=callbacks, verbose= 1)
                    times_dict['lr_{}_bz_{}'.format(lrate, hp_batchsize)] = np.round((time.time()-tic)/60)
                    

                    history = pmodel.history.history['loss']
                    # include save_format as we use a custom loss (see load_model below for futher syntax)
                    pmodel.save(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize)), save_format = 'tf')
                    np.save(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize)), history)

                elif os.path.exists(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize))):
                    pmodel = load_model(os.path.join(path_model, r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize)), compile=False)
                    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = optimizer)
                    print(' ... Model with lr {} and bz {} loaded.'.format(lrate, hp_batchsize))

                    with open(os.path.join(path_model, r'model_lr_{}_bz_{}_hist.npy'.format(lrate, hp_batchsize)), 'rb') as f:
                        history = np.load(f, allow_pickle=True)

                    # did exploding gradients occur during training? check loaded model
                    bool_nan_val = check_exploded_gradients(pmodel)
                    if bool_nan_val == True:
                        print('Model with nan-paramter-values! ', r'model_lr_{}_bz_{}.h5'.format(lrate, hp_batchsize))
                else:
                    print('train-flag is off and model with lr {} and bz {} not yet trained.'.format(lrate, hp_batchsize))
                    
                # plot training history
                if baseline_sex != 'none':
                    loss = pmodel.evaluate(x=[x_train[:,:,base_features], x_train[:,:,res_features]], y = y_train, batch_size=1024)
                    y_pred = pmodel.predict([x_test[:,:,base_features], x_test[:,:,res_features]])
                else:
                    loss = pmodel.evaluate(x=x_train[:,:,res_features], y = y_train, batch_size=1024)
                    y_pred = pmodel.predict(x_test[:,:,res_features])
                
                # ----------
                # intrinsic economic validation
                
                P_0 = -eval_loss_raw(y_true = y_test, y_pred = y_pred).numpy().reshape((-1,1))
                
                
                # premium duration = t/ZahlweiseNum (bspw. 3/(1/12) = 36 [iterations])
                # Note: we need to use raw, i.e. non-scaled values, here
                premium_duration = x_test_raw[:,0,2].reshape((-1,1))/x_test_raw[:,0,3].reshape((-1,1))
                P_true = x_train_raw[:,0,-1].reshape((-1,1))

                zill_factor = neural_premium_zillmerisation(y = y_pred, x= [x_test[:,:, base_features], x_test[:,:, res_features]], freq = x_test[:,0,3], v = GAMMA, 
                                                            t_iter=premium_duration, alpha=ALPHA, beta=BETA) #model = pmodel,
                P_pred = P_0/zill_factor
                rel_error = (P_true-P_pred)/P_true
                q = 0.995
                alphas = [0, 1-q, 0.1, 0.25, 0.5, 0.75, 0.9, q, 1]
                quantiles = np.round(np.quantile(rel_error, alphas),2)
                # metrics_tag = str(int(loss)) + '/' + str(np.round(np.mean(rel_error),4)) + ' ('+ str(np.round(np.min(rel_error),2)) + '/' + str(np.round(np.quantile(rel_error, 1-q),2)) + '/' + str(np.round(np.quantile(rel_error, q),2)) + '/' + str(np.round(np.max(rel_error),2)) + ')' 
                metrics_tag = str(int(loss)) + ' - ' + '/'.join([str(q) for q in quantiles])
                # ----------
                print('t loss/ rel. error quantiles')
                print('\t' + metrics_tag)
                plt.plot(np.arange(len(np.array(history).flatten())), np.array(history).flatten(), label='lr: {}, bz: {}, l/re: {}'.format(lrate, hp_batchsize, metrics_tag))
    
    # save training times
    if bool_train == True:
        with open(os.path.join(path_model, 'training_times.p'), 'wb') as fp:
            pickle.dump(times_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(path_model, 'training_times.p'), 'rb') as fp:
            print('ValueError: unsupported pickle protocol: 5')
            print('We skip loading training times for now!')
            #times_dict = pickle.load(fp)


    # display collective plot for all training-settings loaded in loop
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(path_model, 'HP_seach_lrates.png'), bbox_inches='tight')
    plt.close()

def run_finetuning(baseline_sex, batchsize = 128, lr = 10**-4, epochs = 1000):

    if baseline_sex == 'male':
        path_model = path_models_resnet_hpsearch_male
    elif baseline_sex == 'female':
        path_model = path_models_resnet_hpsearch_female
    elif baseline_sex == 'none':
        path_model = path_models_resnet_hpsearch_none
    else:
        raise ValueError('Unknown Option for baseline_sex.')
    strategy = tf.distribute.MirroredStrategy()
    N_GPUs = strategy.num_replicas_in_sync

    #### load data
    with open(os.path.join(path_data, 'x_train_raw.npy'), 'rb') as f:
        x_train_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_train.npy'), 'rb') as f:
        x_train = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)

    with open(os.path.join(path_data, r'x_test_raw.npy'), 'rb') as f:
        x_test_raw = np.load(f, allow_pickle=True)
        print('\t .. x_test_raw loaded. ', type(x_test_raw), ' of shape ', x_test_raw.shape)
    with open(os.path.join(path_data, r'x_test.npy'), 'rb') as f:
        x_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        print('\t .. x_test loaded.', type(x_test), ' of shape ', x_test.shape)
    # load cash-flow values (test-data w/o premium-related payments)
    with open(os.path.join(path_data, r'y_test.npy'), 'rb') as f:
        y_test = np.load(f, allow_pickle=True)#.astype(np.float64)
        print('\t .. y_test loaded.', type(y_test), ' of shape ', y_test.shape)

    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]

    pmodel = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)
    pmodel.trainable = True
    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = Adam(learning_rate= lr))

    print(pmodel.summary())

    pmodel.fit(x=[x_train[:,:, base_features], x_train[:,:, res_features]], y = y_train, epochs=epochs, batch_size = batchsize*N_GPUs,  # high learning rate to avoid purely local adjustments
                callbacks=[tf.keras.callbacks.LearningRateScheduler(exp_decay_scheduler_finetuning), ES()], verbose= 1)
    y_pred = pmodel.predict([x_test[:,:,base_features], x_test[:,:,res_features]])
 # ----------
    loss = pmodel.evaluate(x=[x_train[:,:, base_features], x_train[:,:, res_features]], y = y_train, batch_size=1024)

    # intrinsic economic validation
    P_0 = -eval_loss_raw(y_true = y_test, y_pred = y_pred).numpy().reshape((-1,1))
    
    
    # premium duration = t/ZahlweiseNum (bspw. 3/(1/12) = 36 [iterations])
    # Note: we need to use raw, i.e. non-scaled values, here
    premium_duration = x_test_raw[:,0,2].reshape((-1,1))/x_test_raw[:,0,3].reshape((-1,1))
    P_true = x_train_raw[:,0,-1].reshape((-1,1))

    zill_factor = neural_premium_zillmerisation(y = y_pred, x= [x_test[:,:, base_features], x_test[:,:, res_features]], freq = x_test[:,0,3], v = GAMMA, 
                                                t_iter=premium_duration, alpha=ALPHA, beta=BETA) #model = pmodel,
    P_pred = P_0/zill_factor
    rel_error = (P_true-P_pred)/P_true
    q = 0.995
    alphas = [0, 1-q, 0.1, 0.25, 0.5, 0.75, 0.9, q, 1]
    quantiles = np.round(np.quantile(rel_error, alphas),2)
    # metrics_tag = str(int(loss)) + '/' + str(np.round(np.mean(rel_error),4)) + ' ('+ str(np.round(np.min(rel_error),2)) + '/' + str(np.round(np.quantile(rel_error, 1-q),2)) + '/' + str(np.round(np.quantile(rel_error, q),2)) + '/' + str(np.round(np.max(rel_error),2)) + ')' 
    metrics_tag = str(int(loss)) + ' - ' + '/'.join([str(q) for q in quantiles])
    # ----------
    print('t loss/ rel. error quantiles ' + str(quantiles))
    print('\t' + metrics_tag)

    pmodel.save(os.path.join(path_model, r'model_finetuned.h5'), save_format = 'tf')




if __name__ ==  '__main__':

    #----------------------
    # settings
    flag_training = False
    flag_finetuning = True
    widths = [50, 50, 50, 50, 50]
    #----------------------

    assert( (flag_training != True) or (flag_finetuning != True), 'either train new models or fine-tune existing') # either train new models or fine-tune existing

    for gender in ['female', 'male']:
        if not flag_finetuning:
            run_manual_HPS(baseline_sex=gender, bool_train=flag_training, widths_lst = widths)
        else:
            # optional: finetuning
            # note: model_best.h5 has to be manually selected before this can be run
            # motivation for non-automated selection: human expertise to evaluate trade-off between variance and bias
            try:
                run_finetuning(gender)
            except Exception as e:
                print('model_best.h5 might not be available yet. Error: ')
                print(e)