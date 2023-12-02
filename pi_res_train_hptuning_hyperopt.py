from functions.sub_hyperopt import hpsearch_model
import numpy as np
import pickle 
import os, time, joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.plotting import main_plot_history



from functions.tf_loss_custom import compute_loss_mae
from functions.sub_hyperopt import hpsearch_model, get_search_space
from global_vars import path_data, path_models_baseline_transfer, path_hyperopt_female, path_hyperopt_male

trial_count = 0

def exp_decay_scheduler(epoch, lr):
    if (epoch >= 50) and (epoch%15==0):
        return lr*0.9
    else:
        return lr

def ES():
    return EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

def create_tf_dataset(X,y, batch_size):
    '''
    Transform Data into a tf.data.Dataset object, e.g. to avoid sharding data.
    Note:   The current model architecture uses a two-headed input for model_base and model_res.
            This requires a slightly more elaborate construction of a tf.data.Dataset by zipping indivual tf.data.Dataset objects
            
    Inputs:
    -------
        X:  List with two elements, each np.arrays of shape (N_batch, N_iterations, N_features_i)
        y:  Numpy array of shape (N_batch, N_iterations, 2)
        batch_size:     batch_size for tf.data.Dataset which will be applied and fixed throughout training
    Outputs:
        train_data: A tf.data.Dataset object in the style of (X, y)
    '''


    # zip the two-headed input as tuples
    train_data = tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(X[i]) for i in [0,1]))
    # also zip target values
    train_data = tf.data.Dataset.zip((train_data, tf.data.Dataset.from_tensor_slices(y)))
    train_data = train_data.shuffle(batch_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)
    return train_data



def run_main(baseline_sex = 'female', eval_nums = 32, bool_train = False, bool_finetune = False):


    '''
    Run hyperopt for the neural network.

    Inputs:
    -------
        baseline_sex:   string in ['male', 'female'], that indicates the DAV 2008T baseline
        bool_train:     bolean, whether training should be performed.
        bool_finetune:  boolean, whether to fine-tune the currently best model

    Outputs:
    --------
        all potential outputs, such as ANN_models (.h5-data), histories or the hyperopt object are saved automatically.
    '''

    EPOCHS = 2#1500
    tf_strategy = tf.distribute.MirroredStrategy()
    N_GPUs = tf_strategy.num_replicas_in_sync

    # option for processing data during training
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    if baseline_sex == 'male':
        path_model = path_hyperopt_male
    elif baseline_sex == 'female':
        path_model = path_hyperopt_female
    else:
        raise ValueError('Unknown Option for baseline_sex.')
    

    #### load data
    with open(os.path.join(path_data, 'x_train_raw.npy'), 'rb') as f:
        x_train_raw = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'x_train.npy'), 'rb') as f:
        x_train = np.load(f, allow_pickle=True)
    with open(os.path.join(path_data, 'y_train.npy'), 'rb') as f:
        y_train = np.load(f, allow_pickle=True)

    N_contracts, N_seq_pad, N_features = x_train.shape


    # select contract-features for res-net
    # recall format: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    res_features = [0,3,6,7]
    base_features = [0,3]

    def objective(params):
        '''
        Define the objective of the hparam-search.
        '''

        global trial_count
        tic = time.time()

        with tf_strategy.scope():
            pmodel = hpsearch_model(**params)
        del params['path_baseline']

        # linear scaling of lrate w.r.t. batch_size; default batch_size = 32 (based in manual HPTuning and lrate range)
        # this is suggested by https://doi.org/10.1007/978-3-030-01424-7_39
        # Note: Distributing training across multiple GPUs makes this rescaling even more relevant
        params['lrate'] = params['lrate']*params['batch_size']/32

        print(f'Trial {trial_count}, setting: ', params)
        try:
            tf_train = create_tf_dataset(X=[x_train[:,:,base_features], x_train[:,:, res_features]], y=y_train, batch_size=params['batch_size']*N_GPUs)
            history = pmodel.fit(x=tf_train, epochs = EPOCHS, callbacks = [LearningRateScheduler(exp_decay_scheduler), ES()], verbose = 2)
        except Exception as e:
            print('tf.data.Dataset approach aboarded.')
            raise ValueError(e)
            # exit()
            # history = pmodel.fit(x = [x_train[:,:,base_features], x_train[:,:, res_features]], y = y_train, batch_size= params['batch_size']*N_GPUs, epochs = EPOCHS, callbacks = [LearningRateScheduler(exp_decay_scheduler), ES()], verbose = 2)
        print('\t one round of fitting completed!')
        loss = compute_loss_mae(y_true = y_train, y_pred = pmodel.predict(x=[x_train[:,:,base_features], x_train[:,:, res_features]]))
        print('loss (via pmodel.predict): ', loss)
        loss = pmodel.evaluate(tf_train, verbose = 0)
        print(f'values: loss= {loss: .4f}')

        # save parametrization and history
        save_model(pmodel, filepath=os.path.join(path_model, r'model_trial_{}.h5'.format(trial_count)))
        pickle.dump( history.history['loss'], open( os.path.join(path_model, r'model_trial_hist_{}.pkl'.format(trial_count)), "wb" ) ) 
        trial_count += 1
        return {'loss': loss, 'status': STATUS_OK, 'eval_time': time.time()-tic, 'iterations': len(history.history['loss'])}


    if bool_train and not bool_finetune:
        try:
            trials = joblib.load(os.path.join(path_model, r'hyperopt.pkl'))
            print('Loading and continuing hyperoptimization.')
            N_past_trials = len(trials.losses())
        except:
            trials = Trials()
            N_past_trials = 0
   
        global trial_count
        trial_count = N_past_trials

        eval_nums = eval_nums + N_past_trials # include number of old trials
        search_space = get_search_space(input_res = len(res_features), 
                                path_baseline=os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))

        _ = fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals= eval_nums,
                trials= trials)  


        # save hyperopt object
        joblib.dump(trials, os.path.join(path_model, r'hyperopt.pkl'))
        # rename best model
        import shutil
        id_min = np.argmin(trials.losses())
        shutil.copy2(src=os.path.join(path_model, r'model_trial_{}.h5'.format(id_min)),
                        dst= os.path.join(path_model, r'model_best.h5'))
    elif bool_finetune: #--------------------------------------
        # finetune the currently best model and save it
        model_best = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)
        model_best.compile(loss = compute_loss_mae, metrics=['mae'], optimizer = Adam(learning_rate=10**-4))

        # retrieve values of hyperparamters;
        trials = joblib.load(os.path.join(path_model, r'hyperopt.pkl'))
        params = trials.best_trial['misc']['vals']
        search_space = get_search_space(input_res = len(res_features), 
                                path_baseline=os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))
        for hps in params.keys():
            params[hps] = params[hps][0] # remove list format from hparam
        hparams = space_eval(search_space, params)    

        try:
            tf_train = create_tf_dataset(X=[x_train[:,:,base_features], x_train[:,:, res_features]], y=y_train, 
                                        batch_size=hparams['batch_size']*N_GPUs)

            history = model_best.fit(x=tf_train, epochs = EPOCHS, callbacks = [ES()], verbose = 1)
        except Exception as e:
            print('tf.data.Dataset approach aboarded.')
            raise ValueError(e)
        
        # save fine-tuned model
        save_model(model_best, filepath=os.path.join(path_model, r'model_best_finetuned.h5'))
        
        
    else: #--------------------------------------
        # analyze results
        trials = joblib.load(os.path.join(path_model, r'hyperopt.pkl'))

        model_best = load_model(os.path.join(path_model, r'model_best.h5'), compile=False)

       
        search_space = get_search_space(input_res = len(res_features), 
                                path_baseline=os.path.join(path_models_baseline_transfer, r'rnn_davT{}.h5'.format(baseline_sex)))

        params = trials.best_trial['misc']['vals']
        for hps in params.keys():
            params[hps] = params[hps][0] # remove list format from hparam
        print(space_eval(search_space, params))

        # exit()
        main_plot_history(trials)


        # exit()
        # display history of all trials
        for i in range(eval_nums):
            hist = pickle.load( open( os.path.join(path_model, r'model_trial_hist_{}.pkl'.format(i)), "rb" ) ) 
            plt.plot(hist)
            plt.yscale('log')
        plt.show()



        
if __name__ == '__main__':

    #--------------------------
    # settings
    n = 16
    training_flag = False
    finetuning_flag = False
    #--------------------------

    for gender in ['female', 'male']:
        run_main(baseline_sex= gender, eval_nums= n, bool_train= training_flag, bool_finetune=finetuning_flag)
        