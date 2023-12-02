import numpy as np
import pickle 
import os, time
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
# import tensorflow_addons as tfa


from functions.tf_loss_custom import compute_loss_mae, eval_loss_raw
from functions.tf_model_res import combine_models, create_mortality_res_net
from functions.sub_backtesting import check_exploded_gradients
from functions.sub_actuarial import neural_premium_zillmerisation

from pi_res_train_hptuning_manual import run_manual_HPS
from model_evaluation import run_econom_eval

from global_vars import ALPHA, BETA, GAMMA, path_data, path_models_baseline_transfer
from global_vars import path_models_resnet_hpsearch_male, path_models_resnet_hpsearch_female, path_models_resnet_hpsearch_none


if __name__ ==  '__main__':

    #----------------------
    # settings
    flag_training = False
    # flag_finetuning = True
    widths = [50, 50, 50, 50, 50]
    #----------------------

    for gender in ['female', 'male']:
        if flag_training:
            run_manual_HPS(baseline_sex=gender, bool_train=flag_training, widths_lst = widths, kfolds=2, HP_BZ=[32])
        else:
            print('----------------')
            print('skipping training .. ')
            print('The evaluation assumes for each gender baseline (i.e. male and female) model_best_cv_1.h5 and model_best_cv_2.h5 to be available in the folder for 2-fold crossvalidation, i.e. ./models/resnet/hp_search_{gender}_cv2/ . Those "best models" were picked manually based on their training loss during HPTuning.')
            print('----------------')

        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL) # supress postscript latency warnings when saving images in an .eps-format
        
        # run evaluation of each fold (performance measured on hold-out set)
        run_econom_eval(baseline_sex= gender, tuning_type= 'manual', 
                        path_tag='_50_50_50_50_50_cv2', kfolds=2)