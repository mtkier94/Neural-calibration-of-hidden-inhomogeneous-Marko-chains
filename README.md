# Neural calibration of hidden inhomogeneous Marko chains -- Information decompression in life insurance
Code and data accompanying the corresponding paper by Kiermayer, M. and Weiß, C. <br/>
The data is provided by msg life central europe gmbh. <br/>
Code was tested for python3.8(.16) and tensorflow==2.7.0, see requirements.txt for more detail. <br/>


## Description of python-scripts
The project contains multiple main-files to be run step-by-step. This is inevidible given the application of transfer learning/ pre-training. <br/>

### Fast-track
The fastest way to replicate results is: <br/>
- run create_data.py to locally create train, test and crossvalidation data <br/>
- run model_evaluation.py to obtain the numerical results (section 5 of the paper)  <br/>
- [optional] run ablation_study_crossval.py to perform a 2-fold crossvalidation (Appendix A.1 of the paper) <br/>

This fast-track approach will re-use all provided checkpoints for pi_base and pi_res (for both gender baselines). <br/>

### End-to-end replication
Alternatively, to replicate all steps or adjust the model to your custom datset (of term life contracts):  <br/>
- run create_data.py to locally create train, test and crossvalidation data <br/>
- run pi_base_train.py to train a new baseline model pi_base. Note that after re-training pi_base, the model pi_res must be re-trained as well due to their inter-dependency. <br/>
      * The repository contains multiple checkpoints for both gender baselines, see e.g. ./models/baseline/hp_search_male . To run a hyperparam search for pi_base, enable training in line 173 (val 'bool_train') and adjust hyperparams in lines 69-73. <br/>
- run pi_res_train_hptuning_hyperopt.py and/ or pi_res_train_hptuning_manual.py to perform a hyperparameter search for pi_res. <br/>
- run model_evaluation.py to obtain the numerical results (section 5 of the paper)  <br/>
     * Note: If models from manual hp-tuning are to be analyzed, the respective model needs to be renamed to "model_best.h5" and the path has to be set accordingly, i.e. currently for the five hidden layers we access the sub-directory "./models/resnet/hp_search_[gender]_50_50_50_50_50".<br/>
- [optional] run ablation_study_crossval.py to perform a 2-fold crossvalidation (Appendix A.1 of the paper) <br/>
  
The file "global_vars.py" contains general information which includes <br/>
  - Paths for saving/ loading data<br/>
  - Hyperparameters such as cost structur and discount rate  (for cash flows) <br/>


## Comments on the data (see ./data)


1) The raw data is provided by msg life central europe gmbh and can be found in "./data/msg_life/Tarifierung_RI_2017.csv". All other data in "./data/msg_life/" is derived from the raw data by data processing. The copyright for this data remains with msg life central europe gmbh, see LICENSE in "./data/msg_life/" <br/>

2) Note: train- and test-data ("[x/y]_[train/test]*.npy") are not uploaded to './data/msg_life/' due to size-contraints (>100MB) on github induced by the time-series-format. However, running "create_data.py" will create all files required for training and/or evaluation.

3) The .csv-files for the DAV 2008T tables (male or female) at "./data/DAV_tables" are obtained via the R package "mortality-tables", see https://gitlab.open-tools.net/R/r-mortality-tables/-/blob/master/data-raw/Germany_Endowments_DAV-T.xlsx <br/>

  

## Description of the folder structure:
  
  - functions: various helper functions, e.g. for actuarial computations, data processing, hp-tuning and more <br/>
  - models: results of manual hyperparameter tuning/ grid-search <br/>
        * baseline: baseline-models for fitting the DAV 2008T mortality rates; sub-folders for male and female version <br/>
        * resnet: residual-model for boosting the baseline for the specific data at hand; sub-folders indicate gender and model widths/ depths <br/>
  - models-hyperopt: results of automated hp-tuning with the 'hyperopt'-package <br/>
        * male: male results, i.e. parametrizations of trials, histories, but also plots <br/>
        * female: structure analog to male-folder, but for female baseline <br/>
   - test-files: various test-scripts to test consistency e.g. of weight transfer from FFN to SimpleRNN and the custom tf-loss function<br/>
