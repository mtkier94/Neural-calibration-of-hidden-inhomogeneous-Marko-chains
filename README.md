# Neural calibration of hidden inhomogeneous Marko chains -- Information decompression in life insurance
Code and data accompanying the corresponding paper by Kiermayer, M. and Weiß, C. <br/>
The data is provided by msg life europe gmbh, see the MIT-license.<br/>


## Description of python-scripts
Given the application of transfer learning/ pre-training, this project contains multiple main-files to be run step-by-step. The order the files are to be run in is indicated by the leading number of the python files. <br/>

These steps include:
  - Data generation and preprocessing (0_main_data_processing.py)<br/>
  - Exploratory data analysis (1_main_eda.py)
  - Configuration of the baseline model ( 2_main_baseline.py )<br/>
  - Configuration of the residual model <br/>
        * Option a): manual hp-tuning (3a_main_hp_manual.py)<br/>
        * Option b): automated hp-tuning: (3b_main_hyperopt.py)<br/>
  - Analyze results, create heatmaps for policyholder-/ risk-types, intrinsic model validation (4_main_analysis_results.py) <br/>
        * Note: If models from manual hp-tuning are to be analyzed, the respective model needs to be renamed to 'model_best.h5' and the path has to be set accordingly, i.e. currently for the five hidden layers we access the sub-directory './models/resnet/hp_search_[gender]_50_50_50_50_50'.<br/>
  
General information in "global_vars.py" includes<br/>
  - Paths for saving/ loading data<br/>
  - Hyperparameters such as cost structur and discount rate  (for cash flows)<br/>


## Comments on the data (see ./data)

The raw data is contained in 'Tarifierung_RI_2017.csv'. All other data is derived from the raw data by data processing.<br/>

Note: train- and test-data ('[x/y]_[train/test].npy') are not uploaded due to size-contrainsts (>100MB) induced by the time-series-format. However, running "0_main_data_processing.py" will create these files.
  

## Description of the folder structure:
  
  - functions: various helper functions, e.g. for actuarial computations, data processing, hp-tuning and more <br/>
  - models: results of manual hyperparameter tuning/ grid-search <br/>
        * baseline: baseline-models for fitting the DAV 2008T mortality rates; sub-folders for male and female version <br/>
        * resnet: residual-model for boosting the baseline for the specific data at hand; sub-folders indicate gender and model widths/ depths <br/>
  - models-hyperopt: results of automated hp-tuning with the 'hyperopt'-package <br/>
        * male: male results, i.e. parametrizations of trials, histories, but also plots <br/>
        * female: structure analog to male-folder, but for female baseline <br/>
   - test-files: various test-scripts to test consistency e.g. of weight transfer from FFN to SimpleRNN and the custom tf-loss function<br/>
