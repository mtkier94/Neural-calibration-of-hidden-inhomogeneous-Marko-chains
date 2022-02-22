# Neural calibration of hidden inhomogeneous Marko chains -- Information decompression in life insurance
Code and data accompanying the corresponding paper by Kiermayer, M. and Weiß, C. <br/>
Preprint available at https://arxiv.org/abs/2201.02397. <br/>
The data is provided by msg life central europe gmbh. <br/>


## Description of python-scripts
Given the application of transfer learning/ pre-training, this project contains multiple main-files to be run step-by-step. The order the files are to be run in is indicated by the leading number of the python files. <br/>

These steps include:
  - Data generation and preprocessing (0_main_data_processing.py) <br/>
  - Exploratory data analysis (1_main_eda.py)
  - Configuration of the baseline model (2_main_baseline.py)<br/>
  - Configuration of the residual model <br/>
        * Option a): manual hp-tuning (3a_main_hp_manual.py)<br/>
        * Option b): automated hp-tuning: (3b_main_hyperopt.py)<br/>
  - Analyze results, create heatmaps for policyholder-/ risk-types, intrinsic model validation (4_main_analysis_results.py) <br/>
        * Note: If models from manual hp-tuning are to be analyzed, the respective model needs to be renamed to "model_best.h5" and the path has to be set accordingly, i.e. currently for the five hidden layers we access the sub-directory "./models/resnet/hp_search_[gender]_50_50_50_50_50".<br/>
  
General information in "global_vars.py" includes <br/>
  - Paths for saving/ loading data<br/>
  - Hyperparameters such as cost structur and discount rate  (for cash flows) <br/>


## Comments on the data (see ./data)

1) The raw data is provided by msg life central europe gmbh and can be found in "./data/msg_life/Tarifierung_RI_2017.csv". All other data in "./data/msg_life/" is derived from the raw data by data processing. The copyright for this data remains with msg life central europe gmbh, see LICENSE in "./data/msg_life/" <br/>

2) Note: train- and test-data ("[x/y]_[train/test]*.npy") are not uploaded to './data/msg_life/' due to size-contraints (>100MB) on github induced by the time-series-format. However, running "0_main_data_processing.py" will create these files.

3) We do not include the .csv-files for the DAV 2008T tables (male or female) in this repository in "./data/DAV_tables", as we do not own copyright for it.<br/>
However, the data is available on the website http://www.aktuar.de/, see https://aktuar.de/unsere-themen/lebensversicherung/sterbetafeln/2018-10-05_DAV-Richtlinie_Herleitung_DAV2008T.pdf <br/>
Alternatively, one may consider the R package "mortality-tables" to retrieve the data, see https://gitlab.open-tools.net/R/r-mortality-tables/-/blob/master/data-raw/Germany_Endowments_DAV-T.xlsx <br/>

  

## Description of the folder structure:
  
  - functions: various helper functions, e.g. for actuarial computations, data processing, hp-tuning and more <br/>
  - models: results of manual hyperparameter tuning/ grid-search <br/>
        * baseline: baseline-models for fitting the DAV 2008T mortality rates; sub-folders for male and female version <br/>
        * resnet: residual-model for boosting the baseline for the specific data at hand; sub-folders indicate gender and model widths/ depths <br/>
  - models-hyperopt: results of automated hp-tuning with the 'hyperopt'-package <br/>
        * male: male results, i.e. parametrizations of trials, histories, but also plots <br/>
        * female: structure analog to male-folder, but for female baseline <br/>
   - test-files: various test-scripts to test consistency e.g. of weight transfer from FFN to SimpleRNN and the custom tf-loss function<br/>
