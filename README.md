# Neural-calibration-of-hidden-inhomogeneous-Marko-chains
Code and data accompanying the corresponding paper by Kiermayer, M. and Weiß, C.


## Description of python-scripts
Given the application of transfer learning/ pre-training, this project contains multiple main-files to be run step-by-step. The order the files are to be run in is indicated by the leading number of the python files. 

These steps include:
  - Data generation and preprocessing ( 0_main_data_processing.py )
  - Exploratory data analysis ( 1_main_eda.py )
  - Configuration of the baseline model ( 2_main_baseline.py )
  - Configuration of the residual model
        * Option a): manual hp-tuning: 3a_main_hp_manual.py
        * Option b): automated hp-tuning: 3b_main_hyperopt.py
  - Analyze results, create heatmaps for policyholder-/ risk-types, intrinsic model validation ( 4_main_analysis_results.py )
  
General information in "global_vars.py" includes
  - Paths for saving/ loading data
  - Hyperparameters such as cost structur and discount rate  (for cash flows)
  

## Description of the folder structure:
  
  - functions: various helper functions, e.g. for actuarial computations, data processing, hp-tuning and more
  - models: results of manual hyperparameter tuning/ grid-search
        * baseline: baseline-models for fitting the DAV 2008T mortality rates; sub-folders for male and female version
        * resnet: residual-model for boosting the baseline for the specific data at hand; sub-folders indicate gender and model widths/ depths
  - models-hyperopt: results of automated hp-tuning with the 'hyperopt'-package
        * male: male results, i.e. parametrizations of trials, histories, but also plots
        * female: structure analog to male-folder, but for female baseline
   - test-files: various test-scripts to test consistency e.g. of weight transfer from FFN to SimpleRNN and the custom tf-loss function
