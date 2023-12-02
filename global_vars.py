import os

path_project = os.path.dirname(os.path.realpath(__file__))
path_data = os.path.join(os.path.join(path_project, 'data'), 'msg_life')
path_dav = os.path.join(os.path.join(path_project, 'data'), 'DAV_tables')
path_data_backtesting = os.path.join(path_data, 'backtesting')

path_hyperopt = os.path.join(path_project, 'models_hyperopt')
path_hyperopt_male = os.path.join(path_hyperopt, 'male')
path_hyperopt_female = os.path.join(path_hyperopt, 'female')
path_hyperopt_none = os.path.join(path_hyperopt, 'none')

path_models = os.path.join(path_project, 'models')
path_models_baseline = os.path.join(path_models, 'baseline')
path_models_baseline_plots = os.path.join(path_models_baseline, 'plots')
path_models_baseline_hpsearch_male = os.path.join(path_models_baseline, 'hp_search_male')
path_models_baseline_hpsearch_female = os.path.join(path_models_baseline, 'hp_search_female')
path_models_baseline_hpsearch_female = os.path.join(path_models_baseline, 'hp_search_none')
path_models_baseline_transfer = os.path.join(path_models_baseline, 'transfer')
path_models_resnet = os.path.join(path_models, 'resnet')

# hparam-tuning paths (manual hypertuning, in contrast to tpe.algo in 'hyperopt' above)
path_models_resnet_hpsearch_male = os.path.join(path_models_resnet, 'hp_search_male')
path_models_resnet_hpsearch_female = os.path.join(path_models_resnet, 'hp_search_female')
path_models_resnet_hpsearch_none = os.path.join(path_models_resnet, 'hp_search_none')
path_functions = os.path.join(path_project, 'functions')
path_plots = os.path.join(path_project, 'plots')

T_MAX = 121 # maximum age, eventually used for scaling the age feature
ALPHA, BETA, GAMMA1, GAMMA2 = 0.025, 0.03, 0.001, 0.001 # cost factors
GAMMA = 1/1.0125 # discount factor (p.a.) for cash-flows
AGE_RANGE = (18, 66) # ages 18 - 66 are covered within the current data set
INIT_AGE_RANGE = (18, 60) # initial ages in the current data set