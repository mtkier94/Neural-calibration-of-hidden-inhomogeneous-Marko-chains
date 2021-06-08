import os

path_project = os.path.dirname(os.path.realpath(__file__))
path_data = os.path.join(path_project, 'data')
path_models = os.path.join(path_project, 'models')
path_models_baseline = os.path.join(path_models, 'baseline')
path_models_baseline_plots = os.path.join(path_models_baseline, 'plots')
path_models_baseline_hpsearch = os.path.join(path_models_baseline, 'hp_search')
path_models_baseline_transfer = os.path.join(path_models_baseline, 'transfer')
path_models_resnet = os.path.join(path_models, 'resnet')
path_models_resnet_wo_padding = os.path.join(path_models_resnet, 'wo_padding')
path_models_resnet_wo_padding_hpsearch = os.path.join(path_models_resnet_wo_padding, 'hp_search')
path_models_resnet_with_padding = os.path.join(path_models_resnet, 'with_padding')
path_models_resnet_with_padding_hpsearch_male = os.path.join(path_models_resnet_with_padding, 'hp_search_male')
path_models_resnet_with_padding_hpsearch_female = os.path.join(path_models_resnet_with_padding, 'hp_search_male')
path_functions = os.path.join(path_project, 'functions')
path_plots = os.path.join(path_project, 'plots')

T_MAX = 121 # maximum age, eventually used for scaling the age feature
GAMMA = 1/1.02 # discount factor (p.a.)

AGE_RANGE = (18, 67) # ages 19 - 66 are covered within the current data set

if __name__ == '__main__':
    pass