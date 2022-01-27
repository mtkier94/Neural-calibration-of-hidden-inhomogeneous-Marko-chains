from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from functions.tf_loss_custom import compute_loss_mae
from functions.tf_model_res import combine_models, create_mortality_res_net, create_mortality_res_net_special

def get_search_space(input_res: int, path_baseline: str):

    '''
    Return the search space for a hyper-parameter search using hyperopt.

    Inputs:
    -------
        input_res: numer of input features to res-net

    Outputs:
    --------
        dictionary with the respective hyperparams and ranges for hyperparam-search
    '''

    hps =  {
        'input_res': input_res,
        # note: if the depth changes, then e.g. width_3 will change from FNN to RNN type layer (-> tricky for 'hyperopt')
        'dense_depth': hp.choice('dense_depth', [3,4,5,6]),
        'dense_width': scope.int(hp.quniform('dense_width', 30, 60, 10)),
        'recurrent_depth': hp.choice('recurrent_depth', [1,2]),
        'recurrent_width': scope.int(hp.quniform('recurrent_width', 30, 60, 10)),
        # 'width_3': scope.int(hp.quniform('width_3', 30, 60, 10)),
        #'width_4': scope.int(hp.quniform('width_4', 30, 60, 10)),
        'lrate': 10**hp.quniform('lrate', -4, -2, 0.5),
        'batch_size': hp.choice('batch_size', [32, 64]),# 128]), #, 256]), # manual HPSearch showed low per-GPU-BZ to be better
        'l2_reg': 0, # set to 0 to increase efficiency of hpsearch; no regularization required or desired # hp.uniform('l2_reg', 0, 2),
        'path_baseline': path_baseline # required for loading base_line in objective-method for 'hyperopt'-search
    }

    return hps

def hpsearch_model(**params):
    '''
    LEGACY FUNCTION
    Helper function to map parameters of the hp-search-space to create the joint model, with base- and residual-component.
    '''

    # baseline model (pretrained)
    pmodel_base = load_model(params['path_baseline'])
    # unpack parameters

    dense_widths = [params['dense_width']]*params['dense_depth']
    recurrent_widths = [params['recurrent_width']]*params['recurrent_depth']

    optimizer = Adam(learning_rate=params['lrate'], clipnorm=100.0)
    l2_reg = params['l2_reg']
    res_features = params['input_res']
    # create residual-model, combine it with the base-model and compile with custom loss
    pmodel_res = create_mortality_res_net_special(dense_layers= dense_widths, recurrent_layers= recurrent_widths, param_l2_penalty=l2_reg, input_shape=(None, res_features), n_out=2)
    pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = True)
    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = optimizer)

    return pmodel

def hpsearch_model_old(**params):
    '''
    LEGACY FUNCTION
    Helper function to map parameters of the hp-search-space to create the joint model, with base- and residual-component.
    '''

    # baseline model (pretrained)
    pmodel_base = load_model(params['path_baseline'])
    # unpack parameters
    try:
        # if params['width_4'] implemented
        # Note: if depth fixed at 3, dropping this parameter is compuationally preferable (HPTuning will tune it without benefit)
        widths_lst = [params['width_1'],params['width_2'],params['width_3'],params['width_4']][0:params['depth']]
    except:
        widths_lst = [params['width_1'],params['width_2'],params['width_3']][0:params['depth']]
    optimizer = Adam(learning_rate=params['lrate'], clipnorm=100.0)
    l2_reg = params['l2_reg']
    res_features = params['input_res']
    # create residual-model, combine it with the base-model and compile with custom loss
    pmodel_res = create_mortality_res_net(hidden_layers=widths_lst, param_l2_penalty=l2_reg, input_shape=(None, res_features), n_out=2)
    pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = True)
    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = optimizer)

    return pmodel

def hpsearch_model_old(**params):
    '''
    LEGACY CODE
    Helper function to map parameters of the hp-search-space to create the joint model, with base- and residual-component.
    '''

    # baseline model (pretrained)
    pmodel_base = load_model(params['path_baseline'])
    # unpack parameters
    try:
        # if params['width_4'] implemented
        # Note: if depth fixed at 3, dropping this parameter is compuationally preferable (HPTuning will tune it without benefit)
        widths_lst = [params['width_1'],params['width_2'],params['width_3'],params['width_4']][0:params['depth']]
    except:
        widths_lst = [params['width_1'],params['width_2'],params['width_3']][0:params['depth']]
    optimizer = Adam(learning_rate=params['lrate'], clipnorm=100.0)
    l2_reg = params['l2_reg']
    res_features = params['input_res']
    # create residual-model, combine it with the base-model and compile with custom loss
    pmodel_res = create_mortality_res_net(hidden_layers=widths_lst, param_l2_penalty=l2_reg, input_shape=(None, res_features), n_out=2)
    pmodel = combine_models(pmodel_base, pmodel_res, bool_masking = True)
    pmodel.compile(loss = compute_loss_mae, metrics=None, optimizer = optimizer)

    return pmodel


