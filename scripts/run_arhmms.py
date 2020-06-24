import argparse
import copy

from flygenvectors.dlc import preprocess_and_split_data, shuffle_data
import flygenvectors.ssmutils as utils


def run_main(args):

    # -------------------------------------
    # load and preprocess data
    # -------------------------------------

    # define sessions
    sess_ids = ['2019_08_08_fly1']

    # preprocessing directives
    preprocess_list = {
        # 'filter': {'type': 'median', 'window_size': 3},
        'filter': {'type': 'savgol', 'window_size': 5, 'order': 2},
        # 'standardize': {}, # zscore labels
        'unitize': {},  # scale labels in [0, 1]
    }

    label_obj = preprocess_and_split_data(sess_ids, preprocess_list, algo='dgp', load_from='pkl')
    model_dir = utils.get_model_dir('dlc-arhmm', preprocess_list, final_str='')
    print('\nsaving models in the following directory: "%s"' % model_dir)

    # shuffle data
    data_tr, tags_tr, _ = shuffle_data(label_obj, dtype='train')
    data_val, tags_val, _ = shuffle_data(label_obj, dtype='val')
    data_test, tags_test, _ = shuffle_data(label_obj, dtype='test')

    D = data_tr[0].shape[1]

    # -------------------------------------
    # define models
    # -------------------------------------

    # hyperparams that are same across all models
    observations = args.observations  # 'ar' | 'diagonal_ar' | 'diagonal_robust_ar'
    init_type = args.init  # 'arhmm' | 'kmeans'
    fit_method = args.fit_method  # 'em' | 'stochastic_em'

    # params that define models
    # n_states = [2, 4, 6, 8, 10, 12, 16, 20, 32]
    # n_states = [2, 4, 8, 12, 16, 24]
    n_states = [2, 4, 8, 16, 24]

    n_lags_standard = [2] if args.stationary else []
    n_lags_sticky = [2] if args.sticky else []
    n_lags_recurrent = [2] if args.recurrent else []
    kappas = [1e4, 1e6]

    model_kwargs = utils.collect_models(
        n_lags_standard, n_lags_sticky, n_lags_recurrent, kappas, observations)

    if init_type == 'arhmm':
        model_dir_ext = model_dir + '-2-state-init'
    else:
        model_dir_ext = model_dir
    if fit_method == 'em':
        pass
    else:
        model_dir_ext += '_%s' % fit_method

    # -------------------------------------
    # fit models
    # -------------------------------------
    fit_kwargs = {
        'save': True,
        'load_if_exists': True,
        'expt_id': sess_ids,
        'model_dir': model_dir_ext,
        'save_dir': None}

    # iterate over model types
    n_sess = len(sess_ids)
    all_results = {}
    for model_name, kwargs in model_kwargs.items():
        model_results = {}

        # add hierarchical tags
        if n_sess > 1:
            kwargs_ = {
                **kwargs,
                'hierarchical_transition_tags': list(range(n_sess)),
                'hierarchical_observation_tags': list(range(n_sess))}
            fit_method = 'stochastic_em'
        else:
            kwargs_ = copy.deepcopy(kwargs)
            tags_tr, tags_val, tags_test = None, None, None

        # iterate over discrete states
        for K in n_states:
            print('Fitting %s with %i states' % (model_name, K))
            model_results[K] = utils.fit_model(
                n_states=K, data_dim=D, input_dim=0, model_kwargs=kwargs_,
                data_tr=data_tr, data_val=data_val, data_test=data_test,
                tags_tr=tags_tr, tags_val=tags_val, tags_test=tags_test,
                init_type=init_type, fit_method=fit_method, fit_kwargs=fit_kwargs)

        all_results[model_name] = model_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--stationary', action='store_true', default=False)
    parser.add_argument('--sticky', action='store_true', default=False)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--recurrent_only', action='store_true', default=False)

    parser.add_argument('--observations', default='diagonal_robust_ar', type=str)
    parser.add_argument('--init', default='arhmm', type=str)
    parser.add_argument('--fit_method', default='em', type=str)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)

