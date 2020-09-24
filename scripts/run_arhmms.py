import argparse
import copy

from flygenvectors.dlc import preprocess_and_split_data, shuffle_data
import flygenvectors.ssmutils as ssmutils


def run_main(args):

    # -------------------------------------
    # load and preprocess data
    # -------------------------------------

    # define sessions
    if dataset[0] == '2':
        # guess an individual dataset based on naming convention "20xx_xx_xx_flyx"
        expt_ids = [dataset]
    elif dataset == 'double':
        expt_ids = ['2019_08_08_fly1', '2019_08_08_fly1_1']
    else:
        raise NotImplementedError

    # preprocessing directives
    preproc_list = {
        'filter': {'type': 'savgol', 'window_size': 5, 'order': 2},
        'unitize': {},  # scale labels in [0, 1]
    }

    label_obj = preprocess_and_split_data(sess_ids, preproc_list, algo='dgp', load_from='h5')

    # shuffle data
    datas_tr, tags_tr, _ = shuffle_data(label_obj, dtype='train')
    datas_val, tags_val, _ = shuffle_data(label_obj, dtype='val')
    datas_test, tags_test, _ = shuffle_data(label_obj, dtype='test')

    D = data_tr[0].shape[1]

    # -------------------------------------
    # define models
    # -------------------------------------

    # # hyperparams that are same across all models
    # observations = args.observations  # 'ar' | 'diagonal_ar' | 'diagonal_robust_ar'
    # init_type = args.init  # 'arhmm' | 'kmeans' | 'em' | 'em-exact'
    # fit_method = args.fit_method  # 'em' | 'stochastic-em'
    #
    # # params that define models
    # # n_states = [2, 4, 6, 8, 10, 12, 16, 20, 32]
    # # n_states = [2, 4, 8, 12, 16, 24]
    # n_states = [2, 4, 8, 16, 24]
    #
    # n_lags_standard = [2] if args.stationary else []
    # n_lags_sticky = [2] if args.sticky else []
    # n_lags_recurrent = [2] if args.recurrent else []
    # kappas = [1e4, 1e6]
    K = args.K
    lags = args.lags
    num_restarts = args.num_restarts
    num_iters = args.num_iters

    init_types = args.inits.split(';')

    # -------------------------------------
    # fit models
    # -------------------------------------
    if args.fit_arhmm:

        print('---------------------------')
        print('Fitting ARHMM with EM')
        print('---------------------------')

        method = 'em'  # 'em' | 'stochastic_em_adam' | 'stochastic_em_sgd' (non-conjugate)
        obs = 'ar'  # 'ar' | 'hierarchical_ar'
        transitions = 'stationary'  # 'stationary' | 'hierarchical_stationary'

        for it in init_types:
            expt_dir = ssmutils.get_expt_dir(dirs['results'], expt_ids)
            save_path = os.path.join(
                dirs['results'], expt_dir, 'multi-session_%s-init_%s' % (it, method))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ssmutils.fit_with_random_restarts(
                K, D, obs, lags, datas_tr, tags=tuple(tags_tr), transitions=transitions,
                num_restarts=num_restarts, num_iters=num_iters,
                method=method, save_path=save_path, init_type=it)

    if args.fit_harhmm_bem:

        print('---------------------------')
        print('Fitting hARHMM with BEM')
        print('---------------------------')

        method = 'em'
        obs = 'hierarchical_ar'  # 'ar' | 'hierarchical_ar'
        transitions = 'hierarchcial_stationary'  # 'stationary' | 'hierarchical_stationary'
        cond_var_A = args.cond_var_A

        for it in init_types:
            expt_dir = ssmutils.get_expt_dir(dirs['results'], expt_ids)
            save_path = os.path.join(
                dirs['results'], expt_dir, 'multi-session_%s-init_bem_condA=%1.1e' % (
                    it, cond_var_A))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ssmutils.fit_with_random_restarts(
                K, D, obs, lags, datas_tr, tags=tuple(tags_tr), transitions=transitions,
                num_restarts=num_restarts, num_iters=num_iters,
                method=method, save_path=save_path, init_type=it)

    if args.fit_harhmm_sem:

        print('---------------------------')
        print('Fitting hARHMM with SEM-conj')
        print('---------------------------')

        method = 'stochastic_em_conj'
        obs = 'hierarchical_ar'  # 'ar' | 'hierarchical_ar'
        transitions = 'hierarchical_stationary'  # 'stationary' | 'hierarchical_stationary'
        cond_var_A = args.cond_var_A
        rates = [float(r) for r in args.sem_lrs.split(';')]

        for it in init_types:
            for rate in rates:
                print('---------------------------')
                print('forgetting rate = %f' % rate)
                print('---------------------------')
                expt_dir = ssmutils.get_expt_dir(dirs['results'], expt_ids)
                save_path = os.path.join(
                    dirs['results'], expt_dir,
                    'multi-session_%s-init_sem_rate=%1.2f_condA=%1.1e' % (it, rate, cond_var_A))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                ssmutils.fit_with_random_restarts(
                    K, D, obs, lags, datas_tr, tags=tuple(tags_tr), transitions=transitions,
                    num_restarts=num_restarts, num_iters=num_iters,
                    method=method,
                    stochastic_mstep_kwargs=dict(forgetting_rate=rate),
                    save_path=save_path, init_type=it, cond_var_A=cond_var_A)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='double', type=str)

    parser.add_argument('--fit_arhmm', action='store_true', default=False)
    # parser.add_argument('--fit_arhmm_ind', action='store_true', default=False)
    parser.add_argument('--fit_harmm_bem', action='store_true', default=False)
    parser.add_argument('--fit_harmm_sem', action='store_true', default=False)

    # init types: random, kmeans, kmeans-diff, umap-kmeans, umap-kmeans-diff
    parser.add_argument('--inits', default='kmeans', type=str)
    parser.add_argument('--sem_lrs', default='0.9', type=str)

    parser.add_argument('--K', default=8, type=int)
    parser.add_argument('--lags', default=3, type=int)
    parser.add_argument('--num_restarts', default=5, type=int)
    parser.add_argument('--num_iters', default=100, type=int)
    parser.add_argument('--cond_var_A', default=1e-4, type=float)

    namespace, _ = parser.parse_known_args()
    run_main(namespace)
