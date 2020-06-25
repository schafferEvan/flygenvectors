import os
import numpy as np
import pickle
from ssm import HMM
from ssm.messages import forward_pass
from scipy.special import logsumexp
from sklearn.metrics import r2_score
from flygenvectors.utils import get_subdirs


# -------------------------------------------------------------------------------------------------
# model fitting functions
# -------------------------------------------------------------------------------------------------

def collect_model_kwargs(
        n_lags_standard, n_lags_sticky, n_lags_recurrent, kappas, observations, fit_hmm=False):
    """Collect model kwargs."""

    model_kwargs = {}

    # add hmms with standard transitions
    if fit_hmm:
        model_kwargs['hmm'] = {
            'transitions': 'standard',
            'observations': 'gaussian'}

    # add models with standard transitions
    for lags in n_lags_standard:
        model_kwargs['arhmm-%i' % lags] = {
            'transitions': 'standard',
            'observations': observations,
            'observation_kwargs': {'lags': lags}}

    # add models with sticky transitions
    for lags in n_lags_sticky:
        for kappa in kappas:
            kap = int(np.log10(kappa))
            model_kwargs['arhmm-s%i-%i' % (kap, lags)] = {
                'transitions': 'sticky',
                'transition_kwargs': {'kappa': kappa},
                'observations': observations,
                'observation_kwargs': {'lags': lags}}

    # add models with recurrent transitions
    for lags in n_lags_recurrent:
        model_kwargs['rarhmm-%i' % lags] = {
            'transitions': 'recurrent',
            'observations': observations,
            'observation_kwargs': {'lags': lags}}

    return model_kwargs


def load_cached_model(fit_func):
    """
    Decorator for model fitting methods, so that models are not refit
    unnecessarily
    """
    # this is the function we replace original function with
    def wrapper(n_states, *args, **kwargs):

        fit_kwargs = kwargs.get('fit_kwargs', None)
        if fit_kwargs is None:
            fit_kwargs = {
                'save': False, 'load_if_exists': False,
                'expt_id': None, 'model_dir': None, 'save_dir': None}
        model_kwargs = kwargs['model_kwargs']

        train_model = True
        load_model = fit_kwargs['load_if_exists']
        save_model = fit_kwargs['save']

        # check if model exists
        if load_model:
            save_file = get_save_file(n_states, model_kwargs, fit_kwargs)
            if os.path.exists(save_file):
                train_model = False

        # fit model
        if train_model:
            model_results = fit_func(n_states, *args, **kwargs)
        else:
            save_file = get_save_file(n_states, model_kwargs, fit_kwargs)
            print('loading model from %s' % save_file)
            with open(save_file, 'rb') as f:
                model_results = pickle.load(f)

        # save resulting model
        if train_model and save_model:
            save_file = get_save_file(n_states, model_kwargs, fit_kwargs)
            print('saving model to %s' % save_file)
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            with open(save_file, 'wb') as f:
                pickle.dump(model_results, f)

        return model_results

    return wrapper


@load_cached_model
def fit_model(
        n_states, data_dim, input_dim, model_kwargs,
        data_tr, data_val, data_test,
        inputs_tr=None, inputs_val=None, inputs_test=None,
        masks_tr=None, masks_val=None, masks_test=None,
        tags_tr=None, tags_val=None, tags_test=None,
        init_type='kmeans', fit_method='em', opt_kwargs=None,
        save_tr_states=False, fit_kwargs=None):

    model = HMM(K=n_states, D=data_dim, M=input_dim, **model_kwargs)
    model.fit_kwargs = fit_kwargs
    model.model_kwargs = model_kwargs
    model.initialize(data_tr, inputs=inputs_tr, masks=masks_tr, tags=tags_tr)
    init_model(init_type, model, data_tr, inputs_tr, masks_tr, tags_tr)

    if fit_method == 'em':

        if opt_kwargs is None:
            opt_kwargs = {
                'num_iters': 150,
                'tolerance': 1e-2,
                'transition_mstep_kwargs': {'optimizer': 'lbfgs', 'tol': 1e-3}}

        lps = model.fit(
            data_tr, inputs=inputs_tr, masks=masks_tr, tags=tags_tr, method='em',
            initialize=False, **opt_kwargs)

    elif fit_method == 'stochastic-em':

        n_trials = len(data_tr)

        if inputs_tr is None:
            M = (model.M,) if isinstance(model.M, int) else model.M
            inputs_tr = [np.zeros((data.shape[0],) + M) for data in data_tr]
        if masks_tr is None:
            masks_tr = [np.ones_like(data, dtype=bool) for data in data_tr]
        if tags_tr is None:
            tags_tr = [None] * n_trials

        if opt_kwargs is None:
            opt_kwargs = {
                'num_epochs': 150,  # total passes through data
                'num_steps': 1  # number of optimization steps per data chunk
            }

        lps = model.fit(
            data_tr, inputs=inputs_tr, masks=masks_tr, tags=tags_tr, method='stochastic_em',
            initialize=False, **opt_kwargs)

    else:
        raise NotImplementedError('"%s is not a valid fit method' % fit_method)

    # compute stats
    ll_val = model.log_likelihood(data_val, inputs=inputs_val, masks=masks_val, tags=tags_val)
    ll_test = model.log_likelihood(data_test, inputs=inputs_test, masks=masks_test, tags=tags_test)

    # sort states by usage
    inputs_tr = [None] * len(data_tr) if inputs_tr is None else inputs_tr
    states_tr = [model.most_likely_states(x, u) for x, u in zip(data_tr, inputs_tr)]
    usage = np.bincount(np.concatenate(states_tr), minlength=n_states)
    model.permute(np.argsort(-usage))
    if save_tr_states:
        states_tr = [model.most_likely_states(x, u) for x, u in zip(data_tr, inputs_tr)]
    else:
        states_tr = []

    # combine results
    model_results = {
        'model': model,
        'states_tr': states_tr,
        'lps': lps,
        'll_val': ll_val,
        'll_test': ll_test}

    return model_results


def init_model(init_type, model, datas, inputs=None, masks=None, tags=None):
    """Initialize ARHMM model according to one of several schemes.

    The different schemes correspond to different ways of assigning discrete states to the data
    points; once these states have been assigned, linear regression is used to estimate the model
    parameters (dynamics matrices, biases, covariance matrices)

    * init_type = random: states are randomly and uniformly assigned
    * init_type = kmeans: perform kmeans clustering on data; note that this is not a great scheme
        for arhmms on the fly data, because the fly is often standing still in many different
        poses. These poses will be assigned to different clusters, thus breaking the "still" state
        into many initial states
    * init_type = pca_me: first compute the motion energy of the data (square of differences of
        consecutive time points) and then perform PCA. A threshold applied to the first dimension
        does a reasonable job of separating the data into "moving" and "still" timepoints. All
        "still" timepoints are assigned one state, and the remaining timepoints are clustered using
        kmeans with (K-1) clusters
    * init_type = arhmm: refinement of pca_me approach: perform pca on the data and take top 4
        components (to speed up computation) and fit a 2-state arhmm to roughly split the data into
        "still" and "moving" states (this is itself initialized with pca_me). Then as before the
        moving state is clustered into K-1 states using kmeans.

    Args:
        init_type (str):
            'random' | 'kmeans' | 'pca_me' | 'arhmm'
        model (ssm.HMM object):
        datas (list of np.ndarrays):
        inputs (list of np.ndarrays):
        masks (list of np.ndarrays):
        tags (list of np.ndarrays):

    """

    from ssm.regression import fit_linear_regression
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from scipy.signal import savgol_filter

    Ts = [data.shape[0] for data in datas]
    K = model.K
    D = model.observations.D
    lags = model.observations.lags

    # --------------------------
    # initialize discrete states
    # --------------------------
    if init_type == 'random':

        zs = [np.random.choice(K, size=T - lags) for T in Ts]

    elif init_type == 'kmeans':

        km = KMeans(K)
        km.fit(np.vstack(datas))
        zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
        zs = [z[lags:] for z in zs]

    elif init_type == 'arhmm':

        D_ = 4
        pca = PCA(D_)
        xs = pca.fit_transform(np.vstack(datas))
        xs = np.split(xs, np.cumsum(Ts)[:-1])

        model_init = HMM(
            K=2, D=D_, M=0, transitions='standard', observations='ar',
            observations_kwargs={'lags': 1})
        init_model('pca-me', model_init, xs)
        model_init.fit(
            xs, inputs=None, method='em', num_iters=100, tolerance=1e-2,
            initialize=False, transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

        # make still state 0th state
        mses = [np.mean(np.square(model_init.observations.As[i] - np.eye(D_))) for i in range(2)]
        if mses[1] < mses[0]:
            # permute states
            model_init.permute([1, 0])
        moving_state = 1

        inputs_tr = [None] * len(datas)
        zs = [model_init.most_likely_states(x, u) for x, u in zip(xs, inputs_tr)]
        zs = np.concatenate(zs, axis=0)

        # cluster moving data
        km = KMeans(K - 1)
        km.fit(np.vstack(datas)[zs == moving_state])
        zs[zs == moving_state] = km.labels_ + 1

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])
        zs = [z[lags:] for z in zs]  # remove the ends

    elif init_type == 'em':

        import copy

        # load already-trained model that used em
        model_kwargs = copy.deepcopy(model.model_kwargs)
        fit_kwargs = copy.deepcopy(model.fit_kwargs)
        # update model dir (remove new init str, add 2-state-init)
        old_model_dir = fit_kwargs['model_dir']
        fit_kwargs['model_dir'] = '_'.join(old_model_dir.split('_')[:-1]) + '_arhmm-init'

        model_file = get_save_file(K, model_kwargs, fit_kwargs)

        print('initializing model from %s' % model_file)
        with open(model_file, 'rb') as f:
            model_init = pickle.load(f)['model']

        # extract states
        if inputs is None:
            zs = [model_init.most_likely_states(x) for x in datas]
        else:
            zs = [model_init.most_likely_states(x, u) for x, u in zip(datas, inputs)]

        zs = [z[lags:] for z in zs]  # remove the ends

    elif init_type == 'em-exact':

        import copy

        # load already-trained model that used em
        model_kwargs = copy.deepcopy(model.model_kwargs)
        fit_kwargs = copy.deepcopy(model.fit_kwargs)
        # update model dir (remove new init str, add 2-state-init)
        old_model_dir = fit_kwargs['model_dir']
        fit_kwargs['model_dir'] = '_'.join(old_model_dir.split('_')[:-1]) + '_arhmm-init'

        model_file = get_save_file(K, model_kwargs, fit_kwargs)

        print('initializing model from %s' % model_file)
        with open(model_file, 'rb') as f:
            model_init = pickle.load(f)['model']

        model.init_state_distn.log_pi0 = model_init.init_state_distn.log_pi0
        model.transitions.log_Ps = model_init.transitions.log_Ps
        model.observations.As = model_init.observations.As
        model.observations.Vs = model_init.observations.Vs
        model.observations.bs = model_init.observations.bs
        model.observations.Sigmas = model_init.observations.Sigmas

    elif init_type == 'pca-me':

        # pca on motion energy
        datas_filt = np.copy(datas)
        for dtmp in datas_filt:
            for i in range(dtmp.shape[1]):
                dtmp[:, i] = savgol_filter(dtmp[:, i], 5, 2)
        pca = PCA(1)
        me = np.square(np.diff(np.vstack(datas_filt), axis=0))
        xs = pca.fit_transform(np.concatenate([np.zeros((1, D)), me], axis=0))[:, 0]
        xs = xs / np.max(xs)

        # threshold data to get moving/non-moving
        thresh = 0.01
        zs = np.copy(xs)
        zs[xs < thresh] = 0
        zs[xs >= thresh] = 1

        # cluster moving data
        km = KMeans(K - 1)
        km.fit(np.vstack(datas)[zs == 1])
        zs[zs == 1] = km.labels_ + 1

        # split
        zs = np.split(zs, np.cumsum(Ts)[:-1])
        zs = [z[lags:] for z in zs]  # remove the ends

    else:
        raise NotImplementedError('Invalid "init_type" of "%s"' % init_type)

    # ------------------------
    # estimate dynamics params
    # ------------------------
    if init_type != 'em-exact':

        # Initialize the weights with linear regression
        Sigmas = []
        for k in range(K):
            ts = [np.where(z == k)[0] for z in zs]
            Xs = [np.column_stack([data[t + lags - l - 1] for l in range(lags)])
                  for t, data in zip(ts, datas)]
            ys = [data[t + lags] for t, data in zip(ts, datas)]

            # Solve the linear regression
            coef_, intercept_, Sigma = fit_linear_regression(Xs, ys)
            model.observations.As[k] = coef_[:, :D * lags]
            model.observations.Vs[k] = coef_[:, D * lags:]
            model.observations.bs[k] = intercept_
            Sigmas.append(Sigma)

        # Set the variances all at once to use the setter
        model.observations.Sigmas = np.array(Sigmas)


# -------------------------------------------------------------------------------------------------
# model evaluation functions
# -------------------------------------------------------------------------------------------------


def extract_state_runs(states, indxs, min_length=20):
    """
    Find contiguous chunks of data with the same state

    Args:
        states (list):
        indxs (list):
        min_length (int):

    Returns:
        list
    """

    K = len(np.unique(np.concatenate([np.unique(s) for s in states])))
    state_snippets = [[] for _ in range(K)]

    for curr_states, curr_indxs in zip(states, indxs):
        i_beg = 0
        curr_state = curr_states[i_beg]
        curr_len = 1
        for i in range(1, len(curr_states)):
            next_state = curr_states[i]
            if next_state != curr_state:
                # record indices if state duration long enough
                if curr_len >= min_length:
                    state_snippets[curr_state].append(
                        curr_indxs[i_beg:i])
                i_beg = i
                curr_state = next_state
                curr_len = 1
            else:
                curr_len += 1
        # end of trial cleanup
        if next_state == curr_state:
            # record indices if state duration long enough
            if curr_len >= min_length:
                state_snippets[curr_state].append(curr_indxs[i_beg:i])
    return state_snippets


def viterbi_ll(model, datas):
    """Calculate log-likelihood of viterbi path."""
    inputs = [None] * len(datas)
    masks = [None] * len(datas)
    tags = [None] * len(datas)
    states = [model.most_likely_states(x, u) for x, u in zip(datas, inputs)]
    ll = 0
    for data, input, mask, tag, state in zip(datas, inputs, masks, tags, states):
        if input is None:
            input = np.zeros_like(data)
        if mask is None:
            mask = np.ones_like(data, dtype=bool)
        likelihoods = model.observations.log_likelihoods(data, input, mask, tag)
        ll += np.sum(likelihoods[(np.arange(state.shape[0]), state)])
    return ll


def k_step_ll(model, datas, k_max):
    """Determine the k-step ahead ll."""

    M = (model.M,) if isinstance(model.M, int) else model.M
    L = model.observations.lags  # AR lags

    k_step_lls = 0
    for data in datas:
        input = np.zeros((data.shape[0],) + M)
        mask = np.ones_like(data, dtype=bool)
        pi0 = model.init_state_distn.initial_state_distn
        Ps = model.transitions.transition_matrices(data, input, mask, tag=None)
        lls = model.observations.log_likelihoods(data, input, mask, tag=None)

        T, K = lls.shape

        # Forward pass gets the predicted state at time t given
        # observations up to and including those from time t
        alphas = np.zeros((T, K))
        forward_pass(pi0, Ps, lls, alphas)

        # pz_tt = p(z_{t},  x_{1:t}) = alpha(z_t) / p(x_{1:t})
        pz_tt = np.exp(alphas - logsumexp(alphas, axis=1, keepdims=True))
        log_likes_list = []
        for k in range(k_max + 1):
            if k == 0:
                # p(x_t | x_{1:T}) = \sum_{z_t} p(x_t | z_t) p(z_t | x_{1:t})
                pz_tpkt = np.copy(pz_tt)
                assert np.allclose(np.sum(pz_tpkt, axis=1), 1.0)
                log_likes_0 = logsumexp(lls[k_max:] + np.log(pz_tpkt[k_max:]), axis=1)
            #                 pred_data = get_predicted_obs(model, data, pz_tpkt)
            else:
                if k == 1:
                    # p(z_{t+1} | x_{1:t}) =
                    # \sum_{z_t} p(z_{t+1} | z_t) alpha(z_t) / p(x_{1:t})
                    pz_tpkt = np.copy(pz_tt)

                # p(z_{t+k} | x_{1:t}) =
                # \sum_{z_{t+k-1}} p(z_{t+k} | z_{t+k-1}) p(z_{z+k-1} | x_{1:t})
                if Ps.shape[0] == 1:  # stationary transition matrix
                    pz_tpkt = np.matmul(pz_tpkt[:-1, None, :], Ps)[:, 0, :]
                else:  # dynamic transition matrix
                    pz_tpkt = np.matmul(pz_tpkt[:-1, None, :], Ps[k - 1:])[:, 0, :]
                assert np.allclose(np.sum(pz_tpkt, axis=1), 1.0)

                # p(x_{t+k} | x_{1:t}) =
                # \sum_{z_{t+k}} p(x_{t+k} | z_{t+k}) p(z_{t+k} | x_{1:t})
                log_likes = logsumexp(lls[k:] + np.log(pz_tpkt), axis=1)
                # compute summed ll only over timepoints that are valid for each value of k
                log_likes_0 = log_likes[k_max - k:]

            log_likes_list.append(np.sum(log_likes_0))

    k_step_lls += np.array(log_likes_list)

    return k_step_lls


def k_step_r2(model, datas, k_max, n_samp=10, with_noise=True, return_type='per_batch_r2'):
    """Determine the k-step ahead r2.

    Args:
        model:
        datas:
        k_max:
        n_samp:
        with_noise:
        return_type:
            'per_batch_r2'
            'total_r2'
            'bootstrap_r2'
            'per_batch_mse'

    Returns:

    """

    N = len(datas)
    L = model.observations.lags  # AR lags
    D = model.D

    x_true_total = []
    x_pred_total = [[] for _ in range(k_max)]
    if return_type == 'per_batch_r2':
        k_step_r2s = np.zeros((N, k_max, n_samp))
    elif return_type == 'total_r2':
        k_step_r2s = np.zeros((k_max, n_samp))
    else:
        raise NotImplementedError('"%s" is not a valid return type' % return_type)

    for d, data in enumerate(datas):
        # print('%i/%i' % (d + 1, len(datas)))

        T = data.shape[0]

        x_true_all = data[L + k_max - 1: T + 1]
        x_pred_all = np.zeros((n_samp, (T - 1), D, k_max))

        # zs = model.most_likely_states(data)

        # collect sampled data
        for t in range(L - 1, T):
            # find the most likely discrete state at time t based on its past
            zs = model.most_likely_states(data[:t + 1])[-L:]
            # sample forward in time n_samp times
            for n in range(n_samp):
                # sample forward in time k_max steps
                _, x_pred = model.sample(
                    k_max, prefix=(zs, data[t - L + 1:t + 1]), with_noise=with_noise)
                # _, x_pred = model.sample(
                #     k_max, prefix=(zs[t-L+1:t+1], data[t-L+1:t+1]), with_noise=False)
                # predicted x values in the forward prediction time
                x_pred_all[n, t - L + 1, :, :] = np.transpose(x_pred)[None, None, :, :]

        # store predicted data
        x_true_total.append(x_true_all)
        for k in range(k_max):
            idxs = (k_max - k - 1, k_max - k - 1 + x_true_all.shape[0])
            x_pred_total[k].append(x_pred_all[:, slice(*idxs), :, k])

    # compute r2s
    if return_type == 'per_batch_r2':
        for d in range(len(datas)):
            for k in range(k_max):
                for n in range(n_samp):
                    k_step_r2s[d, k, n] = r2_score(
                        x_true_total[d], x_pred_total[k][d][n])

    elif return_type == 'total_r2':
        for k in range(k_max):
            for n in range(n_samp):
                k_step_r2s[k, n] = r2_score(
                    np.vstack(x_true_total),
                    np.vstack([x_pred_total[k][d][n] for d in range(len(datas))]))

    return k_step_r2s


def test_k_step_r2():

    class TestObs(object):
        def __init__(self, lags):
            self.lags = lags

    class TestModel(object):
        def __init__(self, D, L):
            self.D = D
            self.observations = TestObs(L)

        def most_likely_states(self, *args):
            return np.arange(self.observations.lags)

        def sample(self, k, prefix=None, with_noise=False):
            _, p = prefix
            data = p[-1]
            assert len(data) == self.D
            return None, data + 0.99 * np.arange(1, k + 1)[:, None]

    data = np.column_stack([np.arange(100), np.arange(1, 101)])
    T, D = data.shape
    k_max = 5
    n_samp = 1
    L = 2

    model = TestModel(D, L)

    r2s = k_step_r2(model, [data], k_max, n_samp=n_samp, with_noise=False)

    assert np.allclose(r2s, 1)
    for k in range(k_max - 1):
        assert r2s[0, k, 0] > r2s[0, k + 1, 0]

    # self contained example
    # data = np.column_stack([np.arange(100), np.arange(1, 101)])
    # T, D = data.shape
    # k_max = 5
    # n_samp = 1
    # L = 2
    #
    # x_true_all = data[L + k_max - 1: T + 1]
    # x_pred_all = np.zeros((n_samp, (T - 1), D, k_max))
    # for t in range(L - 1, T):
    #     for n in range(n_samp):
    #         x_pred = data[t] + 0.99 * np.arange(1, k_max + 1)[:, None]
    #         x_pred_all[n, t - L + 1, :, :] = np.transpose(x_pred)[None, None, :, :]
    # # compute r2
    # # for k in range(k_max):
    # #     idxs = (k_max - k - 1, k_max - k - 1 + x_true_all.shape[0])
    # #     for n in range(n_samp):
    # #         r2 = r2_score(x_true_all, x_pred_all[n, :, :, k][slice(*idxs)])
    # #         print(r2)
    # print('true:')
    # print('beg: {}'.format(x_true_all[0]))
    # print('end: {}'.format(x_true_all[-1]))
    # print('\n')
    # for k in range(k_max):
    #     print(k + 1)
    #     idxs = (k_max - k - 1, k_max - k - 1 + x_true_all.shape[0])
    #     for n in range(n_samp):
    #         test = x_pred_all[n, :, :, k][slice(*idxs)]
    #         print('beg: {}'.format(test[0]))
    #         print('end: {}'.format(test[-1]))


def k_step_r2_old(
        model, datas, k_max, n_samp=10, with_noise=True, return_type='per_batch_r2', n_skip=1):
    """Determine the k-step ahead r2.

    Args:
        model:
        datas:
        k_max:
        n_samp:
        with_noise:
        return_type:
            'per_batch_r2'
            'total_r2'
            'bootstrap_r2'
            'per_batch_mse'

    Returns:

    """

    N = len(datas)
    L = model.observations.lags  # AR lags
    D = model.D

    if return_type == 'per_batch_r2':
        k_step_r2s = np.zeros((N, k_max, n_samp))
    else:
        raise NotImplementedError('"%s" is not a valid return type' % return_type)

    for d, data in enumerate(datas):
        # print('%i/%i' % (d + 1, len(datas)))

        T = data.shape[0]
        ts = range(L - 1, T, n_skip)

        x_true_all = data[L + k_max - 1: T + 1]
        x_pred_all = np.zeros((n_samp, (T - 1), D, k_max))

        # zs = model.most_likely_states(data)

        # collect sampled data
        for t in ts:
            # find the most likely discrete state at time t based on its past
            zs = model.most_likely_states(data[:t + 1])[-L:]
            # sample forward in time n_samp times
            for n in range(n_samp):
                # sample forward in time k_max steps
                _, x_pred = model.sample(
                    k_max, prefix=(zs, data[t - L + 1:t + 1]), with_noise=with_noise)
                # _, x_pred = model.sample(
                #     k_max, prefix=(zs[t-L+1:t+1], data[t-L+1:t+1]), with_noise=False)
                # predicted x values in the forward prediction time
                x_pred_all[n, t - L + 1, :, :] = np.transpose(x_pred)[None, None, :, :]

        # compute r2
        if return_type == 'per_batch_r2':
            for k in range(k_max):
                idxs = (k_max - k - 1, k_max - k - 1 + x_true_all.shape[0])
                for n in range(n_samp):
                    k_step_r2s[d, k, n] = r2_score(
                        x_true_all, x_pred_all[n, :, :, k][slice(*idxs)])

    return k_step_r2s


# -------------------------------------------------------------------------------------------------
# path handling functions
# -------------------------------------------------------------------------------------------------

def get_save_file(n_states, model_kwargs, fit_kwargs):
    from flygenvectors.utils import get_dirs
    model_name = get_model_name(n_states, model_kwargs)
    model_name += '.pkl'
    if fit_kwargs['save_dir'] is not None:
        save_dir = fit_kwargs['save_dir']
    else:
        base_dir = get_dirs()['results']
        model_dir = fit_kwargs['model_dir']
        expt_dir = get_expt_dir(base_dir, fit_kwargs['expt_id'])
        save_dir = os.path.join(base_dir, expt_dir, model_dir)
    return os.path.join(save_dir, model_name)


def get_model_name(n_states, model_kwargs):
    trans = model_kwargs['transitions']
    obs = model_kwargs['observations']
    if obs.find('ar') > -1:
        lags = model_kwargs['observation_kwargs']['lags']
    else:
        lags = 0
    if trans == 'sticky':
        kappa = model_kwargs['transition_kwargs']['kappa']
    else:
        kappa = ''
    model_name = str(
        'obs=%s_trans=%s_lags=%i_K=%02i' % (obs, trans, lags, n_states))
    if trans == 'sticky':
        model_name = str('%s_kappa=%1.0e' % (model_name, kappa))
    return model_name


def get_model_dir(base, preprocess_list, fit_method, init_type, final_str=''):
    model_dir = base
    if 'filter' in preprocess_list.keys():
        if preprocess_list['filter']['type'] == 'median':
            model_dir += '_median-%i' % preprocess_list['filter']['window_size']
        elif preprocess_list['filter']['type'] == 'savgol':
            model_dir += '_savgol-%i-%i' % (
                preprocess_list['filter']['window_size'], preprocess_list['filter']['order'])
        else:
            raise NotImplementedError

    if final_str:
        model_dir += '_' + final_str  # here for legacy reasons

    model_dir += '_%s' % fit_method
    model_dir += '_%s-init' % init_type

    return model_dir


def get_expt_dir(base_dir, expt_ids):
    if isinstance(expt_ids, list) and len(expt_ids) > 1:
        # multisession; see if multisession already exists; if not, create a new one
        subdirs = get_subdirs(base_dir)
        expt_dir = None
        max_val = -1
        for subdir in subdirs:
            if subdir[:5] == 'multi':
                # load csv containing expt_ids
                multi_sess = read_session_info_from_csv(
                    os.path.join(base_dir, subdir, 'session_info.csv'))
                # compare to current ids
                multi_sess = [row['session'] for row in multi_sess]
                if sorted(multi_sess) == sorted(expt_ids):
                    expt_dir = subdir
                    break
                else:
                    max_val = np.max([max_val, int(subdir.split('-')[-1])])
        if expt_dir is None:
            expt_dir = 'multi-' + str(max_val + 1)
            # save csv with expt ids
            export_session_info_to_csv(
                os.path.join(base_dir, expt_dir),
                [{'session': s} for s in expt_ids])
    else:
        if isinstance(expt_ids, list):
            expt_dir = expt_ids[0]
        else:
            expt_dir = expt_ids
    return expt_dir


def read_session_info_from_csv(session_file):
    """Read csv file that contains session info.

    Parameters
    ----------
    session_file : :obj:`str`
        /full/path/to/session_info.csv

    Returns
    -------
    :obj:`list` of :obj:`dict`
        dict for each session which contains lab/expt/animal/session

    """
    import csv
    sessions_multi = []
    # load and parse csv file that contains single session info
    with open(session_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sessions_multi.append(dict(row))
    return sessions_multi


def export_session_info_to_csv(session_dir, ids_list):
    """Export list of sessions to csv file.

    Parameters
    ----------
    session_dir : :obj:`str`
        absolute path for where to save :obj:`session_info.csv` file
    ids_list : :obj:`list` of :obj:`dict`
        dict for each session which contains session

    """
    import csv
    session_file = os.path.join(session_dir, 'session_info.csv')
    if not os.path.isdir(session_dir):
        os.makedirs(session_dir)
    with open(session_file, mode='w') as f:
        session_writer = csv.DictWriter(f, fieldnames=list(ids_list[0].keys()))
        session_writer.writeheader()
        for ids in ids_list:
            session_writer.writerow(ids)
