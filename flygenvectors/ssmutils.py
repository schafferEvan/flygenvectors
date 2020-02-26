import os
import numpy as np
import pickle
from ssm import HMM


def load_cached_model(fit_func):
    """
    Decorator for model fitting methods, so that models are not refit
    unnecessarily
    """
    # this is the function we replace original function with
    def wrapper(n_states, *args, **kwargs):

        fit_kwargs = kwargs.pop('fit_kwargs', None)
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


def split_trials(
        n_trials, rng_seed=0, trials_tr=5, trials_val=1, trials_test=1,
        trials_gap=1):
    """
    Split trials into train/val/test blocks.

    The data is split into blocks that have gap trials between tr/val/test:
    train tr | gap tr | val tr | gap tr | test tr | gap tr

    Args:
        n_trials (int): number of trials to use in the split
        rng_seed (int): numpy random seed for reproducibility
        trials_tr (int): train trials per block
        trials_val (int): validation trials per block
        trials_test (int): test trials per block
        trials_gap (int): gap trials between tr/val/test; there will be a total
            of 3 * `trials_gap` gap trials per block

    Returns:
        (dict)
    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = trials_tr + trials_gap + trials_val + trials_gap + trials_test + trials_gap

    n_blocks = int(np.floor(n_trials / tr_per_block))
    leftover_trials = n_trials - tr_per_block * n_blocks
    if leftover_trials > 0:
        offset = np.random.randint(0, high=leftover_trials)
    else:
        offset = 0
    indxs_block = np.random.permutation(n_blocks)

    batch_indxs = {'train': [], 'test': [], 'val': []}
    for block in indxs_block:
        curr_tr = block * tr_per_block + offset
        batch_indxs['train'].append(np.arange(curr_tr, curr_tr + trials_tr))
        curr_tr += (trials_tr + trials_gap)
        batch_indxs['val'].append(np.arange(curr_tr, curr_tr + trials_val))
        curr_tr += (trials_val + trials_gap)
        batch_indxs['test'].append(np.arange(curr_tr, curr_tr + trials_test))

    for dtype in ['train', 'val', 'test']:
        batch_indxs[dtype] = np.concatenate(batch_indxs[dtype], axis=0)

    return batch_indxs


@load_cached_model
def fit_model(
        n_states, data_dim, input_dim, model_kwargs,
        data_tr, data_val, data_test,
        inputs_tr=None, inputs_val=None, inputs_test=None,
        init_type='kmeans', save_tr_states=False, fit_kwargs=None):

    model = HMM(K=n_states, D=data_dim, M=input_dim, **model_kwargs)
    model.initialize(data_tr, inputs=inputs_tr)
    init_model(init_type, model, data_tr)

    # run EM; specify tolerances for overall convergence and each M-step's convergence
    lps = model.fit(
        data_tr, inputs=inputs_tr, method='em', num_iters=150, tolerance=1e-2, initialize=False,
        transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

    # compute stats
    ll_val = model.log_likelihood(data_val, inputs=inputs_val)
    ll_test = model.log_likelihood(data_test, inputs=inputs_test)

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


def init_model(init_type, model, datas):
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
        init_model('pca_me', model_init, xs)
        model_init.fit(
            xs, inputs=None, method='em', num_iters=100, tolerance=1e-2,
            initialize=False, transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

        # make still state 0th state
        mses = [
            np.mean(np.square(model_init.observations.As[i] - np.eye(D_))) for i in range(2)]
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

    elif init_type == 'pca_me':

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
    # Initialize the weights with linear regression
    Sigmas = []
    for k in range(K):
        ts = [np.where(z == k)[0] for z in zs]
        Xs = [np.column_stack([data[t + l] for l in range(lags)])
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


def get_save_file(n_states, model_kwargs, fit_kwargs):
    from flygenvectors.utils import get_dirs
    model_name = get_model_name(n_states, model_kwargs)
    model_name += '.pkl'
    if fit_kwargs['save_dir'] is not None:
        save_dir = fit_kwargs['save_dir']
    else:
        base_dir = get_dirs()['results']
        model_dir = fit_kwargs['model_dir']
        expt_dir = fit_kwargs['expt_id']
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


def split_runs(indxs, dtypes, dtype_lens):
    """

    Args:
        indxs (list):
        dtypes (list of strs):
        dtype_lens (list of ints):

    Returns:
        dict
    """

    # first sort, then split according to ratio
    i_sorted = np.argsort([len(i) for i in indxs])

    indxs_split = {dtype: [] for dtype in dtypes}
    dtype_indx = 0
    dtype_curr = dtypes[dtype_indx]
    counter = 0
    for indx in reversed(i_sorted):
        if counter == dtype_lens[dtype_indx]:
            # move to next dtype
            dtype_indx = (dtype_indx + 1) % len(dtypes)
            while dtype_lens[dtype_indx] == 0:
                dtype_indx = (dtype_indx + 1) % len(dtypes)
            dtype_curr = dtypes[dtype_indx]
            counter = 0
        indxs_split[dtype_curr].append(indxs[indx])
        counter += 1

    return indxs_split


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
