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

    tr_per_block = \
        trials_tr + trials_gap + trials_val + trials_gap + trials_test + trials_gap

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
        save_tr_states=False, fit_kwargs=None):

    model = HMM(K=n_states, D=data_dim, M=input_dim, **model_kwargs)
    model.initialize(data_tr, inputs=inputs_tr)
    model.observations.initialize(data_tr, inputs=inputs_tr)

    # run EM; specify tolerances for overall convergence and each M-step's
    # convergence
    lps = model.fit(
        data_tr, inputs=inputs_tr, method='em', num_em_iters=50,
        tolerance=1e-1,
        transitions_mstep_kwargs={'optimizer': 'lbfgs', 'tol': 1e-3})

    # compute stats
    ll_val = model.log_likelihood(data_val, inputs=inputs_val)
    ll_test = model.log_likelihood(data_test, inputs=inputs_test)

    # sort states by usage
    inputs_tr = [None] * len(data_tr) if inputs_tr is None else inputs_tr
    states_tr = [model.most_likely_states(x, u) for x, u in
                 zip(data_tr, inputs_tr)]
    usage = np.bincount(np.concatenate(states_tr), minlength=n_states)
    model.permute(np.argsort(-usage))
    if save_tr_states:
        states_tr = [
            model.most_likely_states(x, u) for x, u in zip(data_tr, inputs_tr)]
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
    if obs == 'ar':
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


def extract_high_likelihood_runs(
        likelihoods, l_thresh=0.8, min_length=100, max_length=500,
        comparison='>=', dims='all', skip_indxs=None):
    """
    Find contiguous chunks of data with likelihoods larger than a given value

    TODO: remove; now in dlc.DLCLabels class

    Args:
        likelihoods (np array):
        l_thresh (float): minimum likelihood threshold
        min_length (int): minimum length of high likelihood runs
        max_length (int): maximum length of high likelihood runs; once a run
            surpasses this threshold a new run is started
        comparison (str): comparison operator to use for data ? l_thresh
            '>' | '>=' | '<' | '<='
        dims (str): define whether any or all dims must meet requirement
            'any' | 'all'
        skip_indxs (np bool array or NoneType, optional): same size as
            `likelihoods`, `True` indices will be counted as a negative
            comparison

    Returns:
        list: of run indices
    """

    import operator
    if comparison == '>':
        op = operator.gt
    elif comparison == '>=':
        op = operator.ge
    elif comparison == '<':
        op = operator.lt
    elif comparison == '<=':
        op = operator.le
    else:
        raise ValueError('"%s" is an invalid comparison operator' % comparison)

    if dims == 'any':
        bool_check = np.any
    elif dims == 'all':
        bool_check = np.all
    else:
        raise ValueError('"%s" is an invalid boolean check' % dims)

    T = likelihoods.shape[0]
    if skip_indxs is None:
        skip_indxs = np.full(shape=(T,), fill_value=False)

    indxs = []

    run_len = 1
    i_beg = 0
    i_end = 1

    reset_run = False
    save_run = False
    for t in range(1, T):

        if bool_check(op(likelihoods[t], l_thresh)) and not skip_indxs[t]:
            run_len += 1
            i_end += 1
        else:
            if run_len >= min_length:
                save_run = True
            reset_run = True
        if run_len == max_length:
            save_run = True
            reset_run = True

        if save_run:
            indxs.append(np.arange(i_beg, i_end))
            save_run = False
        if reset_run:
            run_len = 1
            i_beg = t
            i_end = t + 1
            reset_run = False

    # final run
    if run_len - 1 >= min_length:
        indxs.append(np.arange(i_beg, i_end - 1))

    return indxs


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
