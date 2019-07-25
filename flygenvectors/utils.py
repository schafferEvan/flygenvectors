import os
import numpy as np
from ssm import HMM


def get_user():
    """Get name of user running this function"""
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def get_dirs():
    username = get_user()
    if username == 'evan':
        dirs = {
            'data': '',
            'results': ''
        }
    elif username == 'mattw':
        dirs = {
            'data': '/home/mattw/data/schaffer/',  # base data dir
            'results': '/home/mattw/results/fly/'  # base results dir
        }
    else:
        raise ValueError(
            'must update flygenvectors.utils.get_dirs() to include user %s' %
            username)
    return dirs


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


def fit_model(
        n_states, data_dim, input_dim, model_kwargs,
        data_tr, data_val, data_test,
        inputs_tr=None, inputs_val=None, inputs_test=None):

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
    states_tr = [model.most_likely_states(x, u) for x, u in
                 zip(data_tr, inputs_tr)]

    # combine results
    return {
        'model': model,
        'states_tr': states_tr,
        'lps': lps,
        'll_val': ll_val,
        'll_test': ll_test}
