import numpy as np
from sklearn.decomposition import PCA
from ssm import HMM
from ssm.util import find_permutation


def heuristic_segmentation_v1(labels):

    # define behaviors
    D = labels.shape[1]
    n_markers = D / 2
    idxs = {
        'body': np.array([0, 1, 0 + n_markers, 1 + n_markers]).astype('int'),
        'back': np.array([2, 3, 2 + n_markers, 3 + n_markers]).astype('int'),
        'mid': np.array([4, 5, 4 + n_markers, 5 + n_markers]).astype('int'),
        'front': np.array([6, 7, 6 + n_markers, 7 + n_markers]).astype('int')}
    idxs['legs'] = np.concatenate([idxs['back'], idxs['mid'], idxs['front']])

    # still - threshold data to get moving/non-moving
    pca = PCA(1)
    me = np.concatenate([np.zeros((1, D)), np.square(np.diff(labels, axis=0))], axis=0)
    xs_me = pca.fit_transform(me)[:, 0]
    xs_me = xs_me / np.max(xs_me)
    thresh = 0.01
    zs_still = np.zeros_like(xs_me)
    zs_still[xs_me < thresh] = 1

    # running - look at all leg indices
    xs_legs = pca.fit_transform(me[:, idxs['legs']])[:, 0]
    xs_legs = xs_legs / np.max(xs_legs)
    thresh = 0.10
    zs_run = np.zeros_like(xs_legs)
    zs_run[xs_legs >= thresh] = 1

    # front groom
    xs_fg = pca.fit_transform(me[:, idxs['front']])[:, 0]
    xs_fg = xs_fg / np.max(xs_fg)
    thresh = 0.01
    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= thresh) & ~zs_run.astype('bool')] = 1

    # back groom
    xs_bg = pca.fit_transform(me[:, idxs['back']])[:, 0]
    xs_bg = xs_bg / np.max(xs_bg)
    thresh = 0.01
    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= thresh) & ~zs_run.astype('bool')] = 1

    # collect states
    states = 4 * np.ones_like(zs_still, dtype='int')  # default state = 4: undefined
    states[zs_still == 1] = 0
    states[zs_run == 1] = 1
    states[zs_fg == 1] = 2
    states[zs_bg == 1] = 3
    state_mapping = {
        0: 'still',
        1: 'run',
        2: 'front_groom',
        3: 'back_groom',
        4: 'undefined'}

    # smooth states with categorical HMM
    model = HMM(K=5, D=1, observations='categorical', observation_kwargs=dict(C=5))
    lps = model.fit(states[:, None], num_iters=150)
    states_ = model.most_likely_states(states[:, None])
    model.permute(find_permutation(states, states_))
    states_new = model.most_likely_states(states[:, None])

    return states_new, state_mapping


def heuristic_segmentation_v2(labels, ball_me, walk_thresh=0.2):

    from skimage.restoration import denoise_tv_chambolle

    D = labels.shape[1]
    n_markers = int(D / 2)

    idxs = {
        'body': np.array([0, 1, 0 + n_markers, 1 + n_markers]).astype('int'),
        'back': np.array([2, 3, 2 + n_markers, 3 + n_markers]).astype('int'),
        'mid': np.array([4, 5, 4 + n_markers, 5 + n_markers]).astype('int'),
        'front': np.array([6, 7, 6 + n_markers, 7 + n_markers]).astype('int')}
    idxs['legs'] = np.concatenate([idxs['back'], idxs['mid'], idxs['front']])

    # running - look at ball me
    zs_run = np.zeros_like(ball_me)
    ball_me2 = ball_me - np.percentile(ball_me, 1)
    zs_run[ball_me2 >= walk_thresh] = 1

    # still - threshold data to get moving/non-moving
    me = np.concatenate([np.zeros((1, D)), np.square(np.diff(labels, axis=0))], axis=0)

    xs_me = np.mean(me, axis=1)
    xs_me = denoise_tv_chambolle(xs_me, weight=0.05)
    xs_me -= np.min(xs_me)
    xs_me /= np.percentile(xs_me, 99)
    thresh = 0.02
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < thresh) & ~zs_run.astype('bool')] = 1

    # front groom (just look at x movements)
    # xs_fg = pca.fit_transform(me[:, idxs['front']])[:, 0]
    # xs_fg = me[:, idxs['front'][0]]
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)
    thresh = 0.02
    # xs_fg2 = me[:, idxs['front'][1]]
    # xs_fg2 = xs_fg2 / np.max(xs_fg2)
    # zs_fg[(xs_fg >= thresh) & ~zs_run.astype('bool') & (xs_fg2 < 0.1)] = 1

    # back groom
    # xs_bg = pca.fit_transform(me[:, idxs['back']])[:, 0]
    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)
    thresh = 0.02

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')
    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= thresh) & (xs_fg < thresh) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= thresh) & (xs_bg < thresh) & not_run_or_still] = 1

    # collect states
    states = 4 * np.ones_like(zs_still, dtype='int')  # default state = 4: undefined
    states[zs_still==1] = 0
    states[zs_run==1] = 1
    states[zs_fg==1] = 2
    states[zs_bg==1] = 3
    state_mapping = {
        0: 'still',
        1: 'walk',
        2: 'front_groom',
        3: 'back_groom',
        4: 'undefined'}

    # smooth states with categorical HMM
    models = []
    lps = []
    for i in range(5):
        models.append(HMM(K=5, D=1, observations='categorical', observation_kwargs=dict(C=5)))
        lps.append(models[i].fit(states[:, None], num_iters=150, tolerance=1e-2))
    best_idx = np.argmax([lp[-1] for lp in lps])
    model = models[best_idx]

    states_ = model.most_likely_states(states[:, None])
    model.permute(find_permutation(states, states_))
    states_new = model.most_likely_states(states[:, None])

    return states_new, state_mapping
