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


def heuristic_segmentation_v2(
        labels, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02):

    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # misc:

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
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    # front groom (just look at x movements)
    # xs_fg = pca.fit_transform(me[:, idxs['front']])[:, 0]
    # xs_fg = me[:, idxs['front'][0]]
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    # back groom
    # xs_bg = pca.fit_transform(me[:, idxs['back']])[:, 0]
    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')
    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < groom_thresh) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < groom_thresh) & not_run_or_still] = 1

    # collect states
    states = 4 * np.ones_like(zs_still, dtype='int')  # default state = 4: undefined
    states[zs_still == 1] = 0
    states[zs_run == 1] = 1
    states[zs_fg == 1] = 2
    states[zs_bg == 1] = 3
    state_mapping = {
        0: 'still',
        1: 'walk',
        2: 'front_groom',
        3: 'back_groom',
        4: 'undefined'}

    # smooth states with categorical HMM
    np.random.seed(0)
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


def heuristic_segmentation_v3(
        labels, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02):

    # same as v2, but states are saved in different order to match how deepethogram saves hand
    # labels
    #
    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # misc:

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
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    # front groom (just look at x movements)
    # xs_fg = pca.fit_transform(me[:, idxs['front']])[:, 0]
    # xs_fg = me[:, idxs['front'][0]]
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    # back groom
    # xs_bg = pca.fit_transform(me[:, idxs['back']])[:, 0]
    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')
    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < groom_thresh) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < groom_thresh) & not_run_or_still] = 1

    # collect states
    states = 0 * np.ones_like(zs_still, dtype='int')  # default state = 0: undefined
    states[zs_still == 1] = 1
    states[zs_run == 1] = 2
    states[zs_fg == 1] = 3
    states[zs_bg == 1] = 4
    state_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'walk',
        3: 'front_groom',
        4: 'back_groom'}

    # smooth states with categorical HMM
    np.random.seed(0)
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


def heuristic_segmentation_v4(
        labels, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02, ab_thresh=0.5,
        run_smoother=True, n_restarts=3):

    # same as v3, but with additional abdomen-move state
    #
    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # abdomen-move:
    #   - false positives when fly doesn't actually perform this behavior
    # misc:

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
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')

    # front groom (just look at x movements)
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    # back groom
    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    # ### v4.0 ###
    # # abdomen-move
    # xs_ab = np.mean(me[:, idxs['body']], axis=1)
    # xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)
    # xs_ab_d -= np.min(xs_ab_d)
    # xs_ab_d /= np.percentile(xs_ab_d, 99)
    # zs_ab = np.zeros_like(xs_ab_d)
    # zs_ab[(xs_ab_d >= ab_thresh) & not_run_or_still] = 1
    #
    # zs_bg = np.zeros_like(xs_bg)
    # zs_bg[
    #     (xs_bg >= groom_thresh) &
    #     (xs_fg < 0.02) &
    #     not_run_or_still &
    #     ~zs_ab.astype('bool')] = 1
    #
    # zs_fg = np.zeros_like(xs_fg)
    # zs_fg[
    #     (xs_fg >= groom_thresh) &
    #     (xs_bg < 0.02) &
    #     not_run_or_still &
    #     ~zs_ab.astype('bool')] = 1

    # ### v4.0 ###
    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < 0.02) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < 0.02) & not_run_or_still] = 1

    # abdomen-move
    xs_ab = np.mean(me[:, idxs['body']], axis=1)
    xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)
    xs_ab_d -= np.min(xs_ab_d)
    xs_ab_d /= np.percentile(xs_ab_d, 99)
    zs_ab = np.zeros_like(xs_ab_d)
    # zs_ab[(xs_ab_d >= ab_thresh) & ~zs_bg.astype('bool') & ~zs_run.astype('bool')] = 1
    zs_ab[(xs_ab_d >= ab_thresh) & not_run_or_still] = 1

    # collect states
    states = np.zeros_like(zs_still, dtype='int')  # default state = 0: undefined
    states[zs_still == 1] = 1
    states[zs_run == 1] = 2
    states[zs_fg == 1] = 3
    states[zs_bg == 1] = 4
    states[zs_ab == 1] = 5
    state_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'walk',
        3: 'front_groom',
        4: 'back_groom',
        5: 'abdomen_move'}

    K = len(state_mapping)

    # smooth states with categorical HMM
    if run_smoother:
        np.random.seed(0)
        models = []
        lps = []
        for i in range(n_restarts):
            models.append(HMM(K=K, D=1, observations='categorical', observation_kwargs=dict(C=K)))
            lps.append(models[i].fit(states[:, None], num_iters=150, tolerance=1e-2))
        best_idx = np.argmax([lp[-1] for lp in lps])
        model = models[best_idx]

        states_ = model.most_likely_states(states[:, None])
        model.permute(find_permutation(states, states_))
        states_new = model.most_likely_states(states[:, None])
    else:
        states_new = np.copy(states)

    return states_new, state_mapping


def heuristic_segmentation_v5(
        labels, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02, ab_thresh=0.5,
        fidget_thresh=0.1, run_smoother=True, n_restarts=3):

    # same as v4, but with additional fidget state
    #
    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # abdomen-move:
    #   - false positives when fly doesn't actually perform this behavior
    # fidget:
    #   - ?
    # misc:

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
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')

    # front groom (just look at x movements)
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    # back groom
    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < 0.02) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < 0.02) & not_run_or_still] = 1

    # abdomen-move
    xs_ab = np.mean(me[:, idxs['body']], axis=1)
    xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)
    xs_ab_d -= np.min(xs_ab_d)
    xs_ab_d /= np.percentile(xs_ab_d, 99)
    zs_ab = np.zeros_like(xs_ab_d)
    zs_ab[(xs_ab_d >= ab_thresh) & not_run_or_still] = 1

    # fidget
    xs_legs = np.mean(me[:, np.concatenate([idxs['back'], idxs['mid'], idxs['front']])], axis=1)
    xs_legs = denoise_tv_chambolle(xs_legs, weight=0.05)
    xs_legs -= np.min(xs_legs)
    xs_legs /= np.percentile(xs_legs, 99)

    zs_legs = np.zeros_like(xs_legs)
    zs_legs[
        (xs_legs >= fidget_thresh) &
        ~zs_bg.astype('bool') &
        ~zs_fg.astype('bool') &
        ~zs_ab.astype('bool') &
        not_run_or_still] = 1

    # collect states
    states = np.zeros_like(zs_still, dtype='int')  # default state = 0: undefined
    states[zs_still == 1] = 1
    states[zs_run == 1] = 2
    states[zs_fg == 1] = 3
    states[zs_bg == 1] = 4
    states[zs_ab == 1] = 5
    states[zs_legs == 1] = 6
    state_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'walk',
        3: 'front_groom',
        4: 'back_groom',
        5: 'abdomen_move',
        6: 'fidget'}

    K = len(state_mapping)

    # smooth states with categorical HMM
    if run_smoother:
        np.random.seed(0)
        models = []
        lps = []
        for i in range(n_restarts):
            models.append(HMM(K=K, D=1, observations='categorical', observation_kwargs=dict(C=K)))
            lps.append(models[i].fit(states[:, None], num_iters=150, tolerance=1e-2))
        best_idx = np.argmax([lp[-1] for lp in lps])
        model = models[best_idx]

        states_ = model.most_likely_states(states[:, None])
        model.permute(find_permutation(states, states_))
        states_new = model.most_likely_states(states[:, None])
    else:
        states_new = np.copy(states)

    return states_new, state_mapping


def heuristic_segmentation_v6(
        labels, ball_me, walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02, ab_thresh=0.5,
        fidget_thresh=0.1, run_smoother=True, n_restarts=3):

    # same as v5, but with additional constraints on abdomen state
    #
    # failure modes:
    # still:
    #   - slow abdominal/grooming movements counted here; mvmt is slow but consistent over frames
    #   - doesn't work well if fly is almost always moving (06_26_fly1)
    # moving:
    #   - when using ball, will also pick up periods when ball is being pushed away (struggle)
    # front groom:
    #   - often finds walking where mid/back legs are not in motion, but front are (10_14)fly3)
    # back groom:
    #   - sometimes mid legs do back grooming, counted as moving/misc (08_08_fly1)
    # abdomen-move:
    #   - false positives when fly doesn't actually perform this behavior
    # fidget:
    #   - ?
    # misc:

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
    zs_still = np.zeros_like(xs_me)
    zs_still[(xs_me < still_thresh) & ~zs_run.astype('bool')] = 1

    not_run_or_still = ~zs_run.astype('bool') & ~zs_still.astype('bool')

    # front/back groom
    xs_fg = np.mean(me[:, idxs['front']], axis=1)
    xs_fg = denoise_tv_chambolle(xs_fg, weight=0.05)
    xs_fg -= np.min(xs_fg)
    xs_fg /= np.percentile(xs_fg, 99)

    xs_bg = np.mean(me[:, idxs['back']], axis=1)
    xs_bg = denoise_tv_chambolle(xs_bg, weight=0.05)
    xs_bg -= np.min(xs_bg)
    xs_bg /= np.percentile(xs_bg, 99)

    zs_bg = np.zeros_like(xs_bg)
    zs_bg[(xs_bg >= groom_thresh) & (xs_fg < 0.02) & not_run_or_still] = 1

    zs_fg = np.zeros_like(xs_fg)
    zs_fg[(xs_fg >= groom_thresh) & (xs_bg < 0.02) & not_run_or_still] = 1

    not_grooming = ~zs_bg.astype('bool') & ~zs_fg.astype('bool')

    # abdomen-move
    xs_ab = np.mean(me[:, idxs['body']], axis=1)
    xs_ab_d = denoise_tv_chambolle(xs_ab, weight=0.05)
    xs_ab_d -= np.min(xs_ab_d)
    xs_ab_d /= np.percentile(xs_ab_d, 99)
    zs_ab = np.zeros_like(xs_ab_d)
    zs_ab[(xs_ab_d >= ab_thresh) & not_grooming & not_run_or_still] = 1

    # fidget
    xs_legs = np.max(me[:, np.concatenate([idxs['back'], idxs['mid'], idxs['front']])], axis=1)
    xs_legs = denoise_tv_chambolle(xs_legs, weight=0.05)
    xs_legs -= np.min(xs_legs)
    xs_legs /= np.percentile(xs_legs, 99)

    zs_legs = np.zeros_like(xs_legs)
    zs_legs[
        (xs_legs >= fidget_thresh) &
        not_grooming &
        ~zs_ab.astype('bool') &
        not_run_or_still &
        (ball_me2 < 0.2)] = 1
    # only allow fidgets of >x timepoints
    beg_idx = None
    end_idx = None
    for i in range(len(zs_legs)):
        if zs_legs[i] == 1:
            if beg_idx is None:
                beg_idx = i
        else:
            if beg_idx is not None:
                end_idx = i
            if (beg_idx is not None) and (end_idx is not None) and (end_idx - beg_idx < 10):
                zs_legs[beg_idx:end_idx] = 0
            beg_idx = None
            end_idx = None

    # collect states
    states = np.zeros_like(zs_still, dtype='int')  # default state = 0: unclassified
    states[zs_still == 1] = 1
    states[zs_run == 1] = 2
    states[zs_fg == 1] = 3
    states[zs_bg == 1] = 4
    states[zs_ab == 1] = 5
    states[zs_legs == 1] = 6
    state_mapping = {
        0: 'undefined',
        1: 'still',
        2: 'walk',
        3: 'front_groom',
        4: 'back_groom',
        5: 'abdomen_move',
        6: 'fidget'}

    K = len(state_mapping)

    # smooth states with categorical HMM
    if run_smoother:
        np.random.seed(0)
        models = []
        lps = []
        for i in range(n_restarts):
            models.append(HMM(K=K, D=1, observations='categorical', observation_kwargs=dict(C=K)))
            lps.append(models[i].fit(states[:, None], num_iters=150, tolerance=1e-2))
        best_idx = np.argmax([lp[-1] for lp in lps])
        model = models[best_idx]

        states_ = model.most_likely_states(states[:, None])
        model.permute(find_permutation(states, states_))
        states_new = model.most_likely_states(states[:, None])
    else:
        states_new = np.copy(states)

    return states_new, state_mapping


def plot_state_block(states, j=0, mult=50, l=700):
    fig, axes = plt.subplots(mult, 1, figsize=(30, 0.6 * mult))
    for i, ax in enumerate(axes):
        beg = (j * mult + i) * l
        idxs = np.arange(beg, beg + l)
        ax.imshow(states[None, idxs], aspect='auto', cmap='tab20b')
        ax.set_xticks([]); ax.set_yticks([])
        if not ax.is_first_row():
            ax.spines['top'].set_visible(False)
        if not ax.is_last_row():
            ax.spines['bottom'].set_visible(False)
    plt.tight_layout(pad=0)
