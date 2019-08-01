import numpy as np
import matplotlib.pyplot as plt


def plot_pcs(pcs, color, cmap='inferno'):
    plt.scatter(pcs[:, 0], pcs[:, 1], c=color, cmap=cmap, s=5, linewidths=0)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')


def plot_neural_activity(
        pcs=None, neural_data=None, behavior=None, states=None, slc=(0, 5000),
        cmap_rasters='Greys', cmap_states='tab20b'):

    from matplotlib.gridspec import GridSpec

    T = neural_data.shape[0]
    fig = plt.figure(figsize=(10, 10))

    if T < 5000:
        slc = (0, T)

    n_axes = 0
    ratios = []
    if states is not None:
        n_axes += 1
        ratios.append(0.5)
    if behavior is not None:
        n_axes += 1
        ratios.append(0.5)
    if pcs is not None:
        n_axes += 1
        ratios.append(1)
    if neural_data is not None:
        n_axes += 1
        ratios.append(1)
    gs = GridSpec(n_axes, 1, height_ratios=ratios)

    i = 0

    # discrete states
    if states is not None:
        plt.subplot(gs[i, 0])
        plt.imshow(states[None, slice(*slc)], aspect='auto', cmap=cmap_states)
        plt.xlim(0, slc[1] - slc[0])
        plt.xticks([])
        plt.ylabel('State')
        i += 1
        # behavior
    if behavior is not None:
        plt.subplot(gs[i, 0])
        plt.plot(behavior[slice(*slc)], alpha=0.8)
        plt.xlim(0, slc[1] - slc[0])
        plt.xticks([])
        plt.ylabel('Behavior')
        i += 1
        # pcs
    if pcs is not None:
        D = pcs.shape[1]
        plt.subplot(gs[i, 0])
        plt.plot(pcs[slice(*slc)] / 5 + np.arange(D))
        plt.xlim(plt.xlim(0, slc[1] - slc[0]))
        plt.xticks([])
        plt.ylabel('PC')
        i += 1
    # neural data
    vmin = np.quantile(neural_data[slice(*slc)], 0.01)
    vmax = np.quantile(neural_data[slice(*slc)], 0.99)
    plt.subplot(gs[i, 0])
    plt.imshow(
        neural_data[slice(*slc)].T, aspect='auto',
        cmap=cmap_rasters, vmin=vmin, vmax=vmax)
    plt.xlim(0, slc[1] - slc[0])
    plt.xlabel('Time')
    plt.ylabel('Neuron')

    # align y labels
    for ax in fig.axes:
        ax.yaxis.set_label_coords(-0.12, 0.5)

    return fig


def plot_validation_likelihoods(all_results, T_val=1):
    # plot the log likelihood of the validation data
    fig = plt.figure(figsize=(8, 6))
    for model_name, model_results in all_results.items():
        Ks = sorted(model_results.keys())
        lls_val = np.array([model_results[K]['ll_val'] for K in Ks])
        plt.plot(
            Ks, lls_val / T_val, ls='-', marker='o',
            alpha=1, label=model_name.upper())
    #     plt.legend(loc='lower center', frameon=False)
    plt.legend(loc='lower left', frameon=False)
    plt.xlim(0, max(Ks)+1)
    plt.gca().set_xticks(Ks)
    plt.gca().set_xticklabels(Ks)
    plt.xlabel('Discrete states')
    plt.ylabel('Log probability')
    plt.title('Validation data')
    return fig


def plot_dynamics_matrices(model, deridge=False):
    K = model.K
    n_cols = 2
    n_rows = int(np.ceil(K / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    mats = np.copy(model.observations.As)
    if deridge:
        for k in range(K):
            for d in range(model.D):
                mats[k, d, d] = np.nan
        clim = np.nanmax(np.abs(mats))
    else:
        clim = np.max(np.abs(model.observations.As))

    for k in range(K):
        plt.subplot(n_rows, n_cols, k + 1)
        im = plt.imshow(mats[k], cmap='RdBu_r', clim=[-clim, clim])
        plt.xticks([])
        plt.yticks([])
        plt.title('State %i' % k)
    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def plot_dlc_arhmm_states(
        dlc_labels=None, states=None, state_probs=None, slc=(0, 1000)):
    """
    Args:
        dlc_labels (dict): keys are 'x', 'y', 'l', each value is a TxD np array
        states (np array): length T
        state_probs (np array): T x K
    """

    n_dlc_comp = dlc_labels['x'].shape[1]

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 10),
        gridspec_kw={'height_ratios': [0.1, 0.1, 0.4, 0.4]})

    i = 0
    axes[i].imshow(states[None, slice(*slc)], aspect='auto', cmap='tab20b')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title('State')

    i = 1
    n_states = state_probs.shape[1]
    xs_ = [np.arange(slc[0], slc[1]) for _ in range(n_states)]
    ys_ = [state_probs[slice(*slc), j] for j in range(n_states)]
    cs_ = [j for j in range(n_states)]
    multiline(xs_, ys_, ax=axes[i], c=cs_, alpha=0.8, cmap='tab20b', lw=3)
    axes[i].set_xticks([])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(-0.1, 1.1)
    axes[i].set_title('State probabilities')

    i = 2
    coord = 'x'
    behavior = 4 * dlc_labels[coord] / np.max(np.abs(dlc_labels[coord])) + \
               np.arange(dlc_labels[coord].shape[1])
    axes[i].plot(np.arange(slc[0], slc[1]), behavior[slice(*slc), :])
    axes[i].set_xticks([])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(-1, n_dlc_comp)
    axes[i].set_title('%s coords' % coord.upper())

    i = 3
    coord = 'y'
    behavior = 4 * dlc_labels[coord] / np.max(np.abs(dlc_labels[coord])) + \
               np.arange(dlc_labels[coord].shape[1])
    axes[i].plot(np.arange(slc[0], slc[1]), behavior[slice(*slc), :])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(-1, n_dlc_comp)
    axes[i].set_title('%s coords' % coord.upper())

    axes[-1].set_xlabel('Time (bins)')
    plt.tight_layout()
    plt.show()

    return fig


def multiline(xs, ys, c, ax=None, **kwargs):
    """
    Plot lines with different colorings
    Taken from:
    For use with plotting ARHMM state probabilities

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    from matplotlib.collections import LineCollection

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def make_syllable_movies(filepath, hparams, latents, states, trial_idxs,
                         data_generator, min_threshold=0, n_buffer=5,
                         n_pre_frames=3):
    plot_n_frames = hparams['plot_n_frames']
    if hparams['plot_frame_rate'] == 'orig':
        raise NotImplementedError
    else:
        plot_frame_rate = hparams['plot_frame_rate']

    [bs, n_channels, y_dim, x_dim] = data_generator.datasets[0][0][
        'images'].shape

    state_indices = get_discrete_chunks(states, include_edges=True)

    movie_dim1 = n_channels * y_dim
    movie_dim2 = x_dim

    actual_K = len(state_indices)

    # Get all example over threshold
    over_threshold_instances = [[] for _ in range(actual_K)]
    for i_state in range(actual_K):
        if state_indices[i_state].shape[0] > 0:
            over_threshold_instances[i_state] = state_indices[i_state][
                (np.diff(state_indices[i_state][:, 1:3], 1) > min_threshold)[:,
                0]]
            np.random.shuffle(
                over_threshold_instances[i_state])  # Shuffle instances

    dim1 = int(np.floor(np.sqrt(actual_K)))
    dim2 = int(np.ceil(actual_K / dim1))

    # Initialize syllable movie frames
    plt.clf()
    fig_dim_div = movie_dim2 * dim2 / 10  # aiming for dim 1 being 10
    fig, axes = plt.subplots(dim1, dim2, figsize=(
    (movie_dim2 * dim2) / fig_dim_div, (movie_dim1 * dim1) / fig_dim_div))

    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Syllable ' + str(i), fontsize=16)
    fig.tight_layout(pad=0)

    ims = [[] for _ in range(plot_n_frames + bs + 200)]

    # Loop through syllables
    for i_syllable in range(actual_K):
        print(i_syllable)
        if len(over_threshold_instances[i_syllable]) > 0:
            i_chunk = 0
            i_frame = 0

            while i_frame < plot_n_frames:

                if i_chunk >= len(over_threshold_instances[i_syllable]):
                    im = fig.axes[i_syllable].imshow(
                        np.zeros((movie_dim1, movie_dim2)), animated=True,
                        vmin=0, vmax=1, cmap='gray')
                    ims[i_frame].append(im)
                    i_frame += 1
                else:

                    # Get movies/latents
                    which_trial = trial_idxs[
                        over_threshold_instances[i_syllable][i_chunk, 0]]
                    movie_chunk = data_generator.datasets[0][which_trial][
                                      'images'].cpu().detach().numpy()[max(
                        over_threshold_instances[i_syllable][
                            i_chunk, 1] - n_pre_frames, 0):
                                                                       over_threshold_instances[
                                                                           i_syllable][
                                                                           i_chunk, 2]]
                    # movie_chunk = images[over_threshold_instances[i_syllable][i_chunk,0]][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]]

                    if hparams['lab'] == 'musall':
                        movie_chunk = np.transpose(movie_chunk, (0, 1, 3, 2))
                    movie_chunk = np.concatenate([movie_chunk[:, j] for j in
                                                  range(movie_chunk.shape[1])],
                                                 axis=1)

                    # latents_chunk = latents[over_threshold_instances[i_syllable][i_chunk,0]][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]]

                    # print(states[over_threshold_instances[i_syllable][i_chunk,0]][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):min(over_threshold_instances[i_syllable][i_chunk,2]+1,999)])
                    # print(hmm.most_likely_states(latents[over_threshold_instances[i_syllable][i_chunk,0]])[max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):min(over_threshold_instances[i_syllable][i_chunk,2]+1,999)])

                    # print(data_generator.datasets[0][which_trial]['images'][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]].shape)
                    # pred_latents, _, _ = ae_model.encoding(data_generator.datasets[0][which_trial]['images'][max(over_threshold_instances[i_syllable][i_chunk,1]-n_pre_frames,0):over_threshold_instances[i_syllable][i_chunk,2]])
                    # pred_latents = pred_latents.cpu().detach().numpy()

                    # print(np.max(np.abs(latents_chunk-pred_latents)))
                    # print(np.max(np.abs(latents[0][0:len(pred_latents)]-pred_latents)))
                    if np.sum(states[over_threshold_instances[i_syllable][
                        i_chunk, 0]][
                              over_threshold_instances[i_syllable][i_chunk, 1]:
                              over_threshold_instances[i_syllable][
                                  i_chunk, 2] - 1] != i_syllable) > 0:
                        raise ValueError(
                            'Misaligned states for syllable segmentation')

                    # Loop over this chunk
                    for i in range(movie_chunk.shape[0]):

                        im = fig.axes[i_syllable].imshow(movie_chunk[i],
                                                         animated=True, vmin=0,
                                                         vmax=1, cmap='gray')
                        ims[i_frame].append(im)

                        # Add red box if start of syllable
                        syllable_start = n_pre_frames if \
                        over_threshold_instances[i_syllable][
                            i_chunk, 1] >= n_pre_frames else \
                        over_threshold_instances[i_syllable][i_chunk, 1]

                        if i > syllable_start and i < (syllable_start + 2):
                            rect = matplotlib.patches.Rectangle((5, 5), 10, 10,
                                                                linewidth=1,
                                                                edgecolor='r',
                                                                facecolor='r')
                            im = fig.axes[i_syllable].add_patch(rect)
                            ims[i_frame].append(im)

                        i_frame += 1

                    # Add buffer black frames
                    for j in range(n_buffer):
                        im = fig.axes[i_syllable].imshow(
                            np.zeros((movie_dim1, movie_dim2)), animated=True,
                            vmin=0, vmax=1, cmap='gray')
                        ims[i_frame].append(im)
                        i_frame += 1

                    i_chunk += 1

    ani = animation.ArtistAnimation(fig, [ims[i] for i in range(len(ims)) if
                                          ims[i] != []], interval=20,
                                    blit=True, repeat=False)
    writer = FFMpegWriter(fps=plot_frame_rate, metadata=dict(artist='mrw'),
                          bitrate=-1)
    save_file = os.path.join(filepath, 'syllable_behavior_K_' + str(
        hparams['n_arhmm_states']) + '_kappa_' + str(
        hparams['kappa']) + '_noise_' + hparams[
                                 'noise_type'] + '_nlags_' + str(
        hparams['n_lags']) + '.mp4')
    ani.save(save_file, writer=writer)