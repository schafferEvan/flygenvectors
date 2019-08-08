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
    states | state probs | x coords | y coords

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
    _multiline(xs_, ys_, ax=axes[i], c=cs_, alpha=0.8, cmap='tab20b', lw=3)
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


def make_syllable_movie(
        filename, states, frames, frame_indxs, min_threshold=5, n_buffer=5,
        n_pre_frames=3, framerate=20, plot_n_frames=1000, single_state=None):
    """
    Adapted from Ella Batty

    Args:
        filename (str): absolute path
        states (list of np arrays):
        frames (np array): T x ypix x xpix
        min_threshold (int): minimum length of states
        n_buffer (int): black frames between syllable instances
        n_pre_frames (int): n frames before syllabel start
        framerate (float): Hz
        plot_n_frames (int): length of movie
        single_state (int or NoneType): choose only a single state for movie
    """

    from matplotlib.patches import Rectangle
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    if len(frames.shape) == 3:
        [T, y_pix, x_pix] = frames.shape
        n_channels = 1
    elif len(frames.shape) == 4:
        [T, y_pix, x_pix, n_channels] = frames.shape

    # separate states
    if not isinstance(states, list):
        states = [states]
    state_indices = _get_state_runs(states, include_edges=True)
    K = len(state_indices)

    # get all example over threshold
    states_list = [[] for _ in range(K)]
    for curr_state in range(K):
        if state_indices[curr_state].shape[0] > 0:
            states_list[curr_state] = state_indices[curr_state][
                (np.diff(state_indices[curr_state][:, 1:3], 1) > min_threshold)[:, 0]]

    if single_state is not None:
        K = 1
        fig_width = 5
    else:
        fig_width = 10
    n_rows = int(np.floor(np.sqrt(K)))
    n_cols = int(np.ceil(K / n_rows))
    vmin = 0
    vmax = np.max(frames)

    # initialize syllable movie frames
    plt.clf()
    fig_dim_div = x_pix * n_cols / fig_width
    fig_width = (x_pix * n_cols) / fig_dim_div
    fig_height = (y_pix * n_rows) / fig_dim_div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Syllable ' + str(i), fontsize=16)
    fig.tight_layout(pad=0)

    ims = [[] for _ in range(plot_n_frames + 500)]

    # loop through syllables
    for i_k, ax in enumerate(fig.axes):
        print('processing syllable %i/%i' % (i_k + 1, K))

        if single_state is not None:
            i_k = single_state

        if len(states_list[i_k]) == 0:
            continue

        i_chunk = 0
        i_frame = 0
        while i_frame < plot_n_frames:

            if i_chunk >= len(states_list[i_k]):
                # plot black
                im = ax.imshow(
                    np.zeros((y_pix, x_pix)), animated=True,
                    vmin=vmin, vmax=vmax, cmap='gray')
                ims[i_frame].append(im)
                i_frame += 1
            else:
                # get indices into state chunk
                i_idx = states_list[i_k][i_chunk, 0]
                i_beg = states_list[i_k][i_chunk, 1]
                i_end = states_list[i_k][i_chunk, 2]
                # use these to get indices into frames
                m_beg = frame_indxs[i_idx][max(0, i_beg - n_pre_frames)]
                m_end = frame_indxs[i_idx][i_end]
                # grab movie chunk
                movie_chunk = frames[m_beg:m_end]

                # find syllable start
                if i_beg >= n_pre_frames:
                    syllable_start = n_pre_frames
                else:
                    syllable_start = i_beg

                # basic error check
                i_non_k = states[i_idx][i_beg:i_end] != i_k
                if np.any(i_non_k):
                    raise ValueError(
                        'Misaligned states for syllable segmentation')

                # loop over this chunk
                for i in range(movie_chunk.shape[0]):

                    im = ax.imshow(
                        movie_chunk[i], animated=True,
                        vmin=vmin, vmax=vmax, cmap='gray')
                    ims[i_frame].append(im)

                    # Add red box if start of syllable
                    if syllable_start < i < (syllable_start + 2):
                        rect = Rectangle(
                            (5, 5), 10, 10, linewidth=1, edgecolor='r',
                            facecolor='r')
                        im = ax.add_patch(rect)
                        ims[i_frame].append(im)

                    i_frame += 1

                # add buffer black frames
                for j in range(n_buffer):
                    im = ax.imshow(
                        np.zeros((y_pix, x_pix)), animated=True,
                        vmin=vmin, vmax=vmax, cmap='gray')
                    ims[i_frame].append(im)
                    i_frame += 1

                i_chunk += 1

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig, [ims[i] for i in range(len(ims)) if ims[i] != []],
        blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % filename, end='')
    writer = FFMpegWriter(fps=framerate, bitrate=-1)
    ani.save(filename, writer=writer)
    print('done')


def make_syllable_plots(labels, max_snippets=5, max_t=100, coord='y'):
    """
    Plot snippets of dlc coordinates for each syllable

    Args:
        labels (list of dicts):
        max_snippets (int): max snippets per state (max rows)
        max_t (int): max number of time bins for each snippet
        coord (str): 'y', 'x'

    Returns:
        figure handle
    """

    K = len(labels)
    states = np.arange(K)
    n_snippets = [np.min([len(s['x']), max_snippets]) for s in labels]

    fig, axes = plt.subplots(
        max(n_snippets), K, figsize=(4 * K, 2 * max(n_snippets)))
    if len(axes.shape) == 1:
        axes = axes[None, :]
    for ax1 in axes:
        for ax2 in ax1:
            ax2.set_axis_off()

    # get max length on time axis
    n_ts = [
        np.min([
            max_t,
            np.max([len(s) for s in labels[k]['x'][:n_snippets[k]]])])
        for k in states]

    # get max val of coordinate
    max_val = np.max([np.max(np.concatenate(labels[k][coord], axis=0))
                      for k in states])

    for k in states:
        for j in range(n_snippets[k]):
            behavior = 4 * labels[k][coord][j] / max_val + \
                np.arange(labels[k][coord][j].shape[1])
            ax = axes[j, k]
            ax.set_axis_on()
            ax.plot(behavior)
            ax.set_xlim([0, n_ts[k]])
            ax.set_yticks([])
            if ax.is_first_col():
                ax.set_ylabel('Example %i' % j)
            if ax.is_first_row():
                ax.set_title(
                    'DLC %s coordinates: state %i' % (coord.upper(), k))
            if not ax.is_last_row():
                ax.set_xticks([])
            if ax.is_last_row():
                ax.set_xlabel('Time (bins)')
    plt.tight_layout()
    plt.show()

    return fig


def _multiline(xs, ys, c, ax=None, **kwargs):
    """
    Plot lines with different colorings
    Taken from: https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
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


def _get_state_runs(states, include_edges=True):
    """
    Find occurrences of each discrete state
    Author: Ella Batty

    Args:
        states (list): each entry is numpy array containing discrete state for
            each frame
        include_edges (bool): include states at start and end of chunk

    Returns:
        indexing_list: list of length discrete states, each list contains all
            occurences of that discrete state by
            [chunk number, starting index, ending index]

    """

    max_state = max([max(x) for x in states])
    indexing_list = [[] for x in range(max_state + 1)]

    for i_chunk, chunk in enumerate(states):

        # pad either side so we get start and end chunks
        chunk = np.pad(chunk, (1, 1), mode='constant', constant_values=-1)
        # Don't add 1 because of start padding, now index in original unpadded data
        split_indices = np.where(np.ediff1d(chunk) != 0)[0]
        # Last index will be 1 higher that it should be due to padding
        split_indices[-1] -= 1

        for i in range(len(split_indices) - 1):

            # get which state this chunk was (+1 because data is still padded)
            which_state = chunk[split_indices[i] + 1]

            if not include_edges:  # if not including the edges
                if split_indices[i] != 0 and split_indices[i + 1] != (len(chunk) - 2 - 1):
                    indexing_list[which_state].append(
                        [i_chunk, split_indices[i], split_indices[i + 1]])
            else:
                indexing_list[which_state].append(
                    [i_chunk, split_indices[i], split_indices[i + 1]])

    # Convert lists to numpy arrays
    indexing_list = [np.asarray(indexing_list[i_state]) for i_state in
                     range(max_state + 1)]

    return indexing_list
