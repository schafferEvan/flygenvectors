import os
import numpy as np
import matplotlib.pyplot as plt


def plot_validation_likelihoods(all_results, T_val=1, dict_key='ll_val'):
    # plot the log likelihood of the validation data
    fig = plt.figure(figsize=(8, 6))
    for model_name, model_results in all_results.items():
        Ks = sorted(model_results.keys())
        lls_val = np.array([model_results[K][dict_key] for K in Ks])
        plt.plot(Ks, lls_val / T_val, ls='-', marker='o', label=model_name.upper())
    plt.legend(loc='lower right', frameon=False)
    plt.xlim(min(Ks)-1, max(Ks)+1)
    plt.gca().set_xticks(Ks)
    plt.gca().set_xticklabels(Ks)
    plt.xlabel('Discrete states')
    if dict_key == 'll_val':
        plt.ylabel('Log probability')
    else:
        plt.ylabel(dict_key)
    plt.title('Validation data')
    return fig


def plot_dynamics_matrices(model, deridge=False):
    K = model.K
    n_lags = model.observations.lags
    if n_lags == 1:
        n_cols = 3
        fac = 1
    elif n_lags == 2:
        n_cols = 3
        fac = 1 / n_lags
    elif n_lags == 3:
        n_cols = 3
        fac = 1.25 / n_lags
    elif n_lags == 4:
        n_cols = 3
        fac = 1.50 / n_lags
    elif n_lags == 5:
        n_cols = 2
        fac = 1.75 / n_lags
    else:
        n_cols = 1
        fac = 1
    n_rows = int(np.ceil(K / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows * fac))

    mats = np.copy(model.observations.As)
    if deridge:
        for k in range(K):
            for d in range(model.D):
                mats[k, d, d] = np.nan
        clim = np.nanmax(np.abs(mats))
    else:
        clim = np.max(np.abs(mats))

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


def plot_biases(model):
    fig = plt.figure(figsize=(6, 4))

    mats = np.copy(model.observations.bs.T)
    clim = np.max(np.abs(mats))
    im = plt.imshow(mats, cmap='RdBu_r', clim=[-clim, clim], aspect='auto')
    plt.xlabel('State')
    plt.yticks([])
    plt.ylabel('Observation dimension')
    plt.tight_layout()
    plt.colorbar()
    plt.title('State biases')
    plt.show()
    return fig


def plot_state_transition_matrix(model, deridge=False):
    trans = np.copy(model.transitions.transition_matrix)
    if deridge:
        n_states = trans.shape[0]
        for i in range(n_states):
            trans[i, i] = np.nan
        clim = np.nanmax(np.abs(trans))
    else:
        clim = 1
    fig = plt.figure()
    plt.imshow(trans, clim=[-clim, clim], cmap='RdBu_r')
    plt.colorbar()
    plt.title('State transition matrix')
    plt.show()
    return fig


def plot_covariance_matrices(model):
    K = model.K
    n_cols = 3
    n_rows = int(np.ceil(K / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    mats = np.copy(model.observations.Sigmas)
    clim = np.quantile(np.abs(mats), 0.95)

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


def make_syllable_plots(
        labels, max_snippets=5, max_t=100, coord='y', states_to_plot=None):
    """
    Plot snippets of dlc coordinates for each syllable

    Args:
        labels (list of dicts):
        max_snippets (int): max snippets per state (max rows)
        max_t (int): max number of time bins for each snippet
        coord (str): 'y', 'x'
        states_to_plot (list)

    Returns:
        figure handle
    """
    if states_to_plot is None:
        K = len(labels)
        states = np.arange(K)
    else:
        K = len(states_to_plot)
        states = np.array(states_to_plot)
    n_snippets = [np.min([len(s['x']), max_snippets]) for s in labels]

    fig, axes = plt.subplots(
        max(n_snippets), K, figsize=(4 * K, 2 * max(n_snippets)))
    if len(axes.shape) == 1:
        axes = axes[None, :]
    for ax1 in axes:
        for ax2 in ax1:
            ax2.set_axis_off()

    # get max length on time axis
    n_ts = [np.min([max_t, np.max([len(s) for s in labels[k]['x'][:n_snippets[k]]])])
        for k in states]

    # get max val of coordinate
    max_val = np.max([np.max(np.concatenate(labels[k][coord], axis=0))
                      for k in states])

    for i, k in enumerate(states):
        for j in range(n_snippets[k]):
            behavior = 4 * labels[k][coord][j] / max_val + \
                np.arange(labels[k][coord][j].shape[1])
            ax = axes[j, i]
            ax.set_axis_on()
            ax.plot(behavior)
            ax.set_xlim([0, n_ts[i]])
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


def make_syllable_movie(
        filename, states, frames, frame_indxs, min_threshold=5, n_buffer=5,
        n_pre_frames=3, framerate=20, plot_n_frames=1000, single_state=None,
        start_box=False):
    """
    Adapted from Ella Batty

    Args:
        filename (str): absolute path
        states (list of np arrays):
        frames (np array): T x ypix x xpix
        frame_indxs (list of array-like): indices into `frames` that correspond
            to `states`
        min_threshold (int): minimum length of states
        n_buffer (int): black frames between syllable instances
        n_pre_frames (int): n frames before syllabel start
        framerate (float): Hz
        plot_n_frames (int): length of movie
        single_state (int or NoneType): choose only a single state for movie
        start_box (bool): include red box in each panel indicating state onset
    """

    from matplotlib.patches import Rectangle
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    # hard coded params
    im_kwargs = {'animated': True, 'vmin': 0, 'vmax': np.max(frames), 'cmap': 'gray'}
    txt_kwargs = {
        'fontsize': 16, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top'}
    txt_offset_x = 5
    txt_offset_y = 5

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

    # initialize syllable movie frames
    plt.clf()
    fig_dim_div = x_pix * n_cols / fig_width
    fig_width = (x_pix * n_cols) / fig_dim_div
    fig_height = (y_pix * n_rows) / fig_dim_div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    for i, ax in enumerate(fig.axes):
        ax.set_yticks([])
        ax.set_xticks([])
        if i >= K:
            ax.set_axis_off()
        # elif single_state is not None:
        #     ax.set_title('Syllable %i' % single_state, fontsize=16)
        # else:
        #     ax.set_title('Syllable %i' % i, fontsize=16)
    fig.tight_layout(pad=0, h_pad=1.05)

    ims = [[] for _ in range(plot_n_frames + 500)]

    # loop through syllables
    for i_k, ax in enumerate(fig.axes):

        # skip if no syllable in this axis
        if i_k >= K:
            continue

        print('processing syllable %i/%i' % (i_k + 1, K))

        if single_state is not None:
            i_k = single_state

        if len(states_list[i_k]) == 0:
            continue

        if single_state is not None:
            state_txt = '%i' % i_k
        elif K < 10:
            state_txt = '%i' % i_k
        else:
            state_txt = '%02i' % i_k

        i_chunk = 0
        i_frame = 0
        while i_frame < plot_n_frames:

            if i_chunk >= len(states_list[i_k]):
                # plot black
                im = ax.imshow(np.zeros((y_pix, x_pix)), **im_kwargs)
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
                    raise ValueError('Misaligned states for syllable segmentation')

                # loop over this chunk
                for i in range(movie_chunk.shape[0]):

                    # in case chunk is too long
                    if i_frame >= plot_n_frames:
                        continue

                    im = ax.imshow(movie_chunk[i], **im_kwargs)
                    ims[i_frame].append(im)

                    # Add red box if start of syllable
                    if start_box:
                        if syllable_start < i < (syllable_start + 2):
                            rect = Rectangle(
                                (5, 5), 10, 10, linewidth=1, edgecolor='r', facecolor='r')
                            im = ax.add_patch(rect)
                            ims[i_frame].append(im)

                    im = ax.text(txt_offset_x, txt_offset_y, state_txt, **txt_kwargs)
                    ims[i_frame].append(im)

                    i_frame += 1

                # add buffer black frames
                for j in range(n_buffer):
                    # in case chunk is too long
                    if i_frame >= plot_n_frames:
                        continue
                    im = ax.imshow(
                        np.zeros((y_pix, x_pix)), **im_kwargs)
                    ims[i_frame].append(im)
                    im = ax.text(txt_offset_x, txt_offset_y, state_txt, **txt_kwargs)
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
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ani.save(filename, writer=writer)
    print('done')


def make_labeled_movie(
        filename, states, frames, frame_indxs, state_mapping, framerate=20):
    """
    Label frames with state names

    Args:
        filename (str): absolute path
        states (np array):
        frames (np array): T x ypix x xpix
        frame_indxs (list of array-like): indices into `frames` that correspond
            to `states`
        state_mapping (dict): keys are states (ints), values are labels (strs)
        framerate (float): Hz
    """

    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    n_frames = len(frame_indxs)
    if isinstance(state_mapping[0], str):
        plot_colors = False
    else:
        from matplotlib.patches import Rectangle
        plot_colors = True

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_yticks([])
    ax.set_xticks([])

    im_kwargs = {'animated': True, 'vmin': 0, 'vmax': np.max(frames), 'cmap': 'gray'}
    txt_kwargs = {
        'fontsize': 20, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top', 'transform': ax.transAxes}

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(n_frames):

        if i % 100 == 0:
            print('processing frame %03i/%03i' % (i, n_frames))

        ims_curr = []
        im = ax.imshow(frames[frame_indxs[i], :, :], **im_kwargs)
        ims_curr.append(im)
        if plot_colors:
            rect = Rectangle(
                (5, 5), 10, 10, linewidth=1,
                edgecolor=state_mapping[states[i]],
                facecolor=state_mapping[states[i]])
            im = ax.add_patch(rect)
        else:
            im = ax.text(0.03, 0.97, state_mapping[states[i]], **txt_kwargs)
        ims_curr.append(im)

        ims.append(ims_curr)

    fig.tight_layout(pad=0)
    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig, [ims[i] for i in range(len(ims)) if ims[i] != []], blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % filename, end='')
    writer = FFMpegWriter(fps=framerate, bitrate=-1)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ani.save(filename, writer=writer)
    print('done')


def make_labeled_movie_wmarkers(
        filename, states, frames, markers, frame_indxs, state_mapping, framerate=20):
    """
    Label frames with state names

    Args:
        filename (str): absolute path
        states (np array): shape (T,)
        frames (np array): shape (T, ypix, xpix)
        markers (np array): shape (T, 2 * n_markers) (all x dims then all y dims)
        frame_indxs (list of array-like): indices into `frames` that correspond
            to `states`
        state_mapping (dict): keys are states (ints), values are labels (strs)
        framerate (float): Hz
    """

    from matplotlib.gridspec import GridSpec
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    n_frames = len(frame_indxs)
    n_markers = int(markers.shape[1] / 2)
    times = np.arange(n_frames)
    if isinstance(state_mapping[0], str):
        plot_colors = False
    else:
        from matplotlib.patches import Rectangle
        plot_colors = True

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(
        2, 1, figure=fig, height_ratios=[5, 3], hspace=0.05, wspace=0,
        left=0.01, right=0.99, top=0.99, bottom=0.01)
    axes = []
    axes.append(fig.add_subplot(gs[0]))
    axes.append(fig.add_subplot(gs[1]))
    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks([])

    im_kwargs = {'animated': True, 'vmin': 0, 'vmax': np.max(frames), 'cmap': 'gray'}
    txt_kwargs = {
        'fontsize': 20, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'top'}
    tr_kwargs = {'animated': True, 'linewidth': 2}

    spc = 1.0 * abs(markers.max())
    plotting_markers = markers[:, n_markers:] + spc * np.arange(n_markers)
    ymin = min(-spc - 1, np.min(plotting_markers))
    ymax = max(spc * n_markers, np.max(plotting_markers))

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = [[] for _ in range(n_frames)]
    for i in range(n_frames):

        if i % 100 == 0:
            print('processing frame %03i/%03i' % (i, n_frames))

        ims_curr = []

        # ----------------
        # behavioral video
        # ----------------
        im = axes[0].imshow(frames[frame_indxs[i], :, :], **im_kwargs)
        ims_curr.append(im)
        if plot_colors:
            rect = Rectangle(
                (5, 5), 10, 10, linewidth=1,
                edgecolor=state_mapping[states[frame_indxs[i]]],
                facecolor=state_mapping[states[frame_indxs[i]]])
            im = axes[0].add_patch(rect)
        else:
            im = axes[0].text(
                0.03, 0.97, state_mapping[states[frame_indxs[i]]],
                transform=axes[0].transAxes, **txt_kwargs)
        ims_curr.append(im)

        # ----------------
        # markers + states
        # ----------------
        # plot all states (to keep colormap from changing as new states are added)
        im = axes[1].imshow(
            states[None, frame_indxs], aspect='auto', extent=(0, n_frames, ymin, ymax),
            cmap='tab20b', alpha=0.9)
        ims_curr.append(im)
        # plot markers
        for n in range(n_markers):
            im = axes[1].plot(times, plotting_markers[frame_indxs, n], color='k', **tr_kwargs)[0]
            ims_curr.append(im)
        # cover with semi-transparent box
        if i + 1 < n_frames:
            im = axes[1].imshow(
                np.zeros((1, 1)), aspect='auto', extent=(i + 1, n_frames, ymin, ymax),
                cmap='Greys', alpha=0.8, zorder=2)
            # above: set zorder so on top of line plots
            # z=1: states
            # z=0: markers
            ims_curr.append(im)

        ims[i] = ims_curr

    fig.tight_layout(pad=0)
    print('creating animation...', end='')
    ani = animation.ArtistAnimation(
        fig, [ims[i] for i in range(len(ims)) if ims[i] != []], blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % filename, end='')
    writer = FFMpegWriter(fps=framerate, bitrate=-1)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ani.save(filename, writer=writer)
    print('done')


def plot_states_block(states, mult=50):
    fig, axes = plt.subplots(mult, 1, figsize=(30, 0.6 * mult))
    for i, ax in enumerate(axes):
        ax.imshow(states[i][None, :], aspect='auto', cmap='tab20b')
        ax.set_xticks([])
        ax.set_yticks([])
        if not ax.is_first_row():
            ax.spines['top'].set_visible(False)
        if not ax.is_last_row():
            ax.spines['bottom'].set_visible(False)
    plt.tight_layout(pad=0)
    return fig


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