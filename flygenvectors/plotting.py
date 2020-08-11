import os
import numpy as np
import matplotlib.pyplot as plt
import pdb


def make_clust_fig(k_id, cIds, data_dict, expt_id='', nToPlot=10, include_feeding=False):
    from matplotlib import colors
    from matplotlib import axes, gridspec

    plot_time = data_dict['tPl']
    dFF = data_dict['dFF']
    behavior = data_dict['ball']
    dims = np.squeeze(data_dict['dims'])
    background_im = data_dict['im']
    A = data_dict['A']

    cmap = plt.cm.hsv #plt.cm.OrRd
    plotMx = np.min((nToPlot,len(cIds)))
    
    
    # plot sample traces
    plt.rcParams.update({'font.size': 18})
    #     fig, ax = plt.subplots(figsize=(20, 20))
    plt.figure(figsize=(10.5, 10.5))
    gridspec.GridSpec(6,6)

    if(not include_feeding):
        grid_tuple = (8,1)
        beh_subplot = (0,0)
        dff_subplot = (1,0)
        roi_subplot = (4,0)
    else:
        grid_tuple = (9,1)
        beh_subplot = (1,0)
        dff_subplot = (2,0)
        roi_subplot = (5,0)
        ax = plt.subplot2grid(grid_tuple, (0,0), colspan=1, rowspan=1)
        plt.plot(plot_time, data_dict['drink'],'k')
        plt.box()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('feeding')
        plt.tight_layout()
        plt.title(expt_id+': Cluster '+str(k_id)+', nCells='+str(len(cIds)))
        
    
    
    ax = plt.subplot2grid(grid_tuple, beh_subplot, colspan=1, rowspan=1)
    plt.plot(plot_time, behavior,'k')
    plt.box()
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('locomotion')
    plt.tight_layout()
    if(not include_feeding):
        plt.title(expt_id+': Cluster '+str(k_id)+', nCells='+str(len(cIds)))
    

    ax = plt.subplot2grid(grid_tuple, dff_subplot, colspan=1, rowspan=3)
    plt.plot(plot_time, dFF[cIds[:plotMx],:].T)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.xlabel('Time (s)')
    plt.ylabel('Ratiometric $\Delta F/F$')
    
    
    # show all cells belonging to current cluster
    nC = np.shape(A)[1]
    cIdVec = np.zeros((nC,1))
    cIdVec[cIds]=1
    Ar = A*cIdVec #np.sum( A[:,cIds], axis=1 )
    R = np.reshape(Ar,dims, order='F')
    rIm = np.max(R,axis=2)
    crs = colors.Normalize(0, 1, clip=True)(rIm)
    crs = cmap(crs)
    crs[..., -1] = rIm #setting alpha for transparency

    ax = plt.subplot2grid(grid_tuple, roi_subplot, colspan=1, rowspan=4)
    plt.imshow(np.max(background_im,axis=2),cmap=plt.get_cmap('gray'));
    plt.imshow(crs); ax.axis('off')

    return R


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


def plot_dlc_arhmm_states(
        dlc_labels=None, states=None, state_probs=None, slc=(0, 1000), m=20):
    """
    states | state probs | x coords | y coords

    Args:
        dlc_labels (dict): keys are 'x', 'y', 'l', each value is a TxD np array
        states (np array): length T
        state_probs (np array): T x K
    """

    n_dlc_comp = dlc_labels['x'].shape[1]

    if state_probs is not None:
        fig, axes = plt.subplots(
            4, 1, figsize=(12, 10),
            gridspec_kw={'height_ratios': [0.1, 0.1, 0.4, 0.4]})
    else:
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 10),
            gridspec_kw={'height_ratios': [0.1, 0.4, 0.4]})

    i = 0
    axes[i].imshow(states[None, slice(*slc)], aspect='auto', cmap='tab20b')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title('State')

    if state_probs is not None:
        i += 1
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

    i += 1
    coord = 'x'
    behavior = m * dlc_labels[coord] / np.max(np.abs(dlc_labels[coord])) + \
               np.arange(dlc_labels[coord].shape[1])
    axes[i].plot(np.arange(slc[0], slc[1]), behavior[slice(*slc), :])
    axes[i].set_xticks([])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(0, n_dlc_comp + 1)
    axes[i].set_title('%s coords' % coord.upper())

    i += 1
    coord = 'y'
    behavior = m * dlc_labels[coord] / np.max(np.abs(dlc_labels[coord])) + \
               np.arange(dlc_labels[coord].shape[1])
    axes[i].plot(np.arange(slc[0], slc[1]), behavior[slice(*slc), :])
    axes[i].set_xlim(slc[0], slc[1])
    axes[i].set_yticks([])
    axes[i].set_ylim(0, n_dlc_comp + 1)
    axes[i].set_title('%s coords' % coord.upper())

    axes[-1].set_xlabel('Time (bins)')
    plt.tight_layout()
    plt.show()

    return fig


#############################
# SSM-specific plotting utils
#############################
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


def get_model_fit_as_dict(model_fit):
    # takes list of dicts and returns dict of lists
    reg_labels = list(model_fit[0].keys())
    fit_dict = {}
    for j in range(len(reg_labels)):
        vals = [None]*len(model_fit)
        for i in range(len(model_fit)):
            vals[i] = model_fit[i][reg_labels[j]]
        fit_dict[reg_labels[j]] = np.array(vals)
    return fit_dict

    # tau = np.zeros(len(model_fit))
    # phi = np.zeros(len(model_fit))
    # beta_0 = np.zeros(len(model_fit))
    # if isinstance(model_fit[0]['beta_1'],np.ndarray):
    #     beta_1 = np.zeros((len(model_fit),len(model_fit[0]['beta_1'])))
    # else:
    #     beta_1 = np.zeros((len(model_fit),1))
    # rsq = np.zeros(len(model_fit))
    # rsq_null = np.zeros(len(model_fit))
    # stat = np.zeros( (len(model_fit), np.shape(model_fit[0]['stat'])[0] ) )
    # success = []
    # for i in range(len(model_fit)):
    #     tau[i] = model_fit[i]['tau']
    #     phi[i] = model_fit[i]['phi']
    #     beta_0[i] = model_fit[i]['beta_0']
    #     beta_1[i,:] = model_fit[i]['beta_1']
    #     rsq[i] = model_fit[i]['r_sq']
    #     # rsq_null[i] = model_fit[i]['r_sq_null']
    #     for j in range(np.shape(model_fit[0]['stat'])[0]):
    #         stat[i,j] = model_fit[i]['stat'][j][1]
    #     success.append(model_fit[i]['success'])
    # fit_dict['tau'] = tau
    # fit_dict['phi'] = phi
    # fit_dict['beta_0'] = beta_0
    # fit_dict['beta_1'] = beta_1
    # fit_dict['rsq'] = rsq
    # # fit_dict['rsq_null'] = rsq_null
    # fit_dict['stat'] = stat
    # # fit_dict['success'] = success
    # return fit_dict


def show_param_scatter(model_fit, data_dict, param_name, pval=.01):
    f = get_model_fit_as_dict(model_fit)
    param = f[param_name]
    rsq = f['r_sq']
    stat = np.zeros(len(param))
    for i in range(len(param)):
        stat[i] = f['stat'][i][param_name][0][1]
    success = f['success']

    if(param_name=='phi'): 
        param = param/data_dict['scanRate']
        label = 'Phase (s)'
    elif(param_name=='beta_0'): 
        label = r'$\beta_0$'

    pval_text = pval #0.01
    sig = (stat<pval) #*(rsq>rsq_null) #*(stat>0) # 1-sided test that behavior model > null model
    param_sig = param[success*sig]
    param_notsig = param[np.logical_not(success*sig)]
    rsq_sig = rsq[success*sig]
    rsq_notsig = rsq[np.logical_not(success*sig)]

    plt.figure(figsize=(5, 5))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    # ax_histx.set_xscale('log')
    ax_histx.axis('off')
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.axis('off')


    xmin = np.floor(param.min()) #4/scanRate
    xmax = np.ceil(param.max()) #30 #1000/scanRate #6000
    ymin = 0
    ymax = 1

    ax_scatter.scatter(param_notsig,rsq_notsig,c='tab:gray',marker='.',alpha=0.3)
    ax_scatter.scatter(param_sig,rsq_sig,c='tab:blue',marker='.',alpha=0.3) #'#1f77b4'
    # ax_scatter.scatter(tau_sig[tau_is_neg_sig],rsq_sig[tau_is_neg_sig],c='tab:red',marker='.',alpha=0.3)
    # ax_scatter.set_xscale('log')
    ax_scatter.set_xlim((xmin,xmax))
    ax_scatter.set_ylim((ymin,ymax))
    ax_scatter.set_xlabel(label)
    ax_scatter.set_ylabel(r'$r^2$')

    xbinwidth = (xmax-xmin)/40
    ybinwidth = 0.025
    ybins = np.arange(ymin, ymax+ybinwidth, ybinwidth)
    xbins = np.arange(xmin, xmax+xbinwidth, xbinwidth) #np.logspace(np.log(xmin), np.log(xmax),num=len(ybins),base=np.exp(1))
    ax_histx.hist(param, bins=xbins,color='#929591') #log=True,
    ax_histx.set_xlim((xmin,xmax))
    ax_histy.hist(rsq, bins=ybins, orientation='horizontal',color='#929591')
    ax_histy.set_ylim((ymin,ymax))


def show_tau_scatter(model_fit, pval=.01):
    f = get_model_fit_as_dict(model_fit)
    tau = abs(f['tau'])
    tau_is_pos = (f['tau']>=0)
    rsq = f['r_sq']
    # rsq_null = f['rsq_null']
    stat = np.zeros(len(tau))
    for i in range(len(tau)):
        stat[i] = f['stat'][i]['tau'][0][1]
    success = f['success']

    pval_text = pval #0.01
    sig = (stat<pval) #*(rsq>rsq_null) #*(stat>0) # 1-sided test that behavior model > null model
    # sig_text = (stat<pval_text)*(rsq>rsq_null) # 1-sided test that behavior model > null model
    # pdb.set_trace()
    tau_sig = tau[success*sig]
    tau_notsig = tau[np.logical_not(success*sig)]
    rsq_sig = rsq[success*sig]
    rsq_notsig = rsq[np.logical_not(success*sig)]
    tau_is_pos_sig = tau_is_pos[success*sig]
    tau_is_neg_sig = np.logical_not(tau_is_pos_sig)
    # print('sum of neg is '+str(sum(tau_is_neg_sig)))
    # print('sum of neg init is '+str(sum(np.logical_not(tau_is_pos))/len(tau_is_pos)))
    # print('sum of neg post is '+str( sum( np.logical_not(tau_is_pos)[success*sig] ) ))

    print('median tau of significant cells = '+str(np.median(tau_sig)))
    print('frac of significant cells w/ tau above 30s = '+str( np.sum(tau_sig>30)/len(tau_sig) ))
    print('fraction of cells with p<'+str(pval)+' = '+str( (success*sig).sum()/len(success) ))

    plt.figure(figsize=(5, 5))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histx.set_xscale('log')
    ax_histx.axis('off')
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.axis('off')


    xmin = 1 #4/scanRate
    xmax = 60 #1000/scanRate #6000
    ymin = 0
    ymax = 1

    ax_scatter.scatter(tau_notsig,rsq_notsig,c='tab:gray',marker='.',alpha=0.3)
    ax_scatter.scatter(tau_sig[tau_is_pos_sig],rsq_sig[tau_is_pos_sig],c='tab:blue',marker='.',alpha=0.3) #'#1f77b4'
    ax_scatter.scatter(tau_sig[tau_is_neg_sig],rsq_sig[tau_is_neg_sig],c='tab:red',marker='.',alpha=0.3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_xlim((xmin,xmax))
    ax_scatter.set_xticks((1,3,10,30))
    ax_scatter.set_xticklabels(('1','3','10','30'))
    ax_scatter.set_ylim((ymin,ymax))
    ax_scatter.set_xlabel('Optimal time constant (s)')
    ax_scatter.set_ylabel(r'$r^2$')

    # xbinwidth = 100
    ybinwidth = 0.05 #0.025
    ybins = np.arange(ymin, ymax+ybinwidth, ybinwidth)
    xbins = np.logspace(np.log10(xmin), np.log10(xmax),num=len(ybins), base=10) #np.exp(1))
    ax_histx.hist(tau, bins=xbins,log=True,color='#929591')
    ax_histx.set_xlim((xmin,xmax))
    ax_histy.hist(rsq, bins=ybins, orientation='horizontal',color='#929591')
    ax_histy.set_ylim((ymin,ymax))


def show_tau_scatter_legacy(tauList, corrMat, data_dict):
    print('legacy version')
    scanRate = data_dict['scanRate']

    plt.figure(figsize=(5, 5))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histx.set_xscale('log')
    ax_histx.axis('off')
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.axis('off')


    xmin = 1 #4/scanRate
    xmax = 100 #1000/scanRate #6000
    ymin = -0.15
    ymax = 0.95

    av = np.max(corrMat,axis=1)
    am = np.argmax(corrMat,axis=1)
    ax_scatter.scatter(tauList[am]/scanRate,av,marker='.',alpha=0.3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_xlim((xmin,xmax))
    ax_scatter.set_ylim((ymin,ymax))
    ax_scatter.set_xlabel('Optimal time constant (s)')
    ax_scatter.set_ylabel('Max Correlation Coefficient')

    xbinwidth = 100
    ybinwidth = 0.05
    ybins = np.arange(ymin, ymax, ybinwidth)
    xbins = np.logspace(np.log(xmin), np.log(xmax),num=len(ybins),base=np.exp(1))
    ax_histx.hist(tauList[am]/scanRate, bins=xbins,log=True)
    ax_histy.hist(av, bins=ybins, orientation='horizontal')


def make_colorCoded_cellMap(feature_vec, clrs, data_dict, cbounds=()):
    # make brain volume with cells color coded by tau

    from matplotlib import cm
    if(not cbounds): cbounds = (-1,clrs+1)
    c_space = np.linspace(cbounds[0], cbounds[1], num=clrs+2)
    
    A = data_dict['A']
    x = np.linspace(0.0, 1.0, num=clrs)
    # rgb_map = np.squeeze(cm.get_cmap(cmap)(x)[np.newaxis, :, :3])

    # make mask volume transparent where there are cells, zero elsewhere
    cell_loc = np.max(A, axis=1).toarray()>0
    cell_loc_im = np.reshape(np.squeeze(cell_loc), data_dict['dims'], order='F') 
    # mask_vol = np.concatenate( (np.zeros(np.concatenate((dims,[3]))),cell_loc_im[...,np.newaxis] ),axis=3 )
    mask_vol = np.ones(np.concatenate((data_dict['dims'],[4])))
    mask_vol[:,:,:,3] = cell_loc_im

    R = np.zeros(data_dict['dims'])
    for j in range(len(c_space)-1):
        cli = np.flatnonzero(  (feature_vec>c_space[j]) & (feature_vec<c_space[j+1])  )
        if(len(cli)>0):
            A_cli = np.max(A[:,cli], axis=1).toarray()
            R += np.reshape(np.squeeze(A_cli*j), data_dict['dims'], order='F')
            
    return R, mask_vol


def make_labels_for_colorbar(label_list, fullList):
    # fullList=tauList/data_dict['scanRate']
    loc_list = np.zeros(len(label_list))
    for i in range(len(label_list)):
        loc_list[i] = np.argmin(np.abs(fullList-label_list[i]))
    return loc_list


def make_colorBar_for_colorCoded_cellMap(R, mask_vol, tauList, tau_label_list, tau_loc_list, data_dict, cmap=''):
    # make COLORBAR for 'slow' cells, color coded by tau
    from matplotlib import cm
    if(not cmap): 
        cmap = make_hot_without_black()
        colorbar_title = ''
    else:
        colorbar_title = 'CC'
    
    fig, ax = plt.subplots()
    cs = ax.imshow(np.max(R, axis=2), aspect='auto', cmap=cmap)
    ax.imshow(1-np.max(mask_vol,axis=2), aspect='auto')
    cs.set_clim(tau_loc_list[0], tau_loc_list[-1])
    cbar = fig.colorbar(cs, ticks=tau_loc_list)
    cbar.ax.set_yticklabels([str(x) for x in tau_label_list])
    if(not colorbar_title):
        cbar.ax.set_title(r'$\tau (s)$')
    else:
        cbar.ax.set_title(colorbar_title)


def make_colorBar_for_colorCoded_cellMap_points(tau_loc_list, tau_label_list, data_dict, model_fit, tau_argmax, cmap='', pval=0.01, color_lims=[0,200]):
    from matplotlib import colors
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap

    point_size = 2
    dims_in_um = data_dict['template_dims_in_um']
    template_dims = data_dict['template_dims']
    dims = data_dict['dims']
    if(not cmap): 
        cmap = make_hot_without_black()
        colorbar_title = ''
    else:
        colorbar_title = 'CC'
    gry = cm.get_cmap('Greys', 15)
    gry = ListedColormap(gry(np.linspace(.8, 1, 2)))

    if type(model_fit) is dict:
        f = get_model_fit_as_dict(model_fit)
        sig = f['success']*(f['stat']<pval)*(f['rsq']>f['rsq_null']) # 1-sided test that behavior model > null model
        not_sig = np.logical_not(sig)
    else:
        # if not dict, assumed to be list
        sig = np.array([i for i in range(len(model_fit))]) #np.ones(len(model_fit))
        not_sig = np.array([]) #np.zeros(len(model_fit))
    

    totScale = 1
    fig=plt.figure(figsize=(8, 8*totScale))

    totWidth = 1.1*(dims_in_um[2] + dims_in_um[1])
    zpx = dims_in_um[2]/totWidth
    ypx = dims_in_um[1]/totWidth
    xpx = dims_in_um[0]/totWidth

    ax2 = plt.axes([.05+zpx,  (.05+zpx)/totScale,  ypx,  xpx/totScale])
    # ax1 = plt.axes([.04,      (.05+zpx)/totScale,  zpx,  xpx/totScale])
    # ax3 = plt.axes([.05+zpx,  .04/totScale,        ypx,  zpx/totScale])

    # if(len(not_sig)):
    #     ax1.scatter(data_dict['aligned_centroids'][not_sig,2],
    #                 data_dict['aligned_centroids'][not_sig,1], c=.5*np.ones(not_sig.sum()), cmap=gry, s=point_size)
    # ax1.scatter(data_dict['aligned_centroids'][sig,2],
    #             data_dict['aligned_centroids'][sig,1], c=tau_argmax[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    # ax1.set_facecolor((0.0, 0.0, 0.0))
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_xlim(0,template_dims[2])
    # ax1.set_ylim(0,template_dims[0])
    # ax1.invert_yaxis()

    if(len(not_sig)):
        ax2.scatter(data_dict['aligned_centroids'][not_sig,0],
                    data_dict['aligned_centroids'][not_sig,1], c=.5*np.ones(not_sig.sum()), cmap=gry, s=point_size)
    ax2.scatter(data_dict['aligned_centroids'][sig,0],
                data_dict['aligned_centroids'][sig,1], c=tau_argmax[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    ax2.set_facecolor((0.0, 0.0, 0.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0,template_dims[1])
    ax2.set_ylim(0,template_dims[0])
    ax2.invert_yaxis()


    ypx_per_um = template_dims[1]/dims_in_um[1]
    scaleBar_um = 50 #50 um
    bar_color = 'w'
    ax2.plot( template_dims[1]*.97-(scaleBar_um*ypx_per_um,0), (template_dims[0]*.93, template_dims[0]*.93),bar_color)

    # if(len(not_sig)):
    #     ax3.scatter(data_dict['aligned_centroids'][not_sig,0],
    #                 data_dict['aligned_centroids'][not_sig,2], c=.5*np.ones(not_sig.sum()), cmap=gry, s=point_size)
    # ax3.scatter(data_dict['aligned_centroids'][sig,0],
    #             data_dict['aligned_centroids'][sig,2], c=tau_argmax[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    # ax3.set_facecolor((0.0, 0.0, 0.0))
    # ax3.set_xticks([])
    # ax3.set_yticks([])
    # ax3.set_xlim(0,template_dims[1])
    # ax3.set_ylim(0,template_dims[2])
    # ax3.invert_yaxis()

    # ***************************************************

    # make COLORBAR for 'slow' cells, color coded by tau
    cbar = fig.colorbar(cs, ticks=tau_loc_list)
    cbar.ax.set_yticklabels([str(x) for x in tau_label_list])
    if(not colorbar_title):
        cbar.ax.set_title(r'$\tau (s)$')
    else:
        cbar.ax.set_title(colorbar_title)


def show_residual_raster(data_dict, model_fit, exp_date):
    import regression_model as model
    tauLim = 100*data_dict['scanRate']
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    model_residual = np.zeros(data_dict['dFF'][:,M-1:].shape)

    for j in range(data_dict['dFF'].shape[0]):
        d = model_fit[j]
        pars = [d['alpha_0'], d['alpha_1'], d['beta_0'], 
                d['beta_1'], d['tau'] ]
        dFF = data_dict['dFF'][j,:]
        data = [t_exp, time, ball, dFF]
        dFF_fit = model.tau_reg_model(pars, data)
        if (d['tau']>=0):
            model_residual[j,:] = dFF[M-1:]-dFF_fit
        else:
            model_residual[j,:] = dFF[:-M+1]-dFF_fit
        

    # pdb.set_trace()
    data_dict_tmp = data_dict.copy()
    data_dict_tmp['dFF'] = model_residual
    data_dict_tmp['tPl'] = data_dict_tmp['tPl'][M-1:]
    data_dict_tmp['ball'] = data_dict_tmp['ball'][M-1:].flatten()
    if (exp_date == '2018_08_24'):
        axes = show_raster_with_behav(data_dict_tmp,color_range=(-0.1,0.1),include_feeding=False,include_dlc=False)
    else:
        data_dict_tmp['dlc'] = data_dict_tmp['dlc'][M-1:,:]
        axes = show_raster_with_behav(data_dict_tmp,color_range=(-0.1,0.1),include_feeding=False,include_dlc=True)
    axes[0].set_title('Residual')


def show_PC_residual_raster(data_dict):
    # regress out smoothed behavior by predicting behavior from activity of all cells, then projecting out this dimension
    import regression_model as model
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA

    f = model.estimate_SINGLE_neuron_behav_reg_model(data_dict['dFF'].sum(axis=0), data_dict['ball'], data_dict['time'], data_dict['scanRate'])
    tau = f[4]
    tauLim = 100*data_dict['scanRate']
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
    beh_conv = np.convolve(kern,data_dict['ball'],'valid')

    dFF = data_dict['dFF'][:,M-1:].T
    pca = PCA(n_components=4)
    pca.fit(dFF)
    regr = LinearRegression()
    pcbasis = pca.transform(dFF)
    # pdb.set_trace()
    regr.fit(pcbasis, beh_conv)
    b = regr.coef_.T/np.linalg.norm(regr.coef_.T)
    aPr = np.dot(pcbasis,b)
    # lowRankRegRes = pcbasis.T - np.outer(b,aPr)
    lowRankReg = np.outer(b,aPr)
    PCReg = pca.inverse_transform(lowRankReg.T)
    PCRegRes = dFF-PCReg

    # pdb.set_trace()
    data_dict_tmp = data_dict.copy()
    data_dict_tmp['dFF'] = PCRegRes.T
    data_dict_tmp['tPl'] = data_dict_tmp['tPl'][M-1:]
    data_dict_tmp['ball'] = data_dict_tmp['ball'][M-1:]
    axes = show_raster_with_behav(data_dict_tmp,color_range=(-0.1,0.1))
    axes[0].set_title('PC_reg Residual')

    



def show_colorCoded_cellMap_points(data_dict, model_fit, plot_param, cmap='', pval=0.01, color_lims_scale=[-0.75,0.75]):
    """
    Plot map of cells for one dataset, all MIPs, colorcoded by desired quantity

    Args:
        data_dict (dict): dictionary for one dataset
        model_fit (list): list of dictionaries for one dataset
        plot_param (string or list): if string, param to use for color code. If list, pair of param to use (str) and index (int)
        color_lims_scale (list): bounds for min and max.  If _scale[0]<0, enforces symmetry of colormap
    """
    from matplotlib import colors
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap

    point_size = 2
    dims_in_um = data_dict['template_dims_in_um']
    template_dims = data_dict['template_dims']
    dims = data_dict['dims']
    if(not cmap): cmap = make_hot_without_black()
    gry = cm.get_cmap('Greys', 15)
    gry = ListedColormap(gry(np.linspace(.8, 1, 2)))
 
    if type(plot_param) is list:
        plot_field = plot_param[0]
        plot_field_idx = plot_param[1]
    else:
        plot_field = plot_param
        plot_field_idx = np.nan
    if type(model_fit) is list:

        # generate color_data from model_fit
        color_data = np.zeros(len(model_fit))
        sig = [] # significance threshold
        for i in range(len(model_fit)):
            if np.isnan(plot_field_idx):
                color_data[i] = model_fit[i][plot_field]
                sig.append( model_fit[i]['stat'][plot_field][0][1]<pval ) # 1-sided test that behavior model > null model
            else:
                color_data[i] = model_fit[i][plot_field][plot_field_idx]
                sig.append( model_fit[i]['stat'][plot_field][plot_field_idx][1]<pval ) # 1-sided test that behavior model > null model
        if plot_field=='tau': color_data = np.log10(color_data)

        # sig cleanup
        not_sig = np.logical_not(sig)
        sig = np.flatnonzero(sig) #.tolist()
        not_sig = np.flatnonzero(not_sig) #.tolist()
    else:
        # this needs to be updated.  the above takes the 'list' case
        # if not dict, assumed to be list. This is for manual curation and testing
        color_data = plot_param
        sig = np.array([i for i in range(len(color_data))]) #np.ones(len(model_fit))
        # sig = np.array([i for i in range(len(model_fit))]) #np.ones(len(model_fit))
        not_sig = np.array([]) #np.zeros(len(model_fit))
    
    if color_lims_scale[0]<0:
        color_lims = [abs(color_lims_scale[0])*min(min(color_data[sig]),-max(color_data[sig])),
                          color_lims_scale[1]*max(-min(color_data[sig]),max(color_data[sig]))]
    else:
        color_lims = [color_lims_scale[0]*min(color_data[sig]), color_lims_scale[1]*max(color_data[sig])]

    totScale = 1
    plt.figure(figsize=(8, 8*totScale))

    totWidth = 1.1*(dims_in_um[2] + dims_in_um[1])
    zpx = dims_in_um[2]/totWidth
    ypx = dims_in_um[1]/totWidth
    xpx = dims_in_um[0]/totWidth

    ax2 = plt.axes([.05+zpx,  (.05+zpx)/totScale,  ypx,  xpx/totScale])
    ax1 = plt.axes([.04,      (.05+zpx)/totScale,  zpx,  xpx/totScale])
    ax3 = plt.axes([.05+zpx,  .04/totScale,        ypx,  zpx/totScale])
    # pdb.set_trace()
    
    if(len(not_sig)):
        ax1.scatter(data_dict['aligned_centroids'][not_sig,2],
                    data_dict['aligned_centroids'][not_sig,1], c=.5*np.ones(len(not_sig)), cmap=gry, s=point_size)
    ax1.scatter(data_dict['aligned_centroids'][sig,2],
                data_dict['aligned_centroids'][sig,1], c=color_data[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    ax1.set_facecolor((0.0, 0.0, 0.0))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0,template_dims[2])
    ax1.set_ylim(0,template_dims[0])
    ax1.invert_yaxis()

    if(len(not_sig)):
        ax2.scatter(data_dict['aligned_centroids'][not_sig,0],
                    data_dict['aligned_centroids'][not_sig,1], c=.5*np.ones(len(not_sig)), cmap=gry, s=point_size)
    ax2.scatter(data_dict['aligned_centroids'][sig,0],
                data_dict['aligned_centroids'][sig,1], c=color_data[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    ax2.set_facecolor((0.0, 0.0, 0.0))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0,template_dims[1])
    ax2.set_ylim(0,template_dims[0])
    ax2.invert_yaxis()


    ypx_per_um = template_dims[1]/dims_in_um[1]
    scaleBar_um = 50 #50 um
    bar_color = 'w'
    ax2.plot( template_dims[1]*.97-(scaleBar_um*ypx_per_um,0), (template_dims[0]*.93, template_dims[0]*.93),bar_color)

    if(len(not_sig)):
        ax3.scatter(data_dict['aligned_centroids'][not_sig,0],
                    data_dict['aligned_centroids'][not_sig,2], c=.5*np.ones(len(not_sig)), cmap=gry, s=point_size)
    ax3.scatter(data_dict['aligned_centroids'][sig,0],
                data_dict['aligned_centroids'][sig,2], c=color_data[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
    ax3.set_facecolor((0.0, 0.0, 0.0))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0,template_dims[1])
    ax3.set_ylim(0,template_dims[2])
    ax3.invert_yaxis()


def convert_plot_param_to_plot_field(plot_param, model_fit, data_dict):
    # mapper to sort different kinds of behavior so plots are grouped correctly
    if type(plot_param) is list:
        plot_field = plot_param[0]
        if plot_field=='beta_0':
            #behavior can be running or flailing. plot_param[1] is meant to be 0 if running and 1 if flailing.
            #For datasets with only feeding and flailing, there's only one trial, but needs to go with 1s
            data_tmp = model_fit[0][plot_field]
            if plot_param[1]==0:
                # running
                if data_dict['stim'].max()>0:
                    # if there's feeding
                    if data_tmp.shape[0]>1:
                        # if there are multiple trials, first was running
                        plot_field_idx = 0
                    else:
                        # single trial feeding has no running
                        plot_field_idx = None
                else:
                    # single trials without feeding are running
                    plot_field_idx = 0
            elif plot_param[1]==1:
                # flailing
                if data_tmp.shape[0]>1:
                    # if there are multiple trials, second was flailing
                    plot_field_idx = 1
                else:
                    # if there's only one trial...
                    if data_dict['stim'].max()>0:
                        # if that trial has feeding, behavior was flailing
                        plot_field_idx = 0 # note: this is sloppy, but not a bug! Handles single trial feeding->flailing.
                    else:
                        # if that trial has no feeding, also has no flailing
                        plot_field_idx = None
        else:
            plot_field_idx = plot_param[1]
    else:
        plot_field = plot_param
        plot_field_idx = np.nan
    return plot_field, plot_field_idx


def generate_color_data_from_model_fit(model_fit, plot_field, plot_field_idx, pval):
    # generate color_data from model_fit
    color_data = np.zeros(len(model_fit))
    sig = [] # significance threshold
    for i in range(len(model_fit)):
        if model_fit[i]['success']:
            if np.isnan(plot_field_idx):
                color_data[i] = model_fit[i][plot_field]
                sig.append( model_fit[i]['stat'][plot_field][1]<pval ) # 1-sided test that behavior model > null model
            else:
                color_data[i] = model_fit[i][plot_field][plot_field_idx]
                sig.append( model_fit[i]['stat'][plot_field][plot_field_idx][1]<pval ) # 1-sided test that behavior model > null model
        else:
            color_data[i] = np.nan
            sig.append( False )
    return color_data, sig


def show_colorCoded_cellMap_points_grid(data_dict_tot, model_fit_tot, plot_param, cmap, color_lims_params, pval=0.01, sort_by=[]):
    """
    Plot map of cells for many datasets, colorcoded by desired quantity

    Args:
        data_dict_tot (list): list of dictionaries for each dataset
        model_fit (list): list of dictionaries for each dataset
        color_data_tot (list OR string): if list, 
    """

    from matplotlib import colors
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap

    point_size = 0.5
    dims_in_um = data_dict_tot[0]['data_dict']['template_dims_in_um']
    template_dims = data_dict_tot[0]['data_dict']['template_dims']
    dims = data_dict_tot[0]['data_dict']['dims']
    if(not cmap): cmap = make_hot_without_black()
    gry = cm.get_cmap('Greys', 15)
    gry = ListedColormap(gry(np.linspace(.8, 1, 2)))
    height_width_ratio = dims_in_um[1]/dims_in_um[0]
    NF = len(data_dict_tot)

    if isinstance(color_lims_params[0],str):
        color_lims_scale = color_lims_params[1]
        if color_lims_params[0]=='static':
            color_lims_are_static = True
            color_lims_master = [np.inf,-np.inf]
        elif color_lims_params[0]=='global':
            color_lims_are_static = True
            color_lims_master = color_lims_params[1]
        elif color_lims_params[0]=='dynamic':
            color_lims_are_static = False
    else:
        color_lims_scale = color_lims_params
        color_lims_are_static = False



    # FINISH THIS LOOP BELOW AND MOVE TO SEPARATE FUNCTION.  ** ALSO ** FIX REFERNCE TO COLOR LIMS IN MAIN LOOP BELOW

    # find static color lims, if desired
    if color_lims_are_static:
        if color_lims_params[0]=='static':
            for nf in range(NF):
                data_dict = data_dict_tot[nf]['data_dict']
                model_fit = model_fit_tot[nf]
                plot_field, plot_field_idx = convert_plot_param_to_plot_field(plot_param, model_fit, data_dict)
                if plot_field_idx is None: continue
                color_data, sig = generate_color_data_from_model_fit(model_fit, plot_field, plot_field_idx, pval)
                # sig cleanup
                not_sig = np.logical_not(sig)
                sig = np.flatnonzero(sig) #.tolist()
                not_sig = np.flatnonzero(not_sig) #.tolist()

                # get color bounds
                if color_lims_scale[0]<0:
                    color_lims = [abs(color_lims_scale[0])*min(min(color_data[sig]),-max(color_data[sig])),
                                      color_lims_scale[1]*max(-min(color_data[sig]),max(color_data[sig]))]
                else:
                    color_lims = [color_lims_scale[0]*min(color_data[sig]), color_lims_scale[1]*max(color_data[sig])]
                color_lims_master = ( np.min((color_lims_master[0],color_lims[0])), np.max((color_lims_master[1],color_lims[1])) )

            
    # # square version
    # n_cols = int(np.ceil(np.sqrt(NF)))
    # n_rows = n_cols
    # if(NF<=n_cols*(n_rows-1)): n_rows -= 1
    # f, ax = plt.subplots(n_rows, n_cols, figsize=(8, 8/height_width_ratio) )
    
    # # rect version
    n_cols = 5
    n_rows = int(np.ceil(NF/n_cols))
    f, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15/(height_width_ratio*n_cols/n_rows)) )
    plt.subplots_adjust(wspace=0.05, hspace=0 )

    for i in range(n_rows):
        for j in range(n_cols):
            ax[i,j].set_axis_off()

    for nf in range(NF):
        data_dict = data_dict_tot[nf]['data_dict']
        model_fit = model_fit_tot[nf]
        
        # check if this fly has the correct parameter to plot, skip otherwise. Note that plot_field_idx=np.nan is used for other purposes
        plot_field, plot_field_idx = convert_plot_param_to_plot_field(plot_param, model_fit, data_dict)
        if plot_field_idx is None: continue

        # generate color_data from model_fit
        color_data, sig = generate_color_data_from_model_fit(model_fit, plot_field, plot_field_idx, pval)
                
        # sig cleanup
        not_sig = np.logical_not(sig)
        sig = np.flatnonzero(sig) #.tolist()
        not_sig = np.flatnonzero(not_sig) #.tolist()

        # get color bounds
        if color_lims_are_static:
            color_lims = color_lims_master
        else:
            if color_lims_scale[0]<0:
                color_lims = [abs(color_lims_scale[0])*min(min(color_data[sig]),-max(color_data[sig])),
                                  color_lims_scale[1]*max(-min(color_data[sig]),max(color_data[sig]))]
            else:
                color_lims = [color_lims_scale[0]*min(color_data[sig]), color_lims_scale[1]*max(color_data[sig])]

        # optional: reorder list for consistent occlusion. options: {'z', 'val', 'inv_val'}. If empty, default is order of ROI ID
        if sort_by:
            if sort_by=='z':
                sort_val = data_dict['aligned_centroids'][sig,2]
            elif sort_by=='val':
                sort_val = color_data[sig]
            elif sort_by=='inv_val':
                sort_val = -color_data[sig]
            sorted_order = np.argsort(sort_val)
            sig = (np.array(sig)[sorted_order]).tolist()


        j = int(np.floor(nf/n_rows))
        i = round(nf - j*n_rows)
        ax[i,j].set_axis_on()
        ax[i,j].set_aspect(height_width_ratio)
        if(len(not_sig)):
            ax[i,j].scatter(data_dict['aligned_centroids'][not_sig,0],
                        data_dict['aligned_centroids'][not_sig,1], c=.5*np.ones(len(not_sig)), cmap=gry, s=point_size)
        ax[i,j].scatter(data_dict['aligned_centroids'][sig,0],
                    data_dict['aligned_centroids'][sig,1], c=color_data[sig], cmap=cmap, s=point_size, vmin=color_lims[0], vmax=color_lims[1])
        ax[i,j].set_facecolor((0.0, 0.0, 0.0))
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].set_xlim(0,template_dims[1])
        ax[i,j].set_ylim(0,template_dims[0])
        ax[i,j].invert_yaxis()

        ypx_per_um = template_dims[1]/dims_in_um[1]
        scaleBar_um = 50 #50 um
        bar_color = 'w'
        ax[i,j].plot( template_dims[1]*.97-(scaleBar_um*ypx_per_um,0), (template_dims[0]*.93, template_dims[0]*.93),bar_color)


        



def show_colorCoded_cellMap(R, mask_vol, color_lims, data_dict, cmap=''):
    from matplotlib import colors

    dims_in_um = data_dict['dims_in_um']
    dims = data_dict['dims']
    if(not cmap): cmap = make_hot_without_black()
    
    totScale = 1
    totWidth = 1.1*(dims_in_um[2] + dims_in_um[1])
    zpx = dims_in_um[2]/totWidth
    ypx = dims_in_um[1]/totWidth
    xpx = dims_in_um[0]/totWidth

    totScale = 1
    plt.figure(figsize=(8, 8*totScale))

    totWidth = 1.1*(dims_in_um[2] + dims_in_um[1])
    zpx = dims_in_um[2]/totWidth
    ypx = dims_in_um[1]/totWidth
    xpx = dims_in_um[0]/totWidth

    ax2 = plt.axes([.05+zpx,  (.05+zpx)/totScale,  ypx,  xpx/totScale])
    ax1 = plt.axes([.04,      (.05+zpx)/totScale,  zpx,  xpx/totScale])
    ax3 = plt.axes([.05+zpx,  .04/totScale,      ypx,  zpx/totScale])

    if np.shape(mask_vol)[0]:
        # if mask case, omit background im
        cs1 = ax1.imshow(np.max(R, axis=1), aspect='auto', cmap=cmap)
        ax1.imshow(1-np.max(mask_vol,axis=1), aspect='auto')
    else:
        # if no mask, show background im
        rIm = np.max(R, axis=1)
        crs = colors.Normalize(0, 1, clip=True)(rIm)
        crs = cmap(crs)
        crs[..., -1] = rIm #setting alpha for transparency
        ax1.imshow(np.max(data_dict['im'],axis=1),cmap=plt.get_cmap('gray'));
        cs1 = ax1.imshow(crs, aspect='auto')

    cs1.set_clim(color_lims[0], color_lims[-1])
    ax1.set_xticks([])
    ax1.set_yticks([])
    # axes[0, 0].set_title('Side')

    if np.shape(mask_vol)[0]:
        # if mask case, omit background im
        cs2 = ax2.imshow(np.max(R, axis=2), aspect='auto', cmap=cmap)
        ax2.imshow(1-np.max(mask_vol,axis=2), aspect='auto')
    else:
        # if no mask, show background im
        rIm = np.max(R, axis=2)
        crs = colors.Normalize(0, 1, clip=True)(rIm)
        crs = cmap(crs)
        crs[..., -1] = rIm #setting alpha for transparency
        ax2.imshow(np.max(data_dict['im'],axis=2),cmap=plt.get_cmap('gray'));
        cs2 = ax2.imshow(crs, aspect='auto')

    cs2.set_clim(color_lims[0], color_lims[-1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # axes[0, 1].set_title('Top')

    ypx_per_um = dims[1]/dims_in_um[1]
    scaleBar_um = 50 #50 um
    # bar_color = 'k' if cmap=='bwr' else 'w'
    bar_color = 'w'
    ax2.plot( dims[1]*.97-(scaleBar_um*ypx_per_um,0), (dims[0]*.93, dims[0]*.93),bar_color)

    if np.shape(mask_vol)[0]:
        # if mask case, omit background im
        cs3 = ax3.imshow( np.max(R, axis=0).T, aspect='auto', cmap=cmap )
        ax3.imshow(1-np.transpose(np.max(mask_vol,axis=0),(1,0,2)), aspect='auto')
    else:
        # if no mask, show background im
        rIm = np.max(R, axis=0).T
        crs = colors.Normalize(0, 1, clip=True)(rIm)
        crs = cmap(crs)
        crs[..., -1] = rIm #setting alpha for transparency
        ax3.imshow(np.max(data_dict['im'],axis=0).T,cmap=plt.get_cmap('gray'));
        cs3 = ax3.imshow(crs, aspect='auto')

    cs3.set_clim(color_lims[0], color_lims[-1])
    ax3.set_xticks([])
    ax3.set_yticks([])
    # axes[1, 1].set_title('Front')


def trim_dynamic_range(data,q_min,q_max):
    bmin = np.quantile(data, q_min)
    bmax = np.quantile(data, q_max)
    data = (data-bmin)/(bmax-bmin)
    data[data<0]=0
    data[data>1]=1
    return data


def show_raster_with_behav(data_dict,color_range=(0,0.4),include_feeding=False,include_dlc=False,num_cells=[],time_lims=[],slice_time=None,title='Raw'):
    if(include_dlc):
        import matplotlib.pylab as pl
        f, axes = plt.subplots(10,1,gridspec_kw={'height_ratios':[12,1,1,1,1,1,1,1,1,2]},figsize=(10.5, 9))
        dlc_colors = pl.cm.jet(np.linspace(0,1,8))
    else:
        f, axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[8,1]},figsize=(10.5, 6))

    if(color_range=='auto'):
        dFF = trim_dynamic_range(data_dict['dFF'], 0.01, 0.95)
        cmin, cmax = (0,1)
    else:
        dFF = data_dict['dFF']
        cmin, cmax = color_range

    tPl = data_dict['tPl']
    behavior = data_dict['ball']
    trial_flag = data_dict['trialFlag']
    U = np.unique(trial_flag)
    NT = len(U)
    if slice_time:
        tPl = tPl[slice_time]
        behavior = behavior[slice_time]
        trial_flag = trial_flag[slice_time]
    if include_dlc:
        dlc = data_dict['dlc']
        if slice_time:
            dlc = dlc[slice_time]
    if include_feeding:
        feed = data_dict['drink']
        stim = data_dict['stim']
        if slice_time:
            feed = feed[slice_time]
            stim = stim[slice_time]


    if num_cells:
        dFF = dFF[:num_cells,:]

    if time_lims:
        dFF = dFF[:,time_lims[0]:time_lims[1]]
        tPl = tPl[time_lims[0]:time_lims[1]]
        behavior = behavior[time_lims[0]:time_lims[1]]
        trial_flag = trial_flag[time_lims[0]:time_lims[1]]
        if include_dlc:
            dlc = dlc[time_lims[0]:time_lims[1],:]
        if include_feeding:
            feed = feed[time_lims[0]:time_lims[1]]
            stim = stim[time_lims[0]:time_lims[1]]
        
    im = axes[0].imshow(
        dFF, aspect='auto', 
        cmap='inferno', vmin=cmin, vmax=cmax)
    plt.sca(axes[0])
    plt.title(title)
    axes[0].set_xlim([0,len(behavior)])
    plt.xticks([])
    plt.ylabel('Neuron')

    if(include_dlc):
        for i in range(8):
            plt.sca(axes[1+i])
            xdataChunk = np.diff(dlc[:,(i-1)*2]); 
            ydataChunk = np.diff(dlc[:,1+(i-1)*2]);
            legEnergy = xdataChunk**2 + ydataChunk**2;
            m = np.quantile(legEnergy, 0.01)
            M = np.quantile(legEnergy, 0.99)
            legEnergy[legEnergy<m]=m
            legEnergy[legEnergy>M]=M
            axes[1+i].plot(tPl[1:],legEnergy,color=dlc_colors[i])
            axes[1+i].set_xlim([min(tPl),max(tPl)])
            axes[1+i].set_xticks([])
            axes[1+i].set_yticks([])
            if (i==3):
                axes[1+i].set_ylabel('DLC point      \n') #extra space here centers label

    

    plt.sca(axes[-1])
    if(include_feeding):
        for i in range(NT):
            is_this_trial = np.squeeze(trial_flag==U[i])
            behav_this_trial = behavior[is_this_trial]
            time_this_trial = tPl[is_this_trial]
            if i==1:
                axes[-1].plot(time_this_trial,behav_this_trial/behav_this_trial.max(),'gray')
            else:
                axes[-1].plot(time_this_trial,behav_this_trial/behav_this_trial.max(),'k')

        axes[-1].set_xlim([min(tPl),max(tPl)])
        axes[-1].plot(tPl,stim,'c',alpha=0.3)
        axes[-1].plot(tPl,feed,'c',alpha=0.7)
        axes[-1].set_ylabel('feeding\nlocomotion')
    else:
        axes[-1].plot(tPl,behavior,'k')
        axes[-1].set_xlim([min(tPl),max(tPl)])
        axes[-1].set_ylabel('ball\n')
    axes[-1].set_yticks([])
    plt.xlabel('Time (s)')

    plt.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.4, 0.03, 0.4])
    f.colorbar(im, cax=cbar_ax, ticks=[cmin,cmin+.5*(cmax-cmin),cmax])
    cbar_ax.set_title(r'$\Delta R/R$')
    return axes

    


def show_example_cells_bestTau(cell_ids, corrMat, tauList, data_dict):
    from matplotlib.gridspec import GridSpec
    scanRate = data_dict['scanRate']
    dFFc = data_dict['dFF'] #trim_dynamic_range(data_dict['dFF'], 0.01, 0.95)
    behavior = data_dict['ball']
    
    #,constrained_layout=True
    fig = plt.figure(figsize=(16,8))
    gs0 = GridSpec(4,len(cell_ids), height_ratios=[3.,1.,1.,1.],hspace=1.2) #,hspace=[.5,.2,.2,.2]
    gs = GridSpec(4,len(cell_ids), height_ratios=[3.,1.,1.,1.],hspace=.05) #,hspace=[.5,.2,.2,.2]

    for j in range(len(cell_ids)):
        cell_id = cell_ids[j]
        axes = fig.add_subplot(gs0[j])
        #plt.subplot(3,len(cell_ids),j+1)
        axes.plot(tauList/scanRate,corrMat[cell_id,:],label='CC')
        axes.set_xscale('log')
        axes.set_xlim((tauList[0]/scanRate,tauList[-1]/scanRate))
        axes.set_xlabel('time constant (s)')
        axes.set_title('cell '+str(j))
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        if(j==0):
            axes.set_ylabel('CC')
            #axes[0,j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        mn = np.argmin(corrMat[cell_id,:])
        mx = np.argmax(corrMat[cell_id,:])

        x = np.array([i for i in range(500)])
        cTau = tauList[mn]
        eFilt = np.exp(-x/cTau)
        c = np.convolve(eFilt,behavior,'valid')#,'same')
        #print(np.corrcoef(dFFc[cell_id,len(eFilt)-1:], c)[0,1])

        axes = fig.add_subplot(gs[len(cell_ids)+j])
        #plt.subplot(3,len(cell_ids),len(cell_ids)+1+j)
        npl = dFFc[cell_id,len(eFilt)-1:]
        axes.plot(-1+(npl-npl.min())/(npl.max()-npl.min()),'b',label='dFF')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        if(j==0):
            axes.set_ylabel('dFF')
            #axes[1,j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        axes = fig.add_subplot(gs[2*len(cell_ids)+j])
        axes.plot((c-c.min())/(c.max()-c.min()),'r',label='beh smoothed argmin')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        if(j==0):
            axes.set_ylabel('Beh\nMIN')
            #axes[2,j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        axes = fig.add_subplot(gs[3*len(cell_ids)+j])
        #plt.subplot(3,len(cell_ids),2*len(cell_ids)+1+j)
        cTau = tauList[mx]
        eFilt = np.exp(-x/cTau)
        c = np.convolve(eFilt,behavior,'valid')#,'same')
        #plt.plot(-1+(npl-npl.min())/(npl.max()-npl.min()),'b',label='dFF')
        axes.plot((c-c.min())/(c.max()-c.min()),'g',label='beh smoothed argmax')
        axes.set_yticks([])
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.set_xlabel('Time (frames)')
        if(j==0):
            axes.set_ylabel('Beh\nMAX')

def show_avg_traces(dict_tot, example_array_tot):
    # plot AVERAGE
    plt.figure(figsize=(14,5))
    # colors_to_use = ['tab:green','tab:orange']
    from matplotlib import colors
    colors_to_use = np.array([colors.to_rgba('tab:orange'),colors.to_rgba('tab:cyan')])
    colors_to_use[:,:3] *= .95   
    av_tr_tot = []
    # for nf in range(3):
    for nf in range(len(dict_tot['data_tot'])):
        plt.subplot(int(len(dict_tot['data_tot'])/2),2,nf+1)
        #plt.subplot(3,1,nf+1)
        data_dict = dict_tot['data_tot'][nf]['data_dict']
        model_fit = dict_tot['model_fit'][nf]
        
        plt.plot( data_dict['behavior']/data_dict['behavior'].max(), 'k')
        av_tr_fly = []
        for k in range(len(example_array_tot)):
            av_tr = np.zeros(len(data_dict['behavior']))
            for i in range(len(example_array_tot[k][nf])):
                j = np.array(example_array_tot[k][nf])[i]
                av_tr += data_dict['rate'][j,:]
            plt.plot( av_tr/av_tr.max(), c=colors_to_use[k,:] )
            av_tr_fly.append( av_tr )
        plt.gca().set_axis_off()
        av_tr_tot.append( av_tr_fly )
    return av_tr_tot

def show_maps_for_avg_traces(dict_tot, example_array_tot):
    ### show map of points used for 'show_avg_traces'
    # plot_filter = copy.deepcopy( model_eval_tot_post )
    import copy
    #color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    plot_filter = [] # this recreates the needed piece of model_eval_tot_post

    idx_tot = []
    color_lims_tot = []
    tmp_dict = {}
    tmp_dict['stat'] = [1,1]
    MAP_tot = dict_tot['MAP_tot']
    data_tot = dict_tot['data_tot']
    vmin= 0 
    vmax= 9 # length of 'Paired' colormap
    colors_to_use = [1,9] #[9,5,1] #[5,1] #[1,4,9]
    # sigLimSec = 100
    # tauList = np.logspace(-1,np.log10(sigLimSec),num=100)
    for nf in range(len(MAP_tot)):
        data_dict = dict_tot['data_tot'][nf]['data_dict']
        map_idx = np.zeros(len(MAP_tot[nf][:,0])) #np.log(tauList[(MAP_tot[nf][:,0]).astype(int)])
        idx_tot.append( map_idx )
        
        color_lims_tot.append( [vmin,vmax] )
        tmp_list = []
        for n in range(data_dict['rate'].shape[0]):
            tmp_list.append( copy.deepcopy(tmp_dict) )
            for k in range(len(example_array_tot)):
                if n in example_array_tot[k][nf]:
                    idx_tot[nf][n] = colors_to_use[k] # use k for color code
                    tmp_list[n]['stat'][1] = 0
        plot_filter.append( tmp_list )

    pval=10**-8 #0.00001
    show_colorCoded_cellMap_points_grid(data_tot, plot_filter, idx_tot, cmap='tab10', color_lims_tot=color_lims_tot, pval=pval, sort_by='z')
            



def show_activity_traces(model_fit, data_dict, plot_param, n_ex, slice_time=None):
    """
    plot_param: param for sorting, in format [['partial_reg', idx]] or  'full_reg'
    n_ex: number of examples.  n_ex>0 gives largest, n_ex<0 gives smallest, n_ex=0 GIVES AVG of everything
    """
    from matplotlib.gridspec import GridSpec
    scanRate = data_dict['scanRate']
    dFF = data_dict['dFF'] #trim_dynamic_range(data_dict['dFF'], 0.01, 0.95)

    f = get_model_fit_as_dict(model_fit)
    if type(plot_param) is list:
        param = f[plot_param[0]][:,plot_param[1]]
        ttl = plot_param[0]+'_'+str(plot_param[1])+' Ex:'+str(n_ex)
    else:
        param = f[plot_param]
        ttl = plot_param+' Ex:'+str(n_ex)
    s = np.argsort(param)[::-1]

    # pdb.set_trace()
    if isinstance(n_ex, list):
        num_ex = n_ex[1]
        start_ex = n_ex[0]
    else:
        num_ex = n_ex
        start_ex = 0
    if num_ex>0:
        traces = dFF[s[start_ex:start_ex+num_ex],:]
        print(s[:num_ex])
    elif num_ex<0:
        if not start_ex:
            traces = dFF[s[start_ex+num_ex:],:]
            print(s[num_ex:])
        else:
            traces = dFF[s[start_ex+num_ex:start_ex],:]
            # print(s[num_ex:])
    else:
        traces = np.expand_dims( dFF.mean(axis=0), axis=0)

    tPl = data_dict['tPl']
    behavior = data_dict['behavior']
    feed = data_dict['drink']
    stim = data_dict['stim']
    if slice_time:
        tPl = tPl[slice_time]
        behavior = behavior[slice_time]
        feed = feed[slice_time]
        stim = stim[slice_time]

    fig = plt.figure(figsize=(14,5))
    gs = GridSpec(2,1, height_ratios=[3.,1.],hspace=.05)

    axes = fig.add_subplot(gs[0])
    axes.plot(tPl, traces.T)
    axes.set_xlim([min(tPl),max(tPl)])
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_title(ttl)
    
    axes = fig.add_subplot(gs[1])
    axes.plot(tPl, behavior, 'k')
    axes.plot(tPl, stim, 'k', alpha=0.3)
    axes.plot(tPl, feed, 'c',alpha=0.7)
    axes.set_xlim([min(tPl),max(tPl)])
    axes.set_ylabel('feeding\nlocomotion')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_yticks([])
    axes.set_xlabel('Time (s)')




def make_hot_without_black(clrs=100, low_bnd=0.15):
    # old low_bnd=0.1
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    hot = cm.get_cmap('hot', clrs)
    newcmp = ListedColormap(hot(np.linspace(low_bnd, .9, clrs)))
    return newcmp

def cold_to_hot_cmap(show_map=False):
    from matplotlib.colors import LinearSegmentedColormap
    basic_cols=['#95d0fc', (.2,.2,.2), '#ff474c'] #   #03012d, #363737
    my_cmap=LinearSegmentedColormap.from_list('mycmap', basic_cols)
    if show_map:
        display_cmap(my_cmap)
        plt.show()
    return my_cmap


def display_cmap(my_cmap):
    plt.figure(figsize=(5,0.25))
    plt.imshow(np.linspace(0, 100, 256)[None, :], interpolation='nearest', cmap=my_cmap, aspect='auto')
    plt.axis('off')
    # plt.show()



