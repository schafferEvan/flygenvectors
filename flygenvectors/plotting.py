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
