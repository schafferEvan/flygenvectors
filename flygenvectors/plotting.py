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


def get_model_fit_as_dict(model_fit):
    fit_dict = {}
    tau = np.zeros(len(model_fit))
    phi = np.zeros(len(model_fit))
    beta_0 = np.zeros(len(model_fit))
    if isinstance(model_fit[0]['beta_1'],np.ndarray):
        beta_1 = np.zeros((len(model_fit),len(model_fit[0]['beta_1'])))
    else:
        beta_1 = np.zeros((len(model_fit),1))
    rsq = np.zeros(len(model_fit))
    rsq_null = np.zeros(len(model_fit))
    stat = np.zeros( (len(model_fit), np.shape(model_fit[0]['stat'])[0] ) )
    success = []
    for i in range(len(model_fit)):
        tau[i] = model_fit[i]['tau']
        phi[i] = model_fit[i]['phi']
        beta_0[i] = model_fit[i]['beta_0']
        beta_1[i,:] = model_fit[i]['beta_1']
        rsq[i] = model_fit[i]['r_sq']
        # rsq_null[i] = model_fit[i]['r_sq_null']
        for j in range(np.shape(model_fit[0]['stat'])[0]):
            stat[i,j] = model_fit[i]['stat'][j][1]
        success.append(model_fit[i]['success'])
    fit_dict['tau'] = tau
    fit_dict['phi'] = phi
    fit_dict['beta_0'] = beta_0
    fit_dict['beta_1'] = beta_1
    fit_dict['rsq'] = rsq
    # fit_dict['rsq_null'] = rsq_null
    fit_dict['stat'] = stat
    # fit_dict['success'] = success
    return fit_dict


def show_param_scatter(model_fit, data_dict, param_name, pval=.01):
    f = get_model_fit_as_dict(model_fit)
    param = f[param_name]
    rsq = f['rsq']
    rsq_null = f['rsq_null']
    stat = f['stat']
    success = f['success']

    if(param_name=='phi'): 
        param /= data_dict['scanRate'] 
        label = 'Phase (s)'
    elif(param_name=='beta_0'): 
        label = r'$\beta_0$'

    pval_text = pval #0.01
    sig = (stat<pval)*(rsq>rsq_null) #*(stat>0) # 1-sided test that behavior model > null model
    # sig_text = (stat<pval_text)*(rsq>rsq_null) # 1-sided test that behavior model > null model
    param_sig = param[success*sig]
    param_notsig = param[np.logical_not(success*sig)]
    rsq_sig = rsq[success*sig]
    rsq_notsig = rsq[np.logical_not(success*sig)]
    # tau_is_pos_sig = tau_is_pos[success*sig]
    # tau_is_neg_sig = np.logical_not(tau_is_pos_sig)
    
    # print('median tau of significant cells = '+str(np.median(tau_sig)))
    # print('frac of significant cells w/ tau above 60s = '+str( np.sum(tau_sig>60)/len(tau_sig) ))
    # print('fraction of cells with p<'+str(pval)+' = '+str( (success*sig).sum()/len(success) ))

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
    rsq = f['rsq']
    rsq_null = f['rsq_null']
    stat = f['stat']
    success = f['success']

    pval_text = pval #0.01
    sig = (stat<pval)*(rsq>rsq_null) #*(stat>0) # 1-sided test that behavior model > null model
    # sig_text = (stat<pval_text)*(rsq>rsq_null) # 1-sided test that behavior model > null model
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
    print('frac of significant cells w/ tau above 60s = '+str( np.sum(tau_sig>60)/len(tau_sig) ))
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
    xmax = 100 #1000/scanRate #6000
    ymin = 0
    ymax = 1

    ax_scatter.scatter(tau_notsig,rsq_notsig,c='tab:gray',marker='.',alpha=0.3)
    ax_scatter.scatter(tau_sig[tau_is_pos_sig],rsq_sig[tau_is_pos_sig],c='tab:blue',marker='.',alpha=0.3) #'#1f77b4'
    ax_scatter.scatter(tau_sig[tau_is_neg_sig],rsq_sig[tau_is_neg_sig],c='tab:red',marker='.',alpha=0.3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_xlim((xmin,xmax))
    ax_scatter.set_ylim((ymin,ymax))
    ax_scatter.set_xlabel('Optimal time constant (s)')
    ax_scatter.set_ylabel(r'$r^2$')

    # xbinwidth = 100
    ybinwidth = 0.025
    ybins = np.arange(ymin, ymax+ybinwidth, ybinwidth)
    xbins = np.logspace(np.log(xmin), np.log(xmax),num=len(ybins),base=np.exp(1))
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
                sig.append( model_fit[i]['stat'][plot_field][1]<pval ) # 1-sided test that behavior model > null model
            else:
                color_data[i] = model_fit[i][plot_field][plot_field_idx]
                sig.append( model_fit[i]['stat'][plot_field][plot_field_idx][1]<pval ) # 1-sided test that behavior model > null model
                
        # sig cleanup
        not_sig = np.logical_not(sig)
        sig = np.flatnonzero(sig) #.tolist()
        not_sig = np.flatnonzero(not_sig) #.tolist()
    else:
        # this needs to be updated.  the above takes the 'list' case
        # if not dict, assumed to be list. This is for manual curation and testing
        sig = np.array([i for i in range(len(model_fit))]) #np.ones(len(model_fit))
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



def show_colorCoded_cellMap_points_grid(data_dict_tot, model_fit_tot, plot_param, cmap, color_lims_scale, pval=0.01, sort_by=[]):
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

    if type(plot_param) is list:
        plot_field = plot_param[0]
        plot_field_idx = plot_param[1]
    else:
        plot_field = plot_param
        plot_field_idx = np.nan
        
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

        # generate color_data from model_fit
        color_data = np.zeros(len(model_fit))
        sig = [] # significance threshold
        for i in range(len(model_fit)):
            if np.isnan(plot_field_idx):
                color_data[i] = model_fit[i][plot_field]
                sig.append( model_fit[i]['stat'][plot_field][1]<pval ) # 1-sided test that behavior model > null model
            else:
                color_data[i] = model_fit[i][plot_field][plot_field_idx]
                sig.append( model_fit[i]['stat'][plot_field][plot_field_idx][1]<pval ) # 1-sided test that behavior model > null model
                
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


def show_raster_with_behav(data_dict,color_range=(0,0.4),include_feeding=False,include_dlc=False,num_cells=[],time_lims=[]):
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
    if include_dlc:
        dlc = data_dict['dlc']
    if include_feeding:
        feed = data_dict['drink']

    if num_cells:
        dFF = dFF[:num_cells,:]

    if time_lims:
        dFF = dFF[:,time_lims[0]:time_lims[1]]
        tPl = tPl[time_lims[0]:time_lims[1]]
        behavior = behavior[time_lims[0]:time_lims[1]]
        if include_dlc:
            dlc = dlc[time_lims[0]:time_lims[1],:]
        if include_feeding:
            feed = feed[time_lims[0]:time_lims[1]]
        
    im = axes[0].imshow(
        dFF, aspect='auto', 
        cmap='inferno', vmin=cmin, vmax=cmax)
    plt.sca(axes[0])
    plt.title('Raw')
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
    axes[-1].plot(tPl,behavior,'k')
    axes[-1].set_xlim([min(tPl),max(tPl)])
    if(include_feeding):
        axes[-1].plot(tPl,feed*max(behavior),'c')
        axes[-1].set_ylabel('feeding\nlocomotion')
    else:
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
            


def display_cmap(cmap):
    plt.imshow(np.linspace(0, 100, 256)[None, :],  aspect=25,    interpolation='nearest', cmap=cmap) 
    plt.axis('off')

def make_custom_cold_to_hot():
    from matplotlib.colors import LinearSegmentedColormap
    basic_cols=['#ff474c', (.2,.2,.2), '#95d0fc'] #   #03012d, #363737
    my_cmap=LinearSegmentedColormap.from_list('mycmap', basic_cols)
    return my_cmap


def make_hot_without_black(clrs=100, low_bnd=0.15):
    # old low_bnd=0.1
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    hot = cm.get_cmap('hot', clrs)
    newcmp = ListedColormap(hot(np.linspace(low_bnd, .9, clrs)))
    return newcmp

