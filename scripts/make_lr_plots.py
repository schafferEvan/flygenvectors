import sys
sys.path.insert(0, '../flygenvectors/')

import os
import numpy as np
from glob import glob
import pickle
import copy
from importlib import reload

import scipy.io as sio
from scipy import sparse, signal
from scipy.stats import zscore, kurtosis, skew
from scipy.optimize import minimize

from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn import manifold

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


import data as dataUtils
# import regression_inv_model as model
# import regression_model as reg_model
import lr_model as model

import plotting
import flygenvectors.ssmutils as utils
import flygenvectors.utils as futils
from sklearn.linear_model import ElasticNet

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")


# LOAD DATA ---------------------------------------------------------------------------
pickle_output = False

# ['2019_06_30','fly1'], ['2019_10_14','fly4'], ['2019_10_18','fly3'], ['2019_10_02','fly2'],
# xx ['2019_10_14','fly2'],
exp_list = [
            ['2018_08_24','fly3_run1'],
            ['2018_08_24','fly2_run2'],
            ['2019_07_01','fly2'],
            ['2019_10_14','fly3'],
            ['2019_06_28','fly2'],
            ['2019_10_21','fly1'],
            ['2019_10_10','fly3'],
            ['2019_08_14','fly1']]

for i in range(len(exp_list)):
    exp_date = exp_list[i][0]
    fly_num = exp_list[i][1]
    # exp_date = '2019_07_01' #'2018_08_24' 
    # fly_num = 'fly2' #'fly3_run1'

    expt_id = exp_date+'_'+fly_num
    dirs = futils.get_dirs()
    fig_dirs = futils.get_fig_dirs(expt_id)
    data_dict_init = dataUtils.load_timeseries_simple(exp_date,fly_num,dirs['data'])
    # pdb.set_trace()
    ro = model.lr_obj(exp_id=expt_id, 
                       data_dict=data_dict_init,
                       fig_dirs=fig_dirs,
                       split_behav=False,
                       iters=150, 
                       n_components=100, 
                       use_p2=False)
    print('Loading data from '+expt_id)
    ro.data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )

    # load model
    print('Loading model from '+expt_id)
    if ro.use_p2:
        pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'_wP2_PCreg_model.pkl'
    else:
        pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'PCreg_model.pkl'
    ro.model_fit = pickle.load( open( pkl_name, "rb" ) )

    # get drift residual
    dFF_drift_res = ro.data_dict['dFF'].copy()
    a1 = ro.model_fit['a1']
    t = np.arange(0,dFF_drift_res.shape[1],1)/dFF_drift_res.shape[1]
    atime = np.array([np.ones(t.shape), t, t**2])
    for i in range(dFF_drift_res.shape[0]):
        P1i = a1[i,:]@atime
        dFF_drift_res[i,:] -= P1i


    ## get residual ALSO subtracting dominant mode
    dom_mode = np.expand_dims(ro.model_fit['U'][:,0],axis=1) @ np.expand_dims(ro.model_fit['V'][:,0],axis=0)
    dFF_dom_res = dFF_drift_res - dom_mode


    ## sort residual by similarity to previous row
    plot_order = np.zeros(dFF_dom_res.shape[0])
    accounted_for = np.zeros(dFF_dom_res.shape[0])
    last_idx = 0
    accounted_for[0] = 1
    ctr = 0
    while accounted_for.sum()<len(accounted_for):
        if not np.mod(accounted_for.sum(),50): print(int(accounted_for.sum()),end=' ')
        # find the next partner
        argmn = None
        mn = np.inf
        for i in range(len(plot_order)):
            if not accounted_for[i]:
                v = (dFF_dom_res[last_idx,:]-dFF_dom_res[i,:]).var()
                if v<mn:
                    argmn=i
                    mn=v
        
        accounted_for[argmn] = 1
        last_idx=argmn
        ctr += 1
        plot_order[ctr]=argmn

    ## make sample plot of sorted residual
    print('Plot Sample Residual')
    data_dict_res = copy.deepcopy(ro.data_dict)
    data_dict_res['dFF'] = dFF_dom_res[plot_order.astype(int)[150:250],:]
    ax = plotting.show_raster_with_behav(data_dict_res,color_range=(-0.05,0.05),include_feeding=False,include_dlc=False)
    ax[0].set_title('Residual')
    plt.savefig(fig_dirs['pcfig_folder'] + expt_id +'_residual_examples.pdf', bbox_inches='tight')
    # plt.show()


    # ## show properly scaled maps for original (unaligned) model
    # print('Show Properly Scaled Spatial Maps')
    # my_cmap = plotting.cold_to_hot_cmap()
    # for nc in range(ro.n_components):
    #     idx = ro.model_fit['U'][:,nc]
    #     plotting.show_colorCoded_cellMap_points(ro.data_dict, idx, idx, cmap=my_cmap, pval=0.01, color_lims_scale=[-0.9,0.9])
    #     plt.savefig(fig_dirs['pcfig_folder'] + expt_id +'_PCmap'+str(nc)+'.pdf', bbox_inches='tight')
    #     # plt.show()


    # ## show maps next to timeseries (U and V)
    # print('Show Spatial Maps with Timeseries (U and V)')
    # cmap = plotting.cold_to_hot_cmap() #plotting.make_hot_without_black()
    # gry = cm.get_cmap('Greys', 15)
    # gry = ListedColormap(gry(np.linspace(.8, 1, 2)))
    # tab_list = 10*('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    #             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')   

    # for nc in range(ro.n_components):
    #     idx = ro.model_fit['U'][:,nc]
    #     f, (a0, a1) = plt.subplots(1, 2, figsize=(18,4), gridspec_kw={'width_ratios': [1, 2.75]})
    #     a0.scatter(ro.data_dict['aligned_centroids'][:,0],
    #                 ro.data_dict['aligned_centroids'][:,1], c=idx, cmap=cmap, s=2,
    #                vmin=.5*np.min( (idx.min(),-idx.max()) ), vmax=.5*np.max( (-idx.min(),idx.max()) ) )
    #     a0.set_facecolor((0.0, 0.0, 0.0))
    #     a0.set_xticks([])
    #     a0.set_yticks([])
    #     a0.invert_yaxis()
    #     a1.plot(ro.data_dict['tPl'], ro.data_dict['behavior']/ro.data_dict['behavior'].max(),'k', label='running vigor')
    #     a1.plot(ro.data_dict['tPl'], ro.model_fit['V'][:,nc]/ro.model_fit['V'][:,nc].max(), c=tab_list[nc],label='mode '+str(nc))
    #     a1.set_xlabel('Time (s)')
    #     a1.legend()
    #     plt.savefig(fig_dirs['pcfig_folder'] + expt_id +'_PC'+str(nc)+'.pdf', bbox_inches='tight')
    #     # plt.show()


    # ## compare mode 0 to smoothed behavior
    # # convolve behavior with chosen kernel, either piecewise or all together
    # ts = ro.data_dict['time']-ro.data_dict['time'].mean()
    # ts /= abs(ts).max()
    # sigLimSec = 30 
    # tau = 5
    # sigLim = sigLimSec*ro.data_dict['scanRate']
    # M = np.round(-sigLim*np.log(0.1)).astype(int)
            
    # t_exp = np.linspace(1,M,M)/ro.data_dict['scanRate']
    # kern = np.zeros(len(t_exp))
    # p = t_exp>=0
    # # n = t_exp<params['mu']
    # kern[p] = np.exp(-(t_exp[p])/tau)
    # # kern[n] = np.exp((t_exp[n]-params['mu'])/params['tau'])

    # kern /= kern.sum()
    # ball = ro.data_dict['behavior']-ro.data_dict['behavior'].mean()
    # x_c_full = np.convolve(kern,ball,'full')   
    
    # nc=0
    # idx = ro.model_fit['U'][:,nc]
    # f, (a0, a1) = plt.subplots(1, 2, figsize=(18,4), gridspec_kw={'width_ratios': [1, 2.75]})
    # a0.scatter(ro.data_dict['aligned_centroids'][:,0],
    #             ro.data_dict['aligned_centroids'][:,1], c=idx, cmap=cmap, s=2,
    #            vmin=.5*np.min( (idx.min(),-idx.max()) ), vmax=.5*np.max( (-idx.min(),idx.max()) ) )
    # a0.set_facecolor((0.0, 0.0, 0.0))
    # a0.set_xticks([])
    # a0.set_yticks([])
    # a0.invert_yaxis()


    # beh_plot = x_c_full[:-M+1]
    # beh_plot = (beh_plot-beh_plot.min())/(beh_plot.max()-beh_plot.min())
    # mode_plot = ro.model_fit['V'][:,nc]/ro.model_fit['V'][:,nc].max()
    # mode_plot = (mode_plot-mode_plot.min())/(mode_plot.max()-mode_plot.min())
    # mode_plot *= .75

    # a1.plot(ro.data_dict['tPl'], beh_plot,'k', label=r'Smoothed Running ($\tau=5s$)')
    # a1.plot(ro.data_dict['tPl'], mode_plot, c=tab_list[nc],label='Component '+str(nc+1))
    # a1.set_xlabel('Time (s)')
    # a1.legend()
    # plt.savefig(fig_dirs['pcfig_folder'] + expt_id +'_PC_0_vs_smooth_run.pdf', bbox_inches='tight')
    # # plt.show()

    # examine sparseness
    participation = np.zeros(ro.model_fit['U'].shape[1])
    for nc in range(ro.model_fit['U'].shape[1]):
        v = ro.model_fit['U'][:,nc]
        participation[nc] = (  sum(v**2)**2/sum(v**4)  )/len(v)
    plt.figure(figsize=(5,5))
    plt.plot([1,41],[.333,.333],'--',color='#7d7f7c')
    plt.plot([i for i in range(1,41)],participation[:40],'.-',markersize=12,color='#26538d')
    # plt.errorbar(x, m, yerr=s,color='#26538d',ecolor='#95d0fc',capsize=1)
    plt.ylabel('Participation Ratio')
    plt.xlabel('Mode')
    plt.ylim(0,plt.ylim()[1])
    plt.xlim(0,40)
    plt.xticks([10,20,30,40])
    # plt.yticks([0.2,0.3,0.4,0.5,0.6])
    plt.tight_layout()
    plt.savefig(fig_dirs['pcfig_folder'] + expt_id + '_participation_ratio.pdf', bbox_inches='tight')
    # plt.show()


    try:
        # PCA after detrending
        # split into train/test trials
        data_neural = dFF_drift_res.copy().T #data_dict['dFF'].copy().T
        # data_neural = dict_temp['dFF'].copy().T

        ro.reg_obj.get_train_test_data()
        n_components = 40
        ll_train = np.full(n_components, fill_value=np.nan)
        ll_val = np.full(n_components, fill_value=np.nan)
        for n in range(n_components):
            pca = PCA(n_components=n+1)
            pca.fit(ro.reg_obj.data_dict['train_all'])
            ll_train[n] = pca.score(ro.reg_obj.data_dict['train_all'])
            ll_val[n] = pca.score(ro.reg_obj.data_dict['val_all'])

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(n_components)+1, np.cumsum(pca.explained_variance_ratio_),'k.')
        plt.xlabel('Component')
        plt.ylabel('Explained variance')
        plt.ylim(0,1)

        plt.subplot(122)
        plt.plot(np.arange(n_components)+1, ll_val,'k.', label='val')
        plt.xlabel('Component')
        plt.ylabel('Average log-likelihood')
        plt.tight_layout()
        plt.savefig(fig_dirs['pcfig_folder'] + expt_id +'_PC_loglikelihood.pdf',transparent=False, bbox_inches='tight')

    except Exception as e:
        print(e)
        print('\r ***** FAILED *****: '+expt_id+'\r\r\r')


