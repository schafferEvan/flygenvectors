
import sys
sys.path.insert(0, '../flygenvectors/')

import os
import numpy as np
from glob import glob
import pickle

import scipy.io as sio
from scipy import sparse, signal
from scipy.stats import zscore

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from skimage.restoration import denoise_tv_chambolle

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import data as dataUtils
import regression_model as model
import plotting
# import flygenvectors.ssmutils as utils
import copy
import pdb
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")



def run_all(input_dict=None):
    if not input_dict:
        main_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/dataShare/_main/' #'/Volumes/data1/_flygenvectors_dataShare/_main/_sparseLines/'
        main_fig_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/figs/' #'/Volumes/data1/figsAndMovies/figures/'
        exp_date = '2019_07_01' #'2018_08_24' 
        fly_num = 'fly2' #'fly3_run1'
        remake_pickle = False   # rerun regression
        activity = 'dFF'        # metric of neural activity {'dFF', 'rate'}, the latter requires deconvolution
        split_behav = False     # treat behavior from each trial as separate regressor
        elasticNet = False      # run regression with elastic net regularization (alternative is OLS)
    else:
        main_dir = input_dict['main_dir']
        main_fig_dir = input_dict['main_fig_dir']
        exp_date = input_dict['exp_date']
        fly_num = input_dict['fly_num']
        remake_pickle = input_dict['remake_pickle']
        activity = input_dict['activity']
        split_behav = input_dict['split_behav']
        elasticNet = input_dict['elasticNet']


    # LOAD DATA ---------------------------------------------------------------------------
    pval = 0.01
    data_dict = dataUtils.load_timeseries_simple(exp_date,fly_num,main_dir)
    data_dict_template = dataUtils.load_timeseries_simple('2018_08_24','fly2_run2',main_dir)
    data_dict['template_dims_in_um'] = data_dict_template['dims_in_um']
    data_dict['template_dims'] = data_dict_template['dims']
    expt_id = exp_date+'_'+fly_num
    fig_folder = main_fig_dir+expt_id+'/'
    regfig_folder = fig_folder+'regmodel/'
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    if not os.path.exists(regfig_folder):
        os.mkdir(regfig_folder)
        

    # CHOOSE BEHAVIOR TIMESERIES WITH WHICH TO DO ANALYSES -----------------------------------
    behavior_source = 'ball_raw' #{dlc_med, ball_raw, ball_binary}

    if (behavior_source == 'dlc_med'):
        dlc_energy = dataUtils.get_dlc_motion_energy(data_dict)
        data_dict['behavior'] = np.median(dlc_energy,axis=1)
        data_dict['behavior'] = np.concatenate((data_dict['behavior'],[data_dict['behavior'][-1]]))
        # data_dict['time'] = data_dict['time'][:-1]
        # data_dict['ball'] = data_dict['ball'][:-1]
        # data_dict['dFF'] = data_dict['dFF'][:,:-1]
    elif (behavior_source == 'ball_raw'):
        data_dict['behavior'] = data_dict['ball'].flatten()
    elif (behavior_source == 'ball_binary'):
        data_dict['behavior'] = dataUtils.binarize_timeseries(data_dict['ball'])


    # VISUALIZE RAW DATA --------------------------------------------------------------------
    # Note: optional second arg of show_raster_with_behav is color range,
    # accepts (min,max) tuple, defaults to (0,0.4), also accepts 'auto'
    dt = data_dict['time'][1]-data_dict['time'][0]
    tPl = data_dict['time'][0]+np.linspace(0,dt*len(data_dict['time']),len(data_dict['time']))
    data_dict['tPl'] = tPl
    if (exp_date == '2018_08_24'):
        plotting.show_raster_with_behav(data_dict,color_range=(0,0.3),include_feeding=False,include_dlc=False)
    else:
        plotting.show_raster_with_behav(data_dict,color_range=(0,0.3),include_feeding=False,include_dlc=True)
    plt.savefig(fig_folder + exp_date + '_' + fly_num +'_raster.pdf',transparent=True, bbox_inches='tight')




    # ANALYSIS OF TIME CONSTANTS RELATING DFF TO BEHAVIOR ------------------------------------
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    ro = model.reg_obj(activity=activity)
    ro.data_dict = data_dict
    ro.params['split_behav'] = split_behav
    ro.elasticNet = elasticNet
    if remake_pickle:
        ro.fit_and_eval_reg_model_extended()
        #model_fit = model.estimate_neuron_behav_reg_model_taulist_shiftlist_gauss(data_dict_decon)
        pickle.dump( ro.model_fit, open( fig_folder + exp_date + '_' + fly_num +'_'+ro.activity+'_gauss_ols_reg_model.pkl', "wb" ) )
    else:
        ro.model_fit = pickle.load( open( fig_folder + exp_date + '_' + fly_num +'_'+ro.activity+'_gauss_ols_reg_model.pkl', "rb" ) )
    f = plotting.get_model_fit_as_dict(ro.model_fit)


    # PLOT TAU SCATTER ---------------------------------------------------------------------------
    plotting.show_tau_scatter(ro.model_fit)
    plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_'+ro.activity+'_gauss_ols_tauScatter.pdf',transparent=True, bbox_inches='tight')

    # PLOT PHI & BETA SCATTER --------------------------------------------------------------------
    param_list=['beta_0','phi']
    for i in range(len(param_list)):
        param = param_list[i]
        plotting.show_param_scatter(ro.model_fit, ro.data_dict, param)
        plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_'+ro.activity+'_gauss_ols_'+param+'Scatter.pdf',transparent=True, bbox_inches='tight')

    # # SHOW MODEL RESIDUAL -------------------------------------------------------
    # plotting.show_residual_raster(data_dict, model_fit, exp_date)
    # plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_residual.pdf',transparent=True,bbox_inches='tight')

    # # SHOW older PCA MODEL RESIDUAL -------------------------------------------------------
    # plotting.show_PC_residual_raster(data_dict)
    # plt.savefig(fig_folder + exp_date + '_' + fly_num +'_PCresidual.pdf',transparent=True,bbox_inches='tight')


    # VISUALIZE RAW DATA *SORTED* by PHI & TAU --------------------------------------------------------------------
    param_list=['tau','phi']
    for i in range(len(param_list)):
        param = param_list[i]
        data_dict_sorted = copy.deepcopy(data_dict)
        tau_arg_order = np.argsort(f[param])
        data_dict_sorted['dFF'] = data_dict_sorted['dFF'][ tau_arg_order ,:]
        plotting.show_raster_with_behav(data_dict_sorted,color_range=(0,0.3),include_feeding=False,include_dlc=False)
        plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_'+ro.activity+'_gauss_ols_raster_'+param+'sorted.pdf',transparent=True, bbox_inches='tight')


    # MAKE BRAIN VOLUME WITH CELLS COLOR CODED BY each parameter -----------------------------------------
    param = 'tau'
    plotting.show_colorCoded_cellMap_points(ro.data_dict, ro.model_fit, param, cmap=plotting.make_hot_without_black(), color_lims_scale=[0.05,0.95]) 
    plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_'+param+'_map.pdf',transparent=False, bbox_inches='tight')

    param_list=['beta_0','phi']
    for i in range(len(param_list)):
        param = param_list[i]
        plotting.show_colorCoded_cellMap_points(ro.data_dict, ro.model_fit, param, cmap=plotting.cold_to_hot_cmap(show_map=False)) #, pval=0.01, color_lims=[vmin,vmax])
        plt.savefig(regfig_folder + exp_date + '_' + fly_num +'_'+param+'_map.pdf',transparent=False, bbox_inches='tight')

        # # MAKE COLORBAR FOR BRAIN VOLUME WITH CELLS COLOR CODED BY TAU
        # fullList=tauList/data_dict['scanRate']
        # tau_label_list=[1,3,10,30,100]
        # tau_loc_list = plotting.make_labels_for_colorbar(tau_label_list, fullList)
        # plotting.make_colorBar_for_colorCoded_cellMap(R, mask_vol, tauList, tau_label_list, tau_loc_list, data_dict)
        # plt.savefig(fig_folder + exp_date + '_' + fly_num +'_tauMap_colorbar.pdf',transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    m = run_all()

