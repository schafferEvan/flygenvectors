
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

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import data as dataUtils
import regression_model as model
import plotting
import flygenvectors.ssmutils as utils

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")







# LOAD DATA ---------------------------------------------------------------------------
# main_dir = '/Users/evan/Dropbox/_sandbox/__flygenvectors_tmp/'
# main_fig_dir = '/Users/evan/Dropbox/_sandbox/__flygenvectors_tmp/'
main_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/dataShare/_main/'
main_fig_dir = '/Users/evan/Dropbox/_AxelLab/__flygenvectors/figs/'
# main_dir = '/Volumes/data1/_flygenvectors_dataShare/_main/_sparseLines/'
# main_fig_dir = '/Volumes/data1/figsAndMovies/figures/'

exp_date = '2019_10_14' #'2019_10_02' #'2019_07_01' #'2018_08_24' 
fly_num = 'fly3' #'fly2' #'fly3_run1'
remake_pickle = False
pval = 0.01

data_dict = dataUtils.load_timeseries_simple(exp_date,fly_num,main_dir)
data_dict_template = dataUtils.load_timeseries_simple('2018_08_24','fly2_run2',main_dir)
data_dict['template_dims_in_um'] = data_dict_template['dims_in_um']
data_dict['template_dims'] = data_dict_template['dims']
expt_id = exp_date+'_'+fly_num
fig_folder = main_fig_dir+expt_id+'/'
clustfig_folder = fig_folder+'clusters/'
if not os.path.exists(fig_folder):
    os.mkdir(main_fig_dir+expt_id)
    os.mkdir(fig_folder)
    os.mkdir(clustfig_folder)
    



# # BINARIZE BEHAVIOR --------------------------------------------------------------------
# gmm = GaussianMixture(n_components=2, means_init=[[0.5],[0.8]])
# log_beh = np.log(data_dict['ball'])-np.log(data_dict['ball']).min()
# log_beh = log_beh/log_beh.max()
# gmm_beh_fit = gmm.fit_predict(log_beh) #np.expand_dims(data_dict['ball'],axis=1))
# mean_0 = log_beh[gmm_beh_fit==0].mean()
# mean_1 = log_beh[gmm_beh_fit==1].mean()
# if (mean_0<mean_1):
#     beh = gmm_beh_fit
# else:
#     beh = np.logical_not(gmm_beh_fit)
# data_dict['ball'] = beh.copy()
data_dict['ball'] = data_dict['ball'].flatten()


# VISUALIZE RAW DATA --------------------------------------------------------------------
# Note: optional second arg of show_raster_with_behav is color range,
# accepts (min,max) tuple, defaults to (0,0.4), also accepts 'auto'
dt = data_dict['time'][1]-data_dict['time'][0]
tPl = data_dict['time'][0]+np.linspace(0,dt*len(data_dict['time']),len(data_dict['time']))
data_dict['tPl'] = tPl
plotting.show_raster_with_behav(data_dict,color_range=(0,0.2),include_feeding=False,include_dlc=True)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_raster.pdf',transparent=True, bbox_inches='tight')




# ANALYSIS OF TIME CONSTANTS RELATING DFF TO BEHAVIOR ------------------------------------
# find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
# tauList, corrMat = dataUtils.estimate_neuron_behav_tau(data_dict)
if remake_pickle:
    model_fit = model.estimate_neuron_behav_reg_model(data_dict)
    pickle.dump( model_fit, open( fig_folder + exp_date + '_' + fly_num +'_reg_model.pkl', "wb" ) )
else:
    model_fit = pickle.load( open( fig_folder + exp_date + '_' + fly_num +'_reg_model.pkl', "rb" ) )
tauList = np.logspace(np.log10(data_dict['scanRate']),np.log10(100*data_dict['scanRate']),num=200)  
plotting.show_tau_scatter(model_fit)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_tauScatter.pdf',transparent=True, bbox_inches='tight')



# SHOW MODEL RESIDUAL -------------------------------------------------------
plotting.show_residual_raster(data_dict, model_fit)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_residual.pdf',transparent=True,bbox_inches='tight')

# # SHOW older PCA MODEL RESIDUAL -------------------------------------------------------
# plotting.show_PC_residual_raster(data_dict)
# plt.savefig(fig_folder + exp_date + '_' + fly_num +'_PCresidual.pdf',transparent=True,bbox_inches='tight')



# MAKE BRAIN VOLUME WITH CELLS COLOR CODED BY TAU -----------------------------------------
f = plotting.get_model_fit_as_dict(model_fit)
f['sig'] = (f['stat']<pval)*(f['stat']>0)
tau_argmax = np.zeros(f['tau'].shape)
for i in range(len(tau_argmax)):
    tau_argmax[i] = np.argmin(np.abs(f['tau'][i]-tauList/data_dict['scanRate']))
tau_argmax = tau_argmax.astype(int)
tau_argmax[np.logical_not(f['success']*f['sig'])] = 0
clrs = len(tauList)
R, mask_vol = plotting.make_colorCoded_cellMap(tau_argmax, clrs, data_dict)

# MAKE COLORBAR FOR BRAIN VOLUME WITH CELLS COLOR CODED BY TAU
fullList=tauList/data_dict['scanRate']
tau_label_list=[1,3,10,30,100]
tau_loc_list = plotting.make_labels_for_colorbar(tau_label_list, fullList)
plotting.make_colorBar_for_colorCoded_cellMap(R, mask_vol, tauList, tau_label_list, tau_loc_list, data_dict)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_tauMap_colorbar.pdf',transparent=True, bbox_inches='tight')

# SHOW BRAIN VOLUME WITH CELLS COLOR CODED BY TAU
# in correct spatial dims (bar=50um), color matches above (in secs)
plotting.show_colorCoded_cellMap(R, mask_vol, tau_loc_list[[0,-1]], data_dict)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_tauMap.pdf',transparent=True, bbox_inches='tight')

# # show location of 'slow' cells, color coded by CC
# cbounds = (-1, 1) # corr from -1 to 1
# tau_valmax = np.max(corrMat,axis=1) # color by max cc instead of argmax as above
# S, mask_vol = plotting.make_colorCoded_cellMap(tau_valmax, clrs, data_dict, cbounds)

# # make COLORBAR for 'brain volume with cells color coded by CC
# cmap = 'bwr'
# ccList=np.linspace(-1.0, 1.0, num=len(tauList))
# label_list=[-1,-0.5,0,0.5,1]
# cc_loc_list = plotting.make_labels_for_colorbar(label_list, ccList)
# plotting.make_colorBar_for_colorCoded_cellMap(S, mask_vol, ccList, label_list, cc_loc_list, data_dict, cmap)
# plt.savefig(fig_folder + exp_date + '_' + fly_num +'_ccMap_colorbar.pdf',transparent=True, bbox_inches='tight')
# plt.show()

# # show brain volume with cells color coded by max CC
# # in correct spatial dims (bar=50um), color from CC=-1 to CC=1
# plotting.show_colorCoded_cellMap(S, mask_vol, tau_loc_list[[0,-1]], data_dict, cmap)
# plt.savefig(fig_folder + exp_date + '_' + fly_num +'_ccMap.pdf',transparent=True, bbox_inches='tight')
# plt.show()


# MAKE BRAIN VOLUME WITH CELLS *as points* COLOR CODED BY TAU ---------------------------------
plotting.show_colorCoded_cellMap_points(data_dict, model_fit, tau_argmax)
plt.savefig(fig_folder + exp_date + '_' + fly_num +'_tauMap_pts.pdf',bbox_inches='tight')



