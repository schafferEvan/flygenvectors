import sys
sys.path.insert(0, '../flygenvectors/')

import os
import numpy as np
from glob import glob
import pickle
import copy
from importlib import reload
import pdb

import scipy.io as sio
from scipy import sparse, signal
from scipy.stats import zscore

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
#from skimage.restoration import denoise_tv_chambolle

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import data as dataUtils
import regression_model as model
import plotting
import flygenvectors.ssmutils_moto as utils
import flygenvectors.utils as futils


# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

remake_pickle = True
activity='dFF'


# LOAD DATA ---------------------------------------------------------------------------
reload(model)
reload(futils)
reload(dataUtils)
pickle_output = False

exp_list = ['2019_07_01_fly2',
            '2019_06_28_fly2',
            '2019_10_14_fly3',
            '2018_08_24_fly3_run1',
            '2018_08_24_fly2_run2',
            '2019_10_21_fly1',
            '2019_10_10_fly3',
            '2019_08_14_fly1',
            '2019_06_30_fly1',
            '2019_10_14_fly2',
            '2019_10_14_fly4',
            '2019_10_02_fly2',
            '2019_08_14_fly3_2',
            '2019_08_14_fly2',
            '2019_08_07_fly2',
            '2019_10_18_fly3',
            '2019_06_26_fly2',
            '2019_08_19_fly1_1'] 

i = int(sys.argv[1]) # DATASET TO LOAD
part_i = int(sys.argv[2])
n_perms = int(sys.argv[3]) # DATASET TO LOAD
# for expt_id in exp_list:
expt_id = exp_list[i] #[0] + '_' + exp_list[i][1]
print(expt_id)

dirs = futils.get_dirs()
fig_dirs = futils.get_fig_dirs(expt_id)
data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )

if (expt_id=='2018_08_24_fly3_run1') or (expt_id=='2018_08_24_fly2_run2'):
     data_dict['beh_labels'] = np.expand_dims( data_dict['behavior'], axis=1).copy()
     data_dict['beh_labels'] = 1*(data_dict['beh_labels']>.0005)
#    ro = model.reg_obj(activity=activity, 
#                        data_dict=copy.deepcopy(data_dict),
#                        exp_id=expt_id)
#    ro.get_smooth_behavior()
#    data_dict['behavior'] = ro.data_dict['behavior']

# crop time (crude bootstrapping)
#if part_i==0:
#    part='beg'
#elif part_i==1:
#    part='mid'
#elif part_i==2:
#    part='end'
#else:
#    print(part)
# for part in ['beg', 'mid', 'end']:
# for part in ['end']:
dict_crop = copy.deepcopy(data_dict)
#l = len(data_dict['behavior'])
#b = round(.05*l)
#if part=='beg':
#    print('using beginning')
#    dict_crop['dFF'] = dict_crop['dFF'][:,:-2*b]
#    dict_crop['time'] = dict_crop['time'][:-2*b]
#    dict_crop['trialFlag'] = dict_crop['trialFlag'][:-2*b]
#    dict_crop['behavior'] = dict_crop['behavior'][:-2*b]
#    dict_crop['beh_labels'] = dict_crop['beh_labels'][:-2*b]
#if part=='mid':
#    print('using middle')
#    dict_crop['dFF'] = dict_crop['dFF'][:,b:-b]
#    dict_crop['time'] = dict_crop['time'][b:-b]
#    dict_crop['trialFlag'] = dict_crop['trialFlag'][b:-b]
#    dict_crop['behavior'] = dict_crop['behavior'][b:-b]
#    dict_crop['beh_labels'] = dict_crop['beh_labels'][b:-b]
#if part=='end':
#    print('using end')
#    dict_crop['dFF'] = dict_crop['dFF'][:,2*b:]
#    dict_crop['time'] = dict_crop['time'][2*b:]
#    dict_crop['trialFlag'] = dict_crop['trialFlag'][2*b:]
#    dict_crop['behavior'] = dict_crop['behavior'][2*b:]
#    dict_crop['beh_labels'] = dict_crop['beh_labels'][2*b:]
part='whole'

ro = model.reg_obj(activity=activity, 
                    data_dict=copy.deepcopy(dict_crop),
                    exp_id=expt_id,
                    use_beh_labels=True,
                    use_only_valid=True)
#ro.is_downsampled = True
ro.exclude_regressors = ['gamma_0']

if (part_i==0) or (part_i==10) or (part_i==999):
    # ro.model_fit = ro.fit_and_eval_reg_model(shifted=None, exclude_regressors=['gamma_0'])
    tau_inits=[2,5,8,20,40]
    initial_conds = ro.get_default_inits()
    ro.model_fit = ro.get_model_mle(shifted=None, initial_conds=initial_conds.copy(), tau_inits=tau_inits)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p0_all.pkl', "wb" ) )

    if (part_i==10) or (part_i==999):
        ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=True)
        pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p2_all.pkl', "wb" ) )
        ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=False)
        pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p2_all.pkl', "wb" ) )

if part_i==1:
    ro.model_fit = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p0_all.pkl', "rb" ) )
    ro.downsample_in_time()
    ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=True)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p2_all.pkl', "wb" ) )
    ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=False)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p2_all.pkl', "wb" ) )



if (part_i==-1) or (part_i==999):
    print('Testing model on circshifted data')
    tau_inits=[2,5,8,20,40]
    initial_conds = ro.get_default_inits()
    ro.downsample_in_time()
    ro.get_circshift_behav_data(n_perms=n_perms)
    ro.model_fit_shifted = [None]*n_perms
    for n in range(n_perms):
        print('Perm '+str(n))
        
        #ro.model_fit_shifted[n] = ro.fit_and_eval_reg_model(shifted=n, exclude_regressors=['gamma_0'])
        ro.model_fit_shifted[n] = ro.get_model_mle(shifted=n, initial_conds=initial_conds.copy(), tau_inits=tau_inits)
        ro.evaluate_model(model_fit=ro.model_fit_shifted[n], parallel=False, 
                        refit_model=False, shifted=n, fields_to_save=['r_sq', 'stat', 'cc'])
        pickle.dump( ro.model_fit_shifted, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_shifted_'+part+'_3p2_all.pkl', "wb" ) )
    









    
