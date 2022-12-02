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

exp_list = [
            '2019_04_18_fly2',
            '2019_04_22_fly1',
            '2019_04_24_fly1',
            '2019_05_07_fly1',
            '2019_04_22_fly3',
            '2019_04_24_fly3',
            '2019_04_26_fly1',
            '2019_04_29_fly1',
            '2019_04_25_fly1',
            '2019_04_25_fly2',
            '2019_04_25_fly3']

i = int(sys.argv[1]) # DATASET TO LOAD
part_i = int(sys.argv[2])
n_perms = int(sys.argv[3])
# for expt_id in exp_list:
expt_id = exp_list[i] #[0] + '_' + exp_list[i][1]
print(expt_id)

n_perms = 5
print('Manually setting n_perms to 5')

dirs = futils.get_dirs()
fig_dirs = futils.get_fig_dirs(expt_id)
data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )
data_dict['beh_labels'] = np.expand_dims( data_dict['behavior'], axis=1).copy()
data_dict['beh_labels'] = 1*(data_dict['beh_labels']>.0005)
dict_crop = copy.deepcopy(data_dict)

part='whole'
print('using whole')

ro = model.reg_obj(activity=activity, 
                    split_behav=True,
                    data_dict=dict_crop,
                    exp_id=expt_id,
                    use_beh_labels=False,
                    use_only_valid=True)
#ro.is_downsampled = True
ro.exclude_regressors = ['gamma_0']

if part_i==0:
    # ro.model_fit = ro.fit_and_eval_reg_model(shifted=None, exclude_regressors=['gamma_0'])
    tau_inits=[2,5,8,20,40]
    initial_conds = ro.get_default_inits()
    ro.model_fit = ro.get_model_mle(shifted=None, initial_conds=initial_conds.copy(), tau_inits=tau_inits)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_4p1_all.pkl', "wb" ) )

    ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=True)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_4p1_all.pkl', "wb" ) )
    ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=False)
    pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_4p1_all.pkl', "wb" ) )

    #ro.data_dict = dict_eval
    #ro.downsample_in_time()
    #ro.evaluate_model(model_fit=ro.model_fit, parallel=True, refit_model=False)
    #pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_4p0_test.pkl', "wb" ) )



if part_i==1:
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
        pickle.dump( ro.model_fit_shifted, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_shifted_'+part+'_4p1_all.pkl', "wb" ) )
    









    
