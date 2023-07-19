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

exp_list = ['2022_01_05_fly2',
            '2022_01_08_fly1',
            '2022_01_08_fly2',
            '2022_01_14_fly1',
            '2022_01_19_fly1',
            '2022_01_21_fly2',
            '2022_01_25_fly2'] 
# exp_list = [
#             '2022_01_14_fly1',
#             ] 

run_fit = True
run_circ = True
n_perms = 100
part='whole'

for expt_id in exp_list:
    print(expt_id)

    dirs = futils.get_dirs()
    fig_dirs = futils.get_fig_dirs(expt_id)
    data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )

    ro = model.reg_obj(activity=activity, 
                        data_dict=data_dict,
                        exp_id=expt_id,
                        use_beh_labels=False,
                        use_only_valid=True)
    #ro.is_downsampled = True
    ro.exclude_regressors = ['gamma_0']

    if run_fit:
        # ro.model_fit = ro.fit_and_eval_reg_model(shifted=None, exclude_regressors=['gamma_0'])
        tau_inits=[2,5,8,12]
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


    if run_circ:
        print('Testing model on circshifted data')
        tau_inits=[2,5,8,12]
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
        









    
