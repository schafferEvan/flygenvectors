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
            '2019_08_14_fly3_2'] 

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



dict_crop = copy.deepcopy(data_dict)

part='whole'

ro = model.reg_obj(activity=activity, 
                    data_dict=copy.deepcopy(dict_crop),
                    exp_id=expt_id,
                    use_beh_labels=True,
                    use_only_valid=True)
ro.is_downsampled = True

# ro.fit_and_eval_reg_model_extended(n_perms=n_perms)
# ro.model_fit = ro.fit_and_eval_reg_model(shifted=None, exclude_regressors=['gamma_0'])
# pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p0_all.pkl', "wb" ) )




ro.exclude_regressors = ['gamma_0']
ro.model_fit = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p0_all.pkl', "rb" ) )

ro.evaluate_model(model_fit=ro.model_fit, parallel=False, refit_model=True)
pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p1_all.pkl', "wb" ) )

ro.evaluate_model(model_fit=ro.model_fit, parallel=False, refit_model=False)
pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_3p2_all.pkl', "wb" ) )

    









    
