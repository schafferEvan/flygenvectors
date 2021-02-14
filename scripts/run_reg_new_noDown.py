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
import flygenvectors.ssmutils as utils
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
            '2018_08_24_fly2_run2'] 

for expt_id in exp_list:
    dirs = futils.get_dirs()
    fig_dirs = futils.get_fig_dirs(expt_id)
    data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )

    # rerun smooth_behavior (this is a hack)
    if (expt_id=='2018_08_24_fly3_run1') or (expt_id=='2018_08_24_fly2_run2'):
        ro = model.reg_obj(activity=activity, 
                            data_dict=copy.deepcopy(data_dict),
                            exp_id=expt_id)
        ro.get_smooth_behavior()
        data_dict['behavior'] = ro.data_dict['behavior']

    # crop time (crude bootstrapping)
    # for part in ['beg', 'mid', 'end']:
    for part in ['end']:
        dict_crop = copy.deepcopy(data_dict)
        l = len(data_dict['behavior'])
        b = round(.15*l)
        if part=='beg':
            dict_crop['dFF'] = dict_crop['dFF'][:,:-2*b]
            dict_crop['time'] = dict_crop['time'][:-2*b]
            dict_crop['trialFlag'] = dict_crop['trialFlag'][:-2*b]
            dict_crop['behavior'] = dict_crop['behavior'][:-2*b]
            dict_crop['beh_labels'] = dict_crop['beh_labels'][:-2*b]
        if part=='mid':
            dict_crop['dFF'] = dict_crop['dFF'][:,b:-b]
            dict_crop['time'] = dict_crop['time'][b:-b]
            dict_crop['trialFlag'] = dict_crop['trialFlag'][b:-b]
            dict_crop['behavior'] = dict_crop['behavior'][b:-b]
            dict_crop['beh_labels'] = dict_crop['beh_labels'][b:-b]
        if part=='end':
            dict_crop['dFF'] = dict_crop['dFF'][:,2*b:]
            dict_crop['time'] = dict_crop['time'][2*b:]
            dict_crop['trialFlag'] = dict_crop['trialFlag'][2*b:]
            dict_crop['behavior'] = dict_crop['behavior'][2*b:]
            dict_crop['beh_labels'] = dict_crop['beh_labels'][2*b:]


        ro = model.reg_obj(activity=activity, 
                            data_dict=copy.deepcopy(dict_crop),
                            exp_id=expt_id)
        self.is_downsampled = True
        ro.fit_and_eval_reg_model_extended()
        pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_'+part+'_2p0.pkl', "wb" ) )
        pickle.dump( ro.model_fit_shifted, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model_shifted_'+part+'_2p0.pkl', "wb" ) )


    