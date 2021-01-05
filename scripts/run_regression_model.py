#!/usr/bin/python
"""
Master script to run regression on one dataset
"""

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


# LOAD DATA ---------------------------------------------------------------------------
pickle_output = True
exp_date = '2018_08_24' 
fly_num = 'fly3_run1'

expt_id = exp_date+'_'+fly_num
dirs = futils.get_dirs()
fig_dirs = futils.get_fig_dirs(expt_id)
data_dict = dataUtils.load_timeseries_simple(exp_date,fly_num,dirs['data'])
# pdb.set_trace()
ro = model.reg_obj(exp_id=expt_id, 
                   data_dict=data_dict,
                   fig_dirs=fig_dirs,
                   split_behav=False)
data_dict = ro.preprocess()

if pickle_output:
    pickle.dump( data_dict, open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "wb" ) )


# ANALYSIS OF TIME CONSTANTS RELATING DFF TO BEHAVIOR ------------------------------------
# find optimal time constant and phase PER NEURON with which to filter ball trace to maximize correlation
# tauList, corrMat = dataUtils.estimate_neuron_behav_tau(data_dict)
activity='dFF'
ro = model.reg_obj(activity=activity)
ro.data_dict = data_dict
ro.params['split_behav'] = False
ro.elasticNet = False
ro.fit_and_eval_reg_model_extended()
pickle.dump( ro.model_fit, open( fig_dirs['pkl_dir'] + expt_id +'_'+ro.activity+'_ols_reg_model.pkl', "wb" ) )

