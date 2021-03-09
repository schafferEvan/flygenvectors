
# ********************************
# ********************************
# DEPRECATED *********************
# ********************************
# ********************************



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
remake_model = True


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
    data_dict = dataUtils.load_timeseries_simple(exp_date,fly_num,dirs['data'])
    # pdb.set_trace()
    ro = model.lr_obj(exp_id=expt_id, 
                       data_dict=data_dict,
                       fig_dirs=fig_dirs,
                       split_behav=False,
                       iters=150, 
                       n_components=100, 
                       use_p2=False)

    if remake_model:
        # PREPROCESS
        ro.data_dict = ro.reg_obj.preprocess()
        pickle.dump( ro.data_dict, open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "wb" ) )

        # FIT MODEL
        print('\rFitting model for: '+expt_id)
        ro.fit_model()
        ro.model_fit['mse'] = ro.mse

        if ro.use_p2:
            pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'_wP2_PCreg_model.pkl'
        else:
            pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'PCreg_model.pkl'
        pickle.dump( ro.model_fit, open( pkl_name, "wb" ) )
        print('Saved Model Output As: '+pkl_name)
    else:
        ro.data_dict = pickle.load( open( fig_dirs['pkl_dir'] + expt_id +'_dict.pkl', "rb" ) )

        # load model
        print('Loading model from '+expt_id)
        if ro.use_p2:
            pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'_wP2_PCreg_model.pkl'
        else:
            pkl_name = fig_dirs['pkl_dir'] + expt_id +'_'+str(ro.n_components)+'PCreg_model.pkl'
        ro.model_fit = pickle.load( open( pkl_name, "rb" ) )


