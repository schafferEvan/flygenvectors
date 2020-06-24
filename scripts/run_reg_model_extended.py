import sys
sys.path.insert(0, '../flygenvectors/')
import os
import numpy as np
from glob import glob
import copy
from importlib import reload
import pickle

import scipy.io as sio
from scipy import sparse, signal
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png', 'pdf')

#import seaborn as sns
#sns.set_style("white")
#sns.set_context("talk")

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
# from skimage.restoration import denoise_tv_chambolle

import data as dataUtils
import regression_model as model
#import plotting
#import flygenvectors.ssmutils as utils

print('imports complete')
<<<<<<< HEAD
sys.stdout.flush() 
=======
sys.stdout.flush()
>>>>>>> 62ae45c30a7c3e4d5f23f1f0b3f1d1edb65313ca


## LOAD DATA
# load deconvolved data and point to the right entry
main_dir = '/moto/axs/projects/traces/flygenvectors/'

exp_list = [
            ['2019_04_18','fly2'],
            ['2019_04_22','fly1'],
            ['2019_04_22','fly3'],
            ['2019_04_24','fly3'],
            ['2019_04_24','fly1'],
            ['2019_04_25','fly3'],
            ['2019_05_07','fly1'],
            ['2019_03_12','fly4'],
            ['2019_02_19','fly1'],
            ['2019_02_26','fly1_2']]

i = int(sys.argv[1]) # DATASET TO LOAD
exp_date = exp_list[i][0]
fly_num = exp_list[i][1]
expt_id = exp_list[i][0] + '_' + exp_list[i][1]
infile = open(main_dir+expt_id+'.pkl','rb')
data_dict = pickle.load(infile)
print(expt_id)
print('data loaded')
<<<<<<< HEAD
sys.stdout.flush() 
=======
sys.stdout.flush()
>>>>>>> 62ae45c30a7c3e4d5f23f1f0b3f1d1edb65313ca


## FIT MODEL
ro = model.reg_obj()
ro.data_dict = data_dict
ro.fit_reg_model_full_likelihood()


## SAVE OUTPUT
#np.savez( main_dir+'/output/'+expt_id+'_reg_model.npz', P_tot=ro.P_tot, obj_tot=ro.obj_tot )
np.save( main_dir+'/output/'+expt_id+'_reg_model_P_tot_alpha_01.npy', ro.P_tot['alpha_01'])
np.save( main_dir+'/output/'+expt_id+'_reg_model_P_tot_trial.npy', ro.P_tot['trial'])
np.save( main_dir+'/output/'+expt_id+'_reg_model_obj_tot.npy', ro.obj_tot)

#model_fit_tot = {'P_tot':ro.P_tot,'obj_tot':ro.obj_tot}
<<<<<<< HEAD
#pickle.dump( model_fit_tot, open( main_dir+'/output/'+expt_id+'_reg_model.pkl', "wb" ) )
=======
#pickle.dump( model_fit_tot, open( main_dir+'/output/'+expt_id+'_reg_model.pkl', "wb" ) )
>>>>>>> 62ae45c30a7c3e4d5f23f1f0b3f1d1edb65313ca
