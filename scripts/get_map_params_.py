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

print('loading data')
sys.stdout.flush()


## SET PARAMETERS
sig = 30 # standard deviation of gaussian blur for each point (in microns)
nearby_th = 1000 #squared distance (um) at which to truncate Gaussian

## LOAD DATA
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

data_tot = []

for i in range(len(exp_list)):
    exp_date = exp_list[i][0]
    fly_num = exp_list[i][1]
    expt_id = exp_list[i][0] + '_' + exp_list[i][1]

    infile = open(main_dir+expt_id+'.pkl','rb')
    data_dict = pickle.load(infile)
    print(expt_id)
    sys.stdout.flush()

    data_tot.append({})
    data_tot[i]['data_dict'] = data_dict


# for all datasets, generate GMM from pointset
n_flies = np.shape(exp_list)[0]

# convert all obj curves into distributions (flip sign, shift & scale to sum to 1)
mvn_array_tot = []
for nf in range(n_flies):
    data_dict = data_tot[nf]['data_dict']
    N = data_dict['rate'].shape[0]
    
    # make array of mvn objects for each cell
    #N = data_dict['aligned_centroids'].shape[0]
    mvn_array = []
    for n in range(N):
        mvn_array.append( multivariate_normal(data_dict['aligned_centroids'][n,:], sig ) ) 
    mvn_array_tot.append( mvn_array )
    

# NT,NP = model_fit[0][0]['obj_tot'].shape
model_as_dist = []
print('loading dist files')
sys.stdout.flush()

for nf in range(n_flies):
    print(nf,end=' ')
    sys.stdout.flush()
    exp_date = exp_list[nf][0]
    fly_num = exp_list[nf][1]
    expt_id = exp_list[nf][0] + '_' + exp_list[nf][1]
    mdist = np.load(main_dir+expt_id+'_as_dist.npy')     
    model_as_dist.append( mdist )
    

# compute MAP estimate
nf = int(sys.argv[1]) # DATASET TO LOAD
data_dict = data_tot[nf]['data_dict']
N = data_dict['rate'].shape[0]
# N = len(model_fit_tot[nf])
pos = data_tot[nf]['data_dict']['aligned_centroids']
mdist = np.load(main_dir+expt_id+'_as_dist.npy')
# NP = sweep_tot[nf]['obj_tot'].shape
NP = mdist.shape


prior_tot_fly = np.zeros(NP)
MAP_tot_fly = np.zeros((NP[0],len(NP)-1))

# for every cell, compute prior from all other datasets
print('\ncomputing posteriors')
sys.stdout.flush()
curated_omit=[]

for nf_oth in range(n_flies):
    print('\n\n'+str(nf_oth))
    if nf_oth==nf: continue
    expt_id_oth = exp_list[nf_oth][0] + '_' + exp_list[nf_oth][1]
    mdist_oth = np.load(main_dir+expt_id_oth+'_as_dist.npy')
    
    if nf_oth not in curated_omit:
        # M = len(model_fit_tot[nf_oth])
        M = data_tot[nf_oth]['data_dict']['rate'].shape[0]
        pos_nf_oth = data_tot[nf_oth]['data_dict']['aligned_centroids']
           
        for n in range(N):
            if not np.mod(n,100): print(n,end=' ')
            sys.stdout.flush()

            # find nearby cells, and                 # **********************
            nearby_list=[]
            for m in range(pos_nf_oth.shape[0]):
                s = ((pos[n,:]-pos_nf_oth[m,:])**2).sum()
                if s<nearby_th:
                    nearby_list.append(m)

            for m in nearby_list:
                # build prior using cells with good regression fits
                # if( model_fit_tot[nf_oth][m]['stat'][1]<p_th ):   # ******* THIS NOW NEEDS TO BE A LOOP OVER PARAMS, CHECK EACH PVAL
                #prior += model_as_dist[nf_oth][m] * mvn_array_tot[nf_oth][m].pdf(pos[n,:]) 
                prior_tot_fly[n] += mdist_oth[m] * mvn_array_tot[nf_oth][m].pdf(pos[n,:])
  
# from prior and likelihood, compute MAP parameters
for n in range(N):
    posterior = prior_tot_fly[n] * mdist[n]      
    #pdb.set_trace()
    MAP_tot_fly[n] = np.unravel_index( np.argmax( posterior ), NP[1:] )              

np.save(main_dir+expt_id+'_prior.npy', prior_tot_fly)
np.save(main_dir+expt_id+'_MAP.npy', MAP_tot_fly)

