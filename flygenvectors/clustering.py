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
from scipy.stats import zscore
from scipy.stats import multivariate_normal

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, colors
from mpl_toolkits import mplot3d
import matplotlib.pylab as pl
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

import data as dataUtils
import regression_model as model
import plotting
import flygenvectors.ssmutils as utils
import flygenvectors.utils as futils
from sklearn.linear_model import ElasticNet

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")






class flyg_clust_obj:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.mvn_obj = mvn_obj(self.data_dict)


    def get_clusters(self, n_neighbors=10, affinity='euclidean', linkage='ward', distance_threshold=0.5):
        connectivity_orig = kneighbors_graph(self.data_dict['train_all'].T, n_neighbors=n_neighbors, include_self=False)
        self.cluster = AgglomerativeClustering(n_clusters=None, connectivity=connectivity_orig,
                                          affinity=affinity, linkage=linkage,distance_threshold=distance_threshold)  
        self.cluster.fit_predict(self.data_dict['train_all'].T)
        self.nClust = len(np.unique(self.cluster.labels_))
        print('N clusters: ' + str(self.nClust) )        


    def get_null_dist(self, n_samples=1000):
        counts = np.bincount(self.cluster.labels_)
        A = counts.max() #np.argmax(counts)
        self.null_dist = np.zeros((A+1,n_samples))
        self.null_symmetry = np.zeros((A+1,n_samples))
        self.n_samples=n_samples

        print('Max Clust size is '+str(A))
        for i in range(2,A+1):
            print(i,end=' ')
            for j in range(n_samples):
                # check if clust is on both sides
                self.mvn_obj.is_on_both_sides = False
                while not self.mvn_obj.is_on_both_sides:
                    self.mvn_obj.order = np.random.permutation(self.data_dict['val_all'].shape[1])
                    self.mvn_obj.is_clust_on_both_sides(self.mvn_obj.order[:i])
                
                # variance of randomly defined clusters
                samp = self.data_dict['val_all'][:,self.mvn_obj.order[:i]]
                self.null_dist[i,j] = samp.var(axis=1).mean()
                
                # corresponding symmetry of randomly defined clusters
                self.mvn_obj.get_img(idx=self.mvn_obj.order[:i])
                self.mvn_obj.get_folded_cdf()
                self.null_symmetry[i,j] = self.mvn_obj.folded_cdf


    def cross_validate_clusters(self, clust_p_thresh=0.05):
        self.clust_p_val = np.zeros(self.nClust)
        self.clust_is_sig = np.zeros(self.nClust)
        for k in range(self.nClust):
            cIds = [i for i,j in enumerate(self.cluster.labels_) if j==k]
            l = len(cIds)
            samp_var = self.data_dict['val_all'][:,cIds].var(axis=1).mean()
            self.clust_p_val[k] = (self.null_dist[l,:]<samp_var).sum()/self.null_dist.shape[1]
            if l==1: self.clust_p_val[k]=1
            self.clust_is_sig[k] = self.clust_p_val[k]<clust_p_thresh


    def get_clust_symmetry(self):
        self.n_sig_clust = sum(self.clust_is_sig)
        self.clust_symmetry = np.zeros(self.nClust) #np.zeros(n_sig_clust)
        self.clust_symmetry_pval = np.zeros(self.nClust) #np.zeros(n_sig_clust)
        self.clust_on_both_sides = np.zeros(self.nClust) #np.zeros(n_sig_clust)
        for k in range(self.nClust):
            cIds = [i for i,j in enumerate(self.cluster.labels_) if j==k]    
            self.mvn_obj.get_img(idx=cIds)
            self.mvn_obj.get_folded_cdf()
            self.clust_symmetry[k] = self.mvn_obj.folded_cdf
            l = len(cIds)
            self.clust_symmetry_pval[k] = (self.null_symmetry[l,:]<self.clust_symmetry[k]).sum()/self.n_samples
            
            # check for both sides
            self.mvn_obj.is_clust_on_both_sides(cIds)
            self.clust_on_both_sides[k] = self.mvn_obj.is_on_both_sides


    def sort_clusters_by_n_members(self):
        self.n_clust_members = [None]*(np.bincount(self.cluster.labels_).max()+1)
        for k in range(self.nClust):
            clust_k = k #sig_clust[k]
            cIds = [i for i,j in enumerate(self.cluster.labels_) if j==clust_k]    
            l = len(cIds)
            if self.n_clust_members[l] is None:
                self.n_clust_members[l] = [k]
            else:
                self.n_clust_members[l].append(k)


    def make_clust_colors(self):
        # keep only clusters that are significant and size=2
        self.agg_clusters = self.cluster.labels_.copy()
        for i in range(len(self.clust_is_sig)):
            if not self.clust_is_sig[i]:
                self.agg_clusters[self.agg_clusters==i] = -1
            if i not in self.n_clust_members[2]:
                self.agg_clusters[self.agg_clusters==i] = -1
                

class mvn_obj:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.sig = 200 # standard deviation of gaussian blur for each point (in microns)
        self.grid_points = 50
        self.mid_x = round(self.grid_points/2)
        self.N = data_dict['aligned_centroids'].shape[0]
        self.mvn_array = []
        for n in range(self.N):
            self.mvn_array.append( multivariate_normal(data_dict['aligned_centroids'][n,:], self.sig ) ) 
            
        mx = data_dict['aligned_centroids'].max(axis=0)
        self.median_fold_point = 190 #138 #127 # hardcoded median based on template
        #self.median_fold_point = np.median(data_dict['aligned_centroids'][:,0]) #mx[0]/2
        gp = self.grid_points
        x, y, z = np.mgrid[0:mx[0]:(mx[0]/gp+1e-10), 0:mx[1]:(mx[1]/gp+1e-10), 0:mx[2]:(mx[2]/gp+1e-10)]
        self.pos = np.empty(x.shape + (3,))
        self.pos[:, :, :, 0] = x; self.pos[:, :, :, 1] = y; self.pos[:, :, :, 2] = z

    def is_clust_on_both_sides(self,idx):
        x_px_vals = self.data_dict['aligned_centroids'][idx,0]
        is_on_left = (x_px_vals<self.median_fold_point).sum()>0
        is_on_right = (x_px_vals>self.median_fold_point).sum()>0
        self.is_on_both_sides = is_on_left and is_on_right
        
    def get_img(self,idx):
        # get pdf of gmm for indices=idx  
        self.grid_image = np.zeros((self.grid_points,self.grid_points,self.grid_points))
        for i in idx:
            self.grid_image += self.mvn_array[i].pdf(self.pos)
            
    def get_folded_cdf(self):
        # estimate CDF of folded distribution  
        L = self.grid_image[:self.mid_x,:,:]
        R = self.grid_image[self.mid_x:,:,:][::-1,:,:]
        self.folded_image = L*R
        self.folded_cdf = self.folded_image.sum()
