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
# import flygenvectors.ssmutils as utils
# import flygenvectors.utils as futils
from sklearn.linear_model import ElasticNet

from joblib import Parallel, delayed
import multiprocessing

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")






class flyg_clust_obj:
    def __init__(self, data_dict, n_cores=None):
        self.data_dict = data_dict
        self.mvn_obj = mvn_obj(self.data_dict)
        if n_cores is None:
            self.num_cores = multiprocessing.cpu_count()
        else:
            self.num_cores = n_cores


    def get_clusters(self, n_neighbors=10, affinity='euclidean', linkage='ward', distance_threshold=0.5, n_clusters=None, quiet=False):
        # hierarchical clustering
        connectivity_orig = kneighbors_graph(self.data_dict['train_all'].T, n_neighbors=n_neighbors, include_self=False)
        if n_clusters is None:
            self.cluster = AgglomerativeClustering(n_clusters=None, connectivity=connectivity_orig,
                                              affinity=affinity, linkage=linkage,distance_threshold=distance_threshold) 
        else:
            self.cluster = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity_orig,
                                              affinity=affinity, linkage=linkage)  
        self.cluster.fit_predict(self.data_dict['train_all'].T)
        self.nClust = len(np.unique(self.cluster.labels_))
        if not quiet:
            print('N clusters: ' + str(self.nClust) )  


    def get_simple_clusters(self, activity='dFF'):   
        # simple clustering looking for closed loops in correlation
        # compute pairwise correlations
        N = self.data_dict[activity].shape[0]
        C = self.data_dict[activity]@self.data_dict[activity].T - np.eye(N)
        accounted_for=[False]*N
        pairs=[]

        for i in range(N):
            if not accounted_for[i]:
                j = np.argmax(C[i,:])
                if np.argmax(C[j,:])==i:
                    pairs.append([i,j])
                    accounted_for[j] = True
                    accounted_for[i] = True
                    
        # C += np.eye(N)
        self.simple_clust = {'C':C, 'pairs':pairs, 'accounted_for':accounted_for}
        print('N clusters (simple): ' + str(len(pairs)) )
        self.order_cells_by_simple_pairs(activity=activity)


    def get_simple_clust_popularity(self):
        C = self.simple_clust['C']
        N = C.shape[0]
        popularity = np.zeros(N)
        for i in range(N):
            j = np.argmax(C[i,:])
            popularity[j] += 1
        return popularity


    def get_simple_clusters_euc(self):   
        # simple clustering looking for closed loops in euclidian distance
        N = self.data_dict['dFF'].shape[0]
        E = np.zeros((N,N))
        for i in range(N):
            E[:,i] = ( self.data_dict['dFF'] - np.roll(self.data_dict['dFF'], -i, axis=0) ).var(axis=1)
        for i in range(N):
            E[i,:] = np.roll(E[i,:],i)
        for i in range(N):
            E[i,i] = np.inf

        accounted_for=[False]*N
        pairs=[]
        for i in range(N):
            if not accounted_for[i]:
                j = np.argmin(E[i,:])
                if np.argmin(E[j,:])==i:
                    pairs.append([i,j])
                    accounted_for[j] = True
                    accounted_for[i] = True
                    
        # C += np.eye(N)
        self.simple_clust_euc = {'E':E, 'pairs':pairs, 'accounted_for':accounted_for}
        print('N clusters (simple euc): ' + str(len(pairs)) )
        # self.order_cells_by_simple_pairs()


    def order_cells_by_simple_pairs(self, activity='dFF'):
        # order remaining cells by sequential similarity to previous seed
        pairs = self.simple_clust['pairs']
        accounted_for = self.simple_clust['accounted_for']
        C = self.simple_clust['C']
        N = self.data_dict[activity].shape[0]
        flat_pairs = [item for sublist in pairs for item in sublist]
        order_from_pairs = np.zeros(N, dtype=int)
        order_from_pairs[:len(flat_pairs)] = flat_pairs

        unaccounted_for = np.flatnonzero(np.array(accounted_for)==False).tolist()
        # new_not_accounted_for = np.ones(len(order_from_pairs)-len(flat_pairs))
        seed = int(order_from_pairs[-1])
        for j in range(N-len(flat_pairs)):
            new_seed = np.argmax(C[seed, unaccounted_for])
            order_from_pairs[j+len(flat_pairs)] = unaccounted_for[new_seed]
            unaccounted_for.remove(unaccounted_for[new_seed])
            seed = new_seed
        self.simple_clust['order_from_pairs'] = order_from_pairs #.astype(int)
        F_ord = self.data_dict[activity][self.simple_clust['order_from_pairs'],:]
        self.simple_clust['C_ord'] = F_ord@F_ord.T
                 

    def normalize_rows_euc(self,input_mat,fix_nans=True): 
        output_mat = np.zeros(input_mat.shape)
        mu = np.zeros(input_mat.shape[0])
        sig = np.zeros(input_mat.shape[0])
        for i in range(input_mat.shape[0]):
            mu[i] = input_mat[i,:].mean()
            sig[i] = input_mat[i,:].std()*np.sqrt(input_mat.shape[1])
            output_mat[i,:] = (input_mat[i,:]-mu[i])/sig[i]
            if fix_nans:
                output_mat[i,:][np.isnan(output_mat[i,:])]=0
        return output_mat, mu, sig


    def unnormalize_rows_euc(self,input_mat,mu,sig): 
        output_mat = np.zeros(input_mat.shape)
        for i in range(input_mat.shape[0]):
            output_mat[i,:] = input_mat[i,:]*sig[i] + mu[i] #(input_mat[i,:]-mu[i])/sig[i]
        return output_mat


    def get_null_dist(self, n_samples=1000, mx_null=None, enforce_both_sides=False, behav_null=False, parallel=True):
        """
        behav_null: whether to use completely random pairs or pairs with similar behav corr
        mx_null: optional upper bound on clust size for which to compute null dist 
                (if large clusters are going to be ignored)
        enforce_both_sides: whether to only test samples from both sides
                            (to test *symmetry* rather than more general structure)
        """
        counts = np.bincount(self.cluster.labels_)
        A = counts.max() #np.argmax(counts)
        if mx_null is None:
            mx_null = A+1
        null_dist = np.zeros((A+1,n_samples))
        null_symmetry = np.zeros((A+1,n_samples))
        null_dist_score = np.zeros((A+1,n_samples))
        self.n_samples=n_samples
        if behav_null:
            behav_null_idx, behav_null_prob = self.get_behav_corr_groups_for_null()
            null_behav_samples = self.generate_behav_samples(behav_null_idx, behav_null_prob, N=mx_null*n_samples)

        print('Max Clust size is '+str(A))
        print('Computing null up to size = '+str(mx_null))
        for i in range(2,mx_null):
            print(i,end=' ')
            if parallel:
                out_tot = Parallel(n_jobs=self.num_cores)(delayed(
                    self.get_null_sample)(enforce_both_sides=enforce_both_sides, behav_null=behav_null, sz=i) for n in range(self.n_samples))
                for j in range(self.n_samples):
                    null_dist[i,j] = out_tot[j][0]
                    null_symmetry[i,j] = out_tot[j][1]
                    null_dist_score[i,j] = out_tot[j][2]
            else:
                for j in range(self.n_samples):
                    null_dist[i,j], null_symmetry[i,j], null_dist_score[i,j] = self.get_null_sample(enforce_both_sides=enforce_both_sides, behav_null=behav_null, sz=i)
            
        return null_dist, null_symmetry, null_dist_score


    def get_null_sample(self, enforce_both_sides=None, behav_null=None, sz=None):
        if enforce_both_sides:
            # check if clust is on both sides
            self.mvn_obj.is_on_both_sides = False
            while not self.mvn_obj.is_on_both_sides:
                if behav_null:
                    print('not yet implemented')
                else:
                    self.mvn_obj.order = np.random.permutation(self.data_dict['val_all'].shape[1])
                self.mvn_obj.is_clust_on_both_sides(self.mvn_obj.order[:sz])
        else:
            if behav_null:
                self.mvn_obj.order = next(null_behav_samples)
            else:
                self.mvn_obj.order = np.random.permutation(self.data_dict['val_all'].shape[1])
        
        # variance of randomly defined clusters
        samp = self.data_dict['val_all'][:,self.mvn_obj.order[:sz]]
        null_dist_sample = samp.var(axis=1).mean()
        
        # corresponding symmetry of randomly defined clusters
        self.mvn_obj.get_img(idx=self.mvn_obj.order[:sz])
        self.mvn_obj.get_folded_cdf()
        null_symmetry_sample = self.mvn_obj.folded_cdf

        # distance score of randomly defined clusters
        self.mvn_obj.get_folded_cdf_totprod(idx=self.mvn_obj.order[:sz])
        null_dist_score_sample = self.mvn_obj.folded_product_image.sum()

        return null_dist_sample, null_symmetry_sample, null_dist_score_sample


    def generate_behav_samples(self, behav_null_idx, behav_null_prob, N=3000, low_bound=.2):
        """
        Randomly select a pair (or more) of cells with approximately similar behav corr
        behav_null_idx, behav_null_prob = get_behav_corr_groups_for_null()
        low_bound: omit samples from small buckets (typically the tails of the distribution)
        Randomly picks a bucket (cells with approx similar behav corr), 
            flips biased coin to decide whether to use that bucket (based on bucket size),
            if valid, returns random permutation of cells in that bucket.
        """
        M = 20 #10 # to get N good samples, need M*N samples to choose from
        bucket_order_init = np.random.randint(0,len(behav_null_idx),M*N)
        p = low_bound + (1-low_bound)*np.random.rand(M*N)
        valid_bucket_order = [None]*N
        valid_samples = [None]*N
        ctr=0
        for i in range(M*N):
            if p[i]<behav_null_prob[bucket_order_init[i]]:
                valid_bucket_order[ctr]=bucket_order_init[i]
                ctr += 1
            if ctr==N: break

        # genrate random permutations from selected buckets
        for i in range(N):
            try:
                valid_samples[i] = np.random.permutation( behav_null_idx[valid_bucket_order[i]] )
            except Exception:
                pdb.set_trace()

        samples_iter = (i for i in valid_samples)
        return samples_iter
        # valid_sample=False
        # while not valid_sample:
        #     bucket = np.random.randint(0,len(behav_null_idx))
        #     p = low_bound + (1-low_bound)*np.random.rand(1)
        #     if p<behav_null_prob[bucket]:
        #         valid_sample=True
        #         order = np.random.permutation( behav_null_idx[bucket] )
        # return order


    def get_behav_corr_groups_for_null(self, bin_size=0.02):
        # make list of cells with approximately equal behav corr 
        # (to generate null hypothesis that behav corr explains spatial order)
        # corr_prob gives prob of each bin, used for later sampling
        beh = self.data_dict['behavior'].copy()
        beh -= beh.mean()
        beh /= beh.std()
        behav_cc = self.data_dict['dFF']@beh/np.sqrt(len(beh))
        binned_corr = np.round( behav_cc/bin_size )
        U = np.unique(binned_corr)
        corr_idx = [None]*len(U)
        corr_prob = np.zeros(len(U))
        for i in range(len(U)):
            corr_idx[i] = np.flatnonzero( binned_corr==U[i] )
            corr_prob[i] = len(corr_idx[i])
        corr_prob /= corr_prob.max() #relative probability
        return corr_idx, corr_prob


    def cross_validate_clusters(self, clust_p_thresh=0.05):
        self.clust_p_val = np.zeros(self.nClust)
        self.clust_is_sig = np.zeros(self.nClust)
        self.clust_val_var = np.zeros(self.nClust)
        for k in range(self.nClust):
            cIds = [i for i,j in enumerate(self.cluster.labels_) if j==k]
            l = len(cIds)
            self.clust_val_var[k] = self.data_dict['val_all'][:,cIds].var(axis=1).mean()
            self.clust_p_val[k] = (self.null_dist[l,:]<self.clust_val_var[k]).sum()/self.null_dist.shape[1]
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


    def get_clust_dist_score(self):
        # similar to get_clust_symmetry, but evaluates spatial organization independent of hemisphere
        self.clust_dist_score = np.zeros(self.nClust) #np.zeros(n_sig_clust)
        self.clust_dist_score_pval = np.zeros(self.nClust) #np.zeros(n_sig_clust)
        for k in range(self.nClust):
            cIds = [i for i,j in enumerate(self.cluster.labels_) if j==k]    
            self.mvn_obj.get_folded_cdf_totprod(idx=cIds)
            self.clust_dist_score[k] = self.mvn_obj.folded_product_image.sum()
            l = len(cIds)
            self.clust_dist_score_pval[k] = (self.null_dist_score[l,:]<self.clust_dist_score[k]).sum()/self.n_samples
            

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


    def make_summary_plot(self, size_list = [2,3,4], behav_null=True):
        plt.figure(figsize=(14,4)) 
        for i in range(len(size_list)):
            to_plot = []
            right_size_clust = self.n_clust_members[size_list[i]] 
            for j in range(self.nClust):
                if self.clust_is_sig[j]:
                    if j in right_size_clust:
                        to_plot.append(j) 
            
            plt.subplot(1,len(size_list),i+1)
            clusts_to_plot = self.clust_dist_score[to_plot]
            ulm = 1.05*clusts_to_plot.max() 
            plt.hist(clusts_to_plot,25,(0,ulm),color='#0485d1',alpha=.6,label='true clusters')

            # null_weights = np.ones_like(self.null_dist_score[size_list[i]]) * len(clusts_to_plot)/len(self.null_dist_score[size_list[i]])
            # plt.hist(self.null_dist_score[size_list[i]],25,(0,ulm),weights=null_weights,color='g',alpha=.5,label='shuffled')

            if behav_null:
                null_weights = np.ones_like(self.null_dist_score_beh[size_list[i]]) * len(clusts_to_plot)/len(self.null_dist_score_beh[size_list[i]])
                plt.hist(self.null_dist_score_beh[size_list[i]],25,(0,ulm),weights=null_weights,color='k',alpha=.7,label='beh. matched shuffled')
            else:
                null_weights = np.ones_like(self.null_dist_score[size_list[i]]) * len(clusts_to_plot)/len(self.null_dist_score[size_list[i]])
                plt.hist(self.null_dist_score[size_list[i]],25,(0,ulm),weights=null_weights,color='k',alpha=.7,label='shuffled')

            plt.xlabel('Spatial Order Score') 
            plt.xlim(0,ulm)
            if i==0:
                plt.ylabel('Count')
                plt.ylim(0,50) 
            elif i==1:
                plt.ylim(0,10)
                plt.yticks([0,5,10])
            else:
                plt.ylim(0,5)
                plt.yticks([0,2.5,5]) 
            plt.legend()
            plt.title('Cluster Size = '+str(size_list[i]))

        plt.tight_layout() 
        # plt.savefig(fig_dirs['clustfig_folder']+expt_id+'_clust_dist_score_hist.pdf',transparent=True, bbox_inches='tight')
        # plt.show()


    def get_linkage(self):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        model = self.cluster
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        return linkage_matrix
                

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
            
        self.mx = data_dict['aligned_centroids'].max(axis=0)
        self.median_fold_point = 190 #138 #127 # hardcoded median based on template
        #self.median_fold_point = np.median(data_dict['aligned_centroids'][:,0]) #mx[0]/2
        gp = self.grid_points
        x, y, z = np.mgrid[0:self.mx[0]:(self.mx[0]/gp+1e-10), 0:self.mx[1]:(self.mx[1]/gp+1e-10), 0:self.mx[2]:(self.mx[2]/gp+1e-10)]
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
            self.grid_image += self.mvn_array[i].pdf(self.pos)*np.prod(self.mx/self.grid_points)
            
    def get_folded_cdf(self):
        # estimate CDF of folded distribution, taking the sum on both hemispheres and then the product across hemispheres (for symmetry score)
        L = self.grid_image[:self.mid_x,:,:]
        R = self.grid_image[self.mid_x:,:,:][::-1,:,:]
        self.folded_image = L*R
        self.folded_cdf = self.folded_image.sum()

    def get_folded_cdf_totprod(self, idx):
        self.folded_product_image = np.ones((self.mid_x,self.grid_points,self.grid_points))
        for i in idx:
            tmp_img = self.mvn_array[i].pdf(self.pos)*np.prod(self.mx/self.grid_points)
            L = tmp_img[:self.mid_x,:,:]
            R = tmp_img[self.mid_x:,:,:][::-1,:,:]
            self.folded_product_image *= (L+R) 



