import os
import sys
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import copy
import pdb



class reg_obj:
    def __init__(self,activity='dFF'):
        self.data_dict = {}
        self.exp_id = None
        self.activity=activity # valid values: {'dFF', 'rate'}, determines what trace is used for fit
        self.params = {'split_behav':True,
                    'run_full_sweep':True,
                    'sigLimSec':60,
                    'phaseLimSec': 60,
                    'M':0,
                    'L':0,
                    'tau':0.1,
                    'mu':0,
                    'tau_feed':0.1
                 }
        self.regressors_dict = {
            'alpha_01':np.array([]), 
            'beta_0':np.array([]), 
            'trial':np.array([]), 
            'drink_hunger':np.array([])
        }
        self.regressors_array = np.array([])
        self.regressors_array_inv = np.array([])
        self.cell_id = 0
        self.phiList = []
        self.tauList = []
        self.phi = 0
        self.p_dict = {} # temp
        self.model_fit = []
        self.P_tot = []
        self.obj_tot = []
        self.baseline_regs_amp_val = 1000
        self.elasticNet = False 
        self.num_tau_feed_steps = 30



    def fit_reg_linear(self, n, phi_idx, tseries_to_sub=None, D=None, Dinv=None, reg_labels=None):
        '''
        n=cell number
        phi=offset
        tseries_to_sub (optional): pre-subtract a timeseries. Useful for parameter sweeps
        D (optional): provide regressor array. Otherwise defaults to reg array attribute of current object
        Dinv (optional): provide regressor array. Otherwise defaults to inv reg array attribute of current object
        reg_labels (optional): required if providing D, unneeded otherwise. Manually provide fields for dict output
        '''
        dFF_full = self.data_dict[self.activity][n,:]
        L = self.params['L']
        dFF = dFF_full[L:-L]
        if tseries_to_sub is not None:
            dFF -= tseries_to_sub

        # for a given tau, fit for other parameters is linear, solved by pseudoinverse
        if D is None:
            D = self.regressors_array[phi_idx]
        if Dinv is None:
            Dinv = self.regressors_array_inv[phi_idx]
        dFF = np.expand_dims(dFF, axis=0)
        # coeffs = np.matmul( np.matmul(dFF,D.T), Dinv )
        coeffs = np.squeeze( (dFF@D.T) @ Dinv )
        dFF_fit = coeffs@D #np.matmul(coeffs,D)
        obj = ((dFF-dFF_fit)**2).sum()

        # split parameters 'coeffs' back into dictionary by labeled regressor
        if reg_labels is None:
            reg_labels = list(self.regressors_dict[phi_idx].keys())
        coeff_dict = {}
        cumulative_tot = 0
        for j in range(len(reg_labels)):
            n_this_reg = self.regressors_dict[phi_idx][ reg_labels[j] ].shape[0]
            # pdb.set_trace()
            coeff_dict[reg_labels[j]] = coeffs[ cumulative_tot+np.arange(n_this_reg) ]
            cumulative_tot += n_this_reg
        
        return coeff_dict, obj



    def fit_reg_linear_batch(self, phi, tseries_to_sub=None, D=None, Dinv=None, reg_labels=None):
        '''
        n=cell number
        phi=offset
        tseries_to_sub (optional): pre-subtract a timeseries. Useful for parameter sweeps
        D (optional): provide regressor array. Otherwise defaults to reg array attribute of current object
        Dinv (optional): provide regressor array. Otherwise defaults to inv reg array attribute of current object
        reg_labels (optional): required if providing D, unneeded otherwise. Manually provide fields for dict output
        '''
        dFF_full = self.data_dict[self.activity]
        
        # dFF slides past beh with displacement phi
        L = self.params['L']
        if(phi==L):
            dFF = dFF_full[:,L+phi:]
        else:
            dFF = dFF_full[:,L+phi:-(L-phi)]
        if tseries_to_sub is not None:
            dFF -= tseries_to_sub

        # for a given tau, fit for other parameters is linear, solved by pseudoinverse
        if D is None:
            D = self.regressors_array
        if Dinv is None:
            Dinv = self.regressors_array_inv
        # dFF = np.expand_dims(dFF, axis=0)
        # # coeffs = np.matmul( np.matmul(dFF,D.T), Dinv )
        coeffs = np.squeeze( (dFF@D.T) @ Dinv )
        dFF_fit = coeffs@D #np.matmul(coeffs,D)
        obj = ((dFF-dFF_fit)**2).sum(axis=1)

        # split parameters 'coeffs' back into dictionary by labeled regressor
        if reg_labels is None:
            reg_labels = list(self.regressors_dict.keys())
        coeff_dict = {}
        cumulative_tot = 0
        for j in range(len(reg_labels)):
            n_this_reg = self.regressors_dict[ reg_labels[j] ].shape[0]
            # pdb.set_trace()
            coeff_dict[reg_labels[j]] = coeffs[:, cumulative_tot+np.arange(n_this_reg) ]
            cumulative_tot += n_this_reg
        
        return coeff_dict, obj




    def get_regressors(self, phi_input=None, amplify_baseline=False, just_null_model=False):
        '''build matrix of all regressors.  Stores output as a dictionary and as a concatenated array.
        Also pre-computes inverse of this array.
        params = {'split_behav':True,
                    'M':M,
                    'L':L,
                    'tau':tau,
                    'mu':mu
                 }
        phi_input: bypasses default sweep through all phi. useful for eval step, ignore for other functions
        amplify_baseline: scale alpha and trial regs to spare them from elastic net penalty
        just_null_model: returns just alpha and trial regs''' 
        self.refresh_params()
        data_dict = self.data_dict
        params = self.params
        if just_null_model: params['split_behav'] = False
        L = params['L']
        time = data_dict['time']-data_dict['time'].mean() #data_dict['time'][0]
        ts_full = time #np.squeeze(time)
        t_exp = np.linspace(1,params['M'],params['M'])/data_dict['scanRate']


        # define trial flag regressors
        U = np.unique(data_dict['trialFlag'])
        NT = len(U)
        # print(str(NT)+' trials')
        trial_regressors = np.zeros((NT-1,-2*L+len(data_dict['trialFlag'])))
        # pdb.set_trace()
        for i in range(NT-1):
            is_this_trial = data_dict['trialFlag']==U[i]
            trial_regressors[i,:] = 1.0*np.squeeze(is_this_trial)[L:-L]
            trial_regressors[i,:] -= trial_regressors[i,:].mean()
            trial_regressors[i,:] /= abs(trial_regressors[i,:]).max()

        # convolve behavior with chosen kernel, either piecewise or all together
        ts = ts_full[L:-L]-ts_full[L:-L].mean()
        ts /= abs(ts).max()
        kern = np.exp(-0.5*((t_exp-params['mu'])/params['tau'])**2)
        # kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern /= kern.sum()
        if params['split_behav']:
            ball = np.zeros((NT,len(data_dict['behavior'])))
            x_c_full = np.zeros((NT,len(data_dict['behavior'])))
            for i in range(NT):
                is_this_trial = np.squeeze(data_dict['trialFlag']==U[i])
                ball[i,is_this_trial] = data_dict['behavior'][is_this_trial]-data_dict['behavior'][is_this_trial].mean()
                # x_c_full[i,:] = np.convolve(kern,ball[i,:],'same')                      # ******* fix 'same' and 'gauss/exp' option *******
                x_c_full[i,is_this_trial] = np.convolve(kern,ball[i,:],'same')[is_this_trial] 
        elif just_null_model:
            x_c_full = []
        else:
            ball = data_dict['behavior']-data_dict['behavior'].mean()
            x_c_full = np.convolve(kern,ball,'same')                      # ******* fix 'same' and 'gauss/exp' option *******
        
        ## legacy version
        # d1 = np.argmax(data_dict['drink'])
        # hunger = np.zeros(data_dict['drink'].shape)
        # hunger[d1:]=1
        # hunger = np.squeeze(hunger)-hunger.mean()
        # if abs(hunger).max()==0:
        #     hunger = np.nan*np.ones(hunger.shape)
        # else:
        #     hunger /= abs(hunger).max()
        d1 = np.argmax(data_dict['drink'])
        feed_raw = np.squeeze(data_dict['drink'].astype(float))
        feed_raw -= feed_raw.mean()
        if abs(feed_raw).max()==0:
            feed_raw = np.nan*np.ones(feed_raw.shape)
        else:
            feed_raw /= abs(feed_raw).max()
        feed_kern = np.exp(-t_exp/params['tau_feed'])
        feed_conv = np.convolve(feed_kern,feed_raw,'full')[:-len(feed_kern)+1]
        feed_conv[:d1-1] = feed_conv[d1-1] #erase conv artifacts
        feed_conv -= feed_conv.mean()
        feed_conv /= feed_conv.max()

        # alpha_regs = np.concatenate( (np.ones((1,len(ts)))/len(ts), ts.T) )
        alpha_regs = np.concatenate( (np.ones((1,len(ts)))/len(ts), ts.T, ts.T**2) )
        # regs = np.concatenate( (alpha_regs, x_c, trial_regressors, np.array([drink[L:-L],hunger[L:-L]])), axis=0 )
        
        if just_null_model:
            phi_list_for_loop = [0]
        elif phi_input is None:
            phi_list_for_loop = self.phiList
        else:
            phi_list_for_loop = [phi_input]
        self.regressors_dict = [None]*len(phi_list_for_loop)
        self.regressors_array = [None]*len(phi_list_for_loop)
        self.regressors_array_inv = [None]*len(phi_list_for_loop)

        for j in range(len(phi_list_for_loop)):
            phi = phi_list_for_loop[j]
            if params['split_behav']:
                x_c = np.zeros((NT,-2*params['L']+len(data_dict['behavior'])))
                for i in range(NT):
                    if(phi==L):
                        x_c[i,:] = x_c_full[i,L+phi:]-x_c_full[i,L+phi:].mean()
                    else:
                        x_c[i,:] = x_c_full[i,L+phi:-(L-phi)]-x_c_full[i,L+phi:-(L-phi)].mean()
                    x_c[i,:] /= abs(x_c[i,:]).max()
            elif just_null_model:
                x_c = []
            else:
                if(phi==L):
                    x_c = x_c_full[L+phi:]-x_c_full[L+phi:].mean()
                else:
                    x_c = x_c_full[L+phi:-(L-phi)]-x_c_full[L+phi:-(L-phi)].mean()
                x_c /= abs(x_c).max()
                x_c = np.expand_dims(x_c, axis=0)

            # scale alpha and trial regs to spare them from elastic net penalty
            if just_null_model:
                self.regressors_dict[j] = {
                    'alpha_01':alpha_regs, 
                    'trial':trial_regressors
                }
            elif amplify_baseline:
                self.regressors_dict[j] = {
                    'alpha_01':alpha_regs*self.baseline_regs_amp_val, 
                    'trial':trial_regressors*self.baseline_regs_amp_val, 
                    'beta_0':x_c 
                }
            else:
                self.regressors_dict[j] = {
                    'alpha_01':alpha_regs, 
                    'trial':trial_regressors, 
                    'beta_0':x_c 
                }
            # if (np.isnan(drink).sum()==0) and (np.isnan(hunger).sum()==0) and not just_null_model:
            #     self.regressors_dict[j]['drink_hunger'] = np.array([drink[L:-L],hunger[L:-L]])
            if (np.isnan(feed_conv).sum()==0) and not just_null_model:
                self.regressors_dict[j]['drink_hunger'] = np.expand_dims(feed_conv[L:-L], axis=0) #np.array(feed_conv[L:-L])

            # pdb.set_trace()
            regressors_array, regressors_array_inv, self.tot_n_regressors = self.get_regressor_array(self.regressors_dict[j])
            if (regressors_array.max()>self.baseline_regs_amp_val) or (regressors_array.min()<-self.baseline_regs_amp_val):
                print('WARNING: regressor not normalized')
            self.regressors_array[j] = regressors_array 
            self.regressors_array_inv[j] = regressors_array_inv


    def get_regressor_array(self, regs):
        reg_labels = list(regs.keys())
        regressors_array = np.concatenate( (regs[reg_labels[0]], regs[reg_labels[1]]), axis=0 )
        for j in range(2,len(reg_labels)):
            regressors_array = np.concatenate( (regressors_array, regs[reg_labels[j]]), axis=0 )
        # regressors_array = np.concatenate( ( regs['alpha_01'], regs['beta_0'], regs['trial'], regs['drink_hunger'] ), axis=0 )
        regressors_array_inv = np.linalg.inv( regressors_array@regressors_array.T )
        tot_n_regressors = regressors_array.shape[0]
        return regressors_array, regressors_array_inv, tot_n_regressors

        
    def dict_to_flat_list(self, my_dict):
        # regenerate concatenated coefficient array
        reg_labels = list(my_dict.keys())
        my_array_nested = []
        for j in range(len(reg_labels)):
            my_array_nested.append( my_dict[ reg_labels[j] ] )
        my_array = [val for sublist in my_array_nested for val in sublist]
        return my_array

    def coeff_dict_from_keys(self, idx):
        # regenerate coeff dict from model_fit and regressor dict
        reg_labels = list(self.regressors_dict[0].keys())
        reg_dict = {}
        for j in range(len(reg_labels)):
            reg_dict[reg_labels[j]] = self.model_fit[idx][reg_labels[j]]
        return reg_dict


    def evaluate_reg_model_extended(self):
        #, n, coeff_dict, phi):
        # regenerate fit from best parameters and evaluate model
        self.refresh_params()
        null_self = copy.deepcopy(self)
        print('evaluating ')
        # pdb.set_trace()
        for n in range(self.data_dict[self.activity].shape[0]):
            if not np.mod(n,round(self.data_dict[self.activity].shape[0]/10)): print(n, end=' ')
            sys.stdout.flush()
            if self.model_fit[n]['success']:
                # pdb.set_trace()
                L = self.params['L']
                dFF_full = self.data_dict[self.activity][n,:]
                dFF = dFF_full[L:-L]
                dFF = np.expand_dims(dFF, axis=0)

                # for a given tau, fit for other parameters is linear, solved by pseudoinverse
                self.params['tau'] = self.model_fit[n]['tau']
                self.params['tau_feed'] = self.model_fit[n]['tau_feed']
                # phi_idx = np.argmin(abs(self.model_fit[n]['phi']-self.phiList))
                if self.elasticNet:
                    self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=True)
                else:
                    self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=False)

                D = self.regressors_array[0] #array len()=1 here
                
                # regenerate concatenated coefficient array
                coeff_dict = self.coeff_dict_from_keys(idx=n)
                coeff_array = self.dict_to_flat_list(coeff_dict)
                reg_labels = list(coeff_dict.keys())
                
                # pdb.set_trace()
                dFF_fit = coeff_array@D
                SS_res = ( (dFF-dFF_fit)**2 )
                # res_var_tot = (dFF-dFF_fit).var()
                SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
                r_sq_tot = 1-SS_res.sum()/SS_tot.sum()
                
                # for all parameter categories, for all parameters in this category, find fit without this parameter to compute p_val
                stat = copy.deepcopy(coeff_dict)
                r_sq = copy.deepcopy(coeff_dict)
                cc = copy.deepcopy(coeff_dict)
                r_sq['tot'] = r_sq_tot
                for i in range(len(reg_labels)):
                    stat_list = []
                    r_sq_list = []
                    cc_list = []
                    for j in range(coeff_dict[reg_labels[i]].shape[0]):
                        j_inc = [x for x in range(coeff_dict[reg_labels[i]].shape[0]) if x != j]
                        reg_null = copy.deepcopy(self.regressors_dict[0])
                        reg_null[reg_labels[i]] = reg_null[reg_labels[i]][j_inc,:]
                        null_self.regressors_dict = [reg_null]
                        
                        # stats and r^2 are computed from *new* fit leaving out a regressor
                        regressors_array, regressors_array_inv, _ = self.get_regressor_array(reg_null) # ****** this should be null_self, eliminate arguments
                        null_self.regressors_array = [regressors_array]
                        null_self.regressors_array_inv = [regressors_array_inv]
                        p_n_dict, _ = null_self.fit_reg_linear(n=n, phi_idx=0) #phi_idx=0 because len(regs)=1
                        p_n = self.dict_to_flat_list(p_n_dict)
                        dFF_fit_null = np.squeeze(p_n@null_self.regressors_array[0])   
                        SS_res_0 = ( (dFF-dFF_fit_null)**2 )
                        # res_var_0 = (dFF-dFF_fit_null).var()
                        stat_list.append( stats.wilcoxon(np.squeeze(SS_res_0),np.squeeze(SS_res)) )
                        r_sq_list.append( (SS_res_0.sum()-SS_res.sum())/SS_res.sum() )  # var explained relative to residual, not total variance
                        
                        # cc is computed from *old* fit leaving out regressor
                        coeff_dict_0 = copy.deepcopy(coeff_dict)
                        coeff_dict_0[reg_labels[i]] = coeff_dict_0[reg_labels[i]][j_inc]
                        coeff_0 = self.dict_to_flat_list(coeff_dict_0)
                        dFF_fit_null_0 = np.squeeze(coeff_0@null_self.regressors_array[0]) 
                        # pdb.set_trace()

                        norm_resid = dFF - dFF_fit_null_0 - (dFF-dFF_fit_null_0).mean()
                        norm_reg = self.regressors_dict[0][reg_labels[i]][j,:].copy()
                        norm_reg -= norm_reg.mean()
                        cc_list.append( (norm_resid*norm_reg).mean()/(norm_resid.std()*norm_reg.std()) )
                        if r_sq_list[-1]<0:
                            print('shit')
                            # pdb.set_trace()
                    stat[reg_labels[i]] = stat_list
                    r_sq[reg_labels[i]] = r_sq_list
                    cc[reg_labels[i]] = cc_list
                
                stat['tau'] = stat['beta_0']
                stat['phi'] = stat['beta_0']    
                self.model_fit[n]['r_sq'] = r_sq #1-SS_res.sum()/SS_tot.sum()
                self.model_fit[n]['stat'] = stat
                self.model_fit[n]['cc'] = cc
            else:
                self.model_fit[n]['r_sq'] = None
                self.model_fit[n]['stat'] = None
                self.model_fit[n]['cc'] = None



    def fit_reg_model_MLE(self, elasticNetParams={'alpha':0.01,'l1_ratio':0.1}):
        # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
        # this is an extension of ".taulist_shiftlist_gauss", with two additions:
        #       (1) merges gauss/exp kernel methods with option for either
        #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
        
        self.refresh_params()
        data_dict = self.data_dict
        tauList = self.tauList.copy()
        tauList_feed = self.tauList_feed.copy()
        tau_star = np.zeros(data_dict[self.activity].shape[0])
        tau_feed_star = np.zeros(data_dict[self.activity].shape[0])
        phi_star = np.zeros(data_dict[self.activity].shape[0])
        fn_min = np.inf*np.ones(data_dict[self.activity].shape[0])
        
        if self.elasticNet:
            elasticNet_obj = ElasticNet(alpha=elasticNetParams['alpha'], l1_ratio=elasticNetParams['l1_ratio'], fit_intercept=True, warm_start=True)

        # fit model -  for each value of tau and phi, check if pInv solution is better than previous
        P = [None]*data_dict[self.activity].shape[0] # [] #np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
        for ii in range(len(tauList_feed)):
            self.params['tau_feed'] = tauList_feed[ii]
            for i in range(len(tauList)):
                if not np.mod(i,2): print(i, end=' ')
                sys.stdout.flush()
                self.params['tau'] = tauList[i]
                if self.elasticNet:
                    self.get_regressors(amplify_baseline=True)
                else:
                    self.get_regressors(amplify_baseline=False)
                 
                for j in range(len(self.phiList)):
                    if np.sum(np.isnan(self.regressors_array[j]))>0: continue
                    for n in range(data_dict[self.activity].shape[0]):
                        if self.elasticNet:
                            p, obj = self.fit_reg_linear_elasticNet(n=n, phi_idx=j, elasticNet_obj=elasticNet_obj)
                        else:
                            # if not elasticNet, do OLS
                            p, obj = self.fit_reg_linear(n=n, phi_idx=j)
                        if (obj<fn_min[n]):
                            tau_star[n] = tauList[i]
                            tau_feed_star[n] = tauList_feed[ii]
                            phi_star[n] = self.phiList[j]
                            P[n] = p
                            fn_min[n] = obj
                    
        # collect output ********* redo with dictionary defined at outset to eliminate need for this ---------
        self.model_fit = []
        for n in range(data_dict[self.activity].shape[0]):
            # if not self.model_fit:
            if P[n] is not None:
                d = copy.deepcopy(P[n])
                d['tau'] = tau_star[n]
                d['tau_feed'] = tau_feed_star[n]
                d['phi'] = int(phi_star[n])
                d['success'] = True #res['success']
            else:
                d = {'success':False}
            d['activity'] = self.activity
            d['kern'] = 'gauss'
            self.model_fit.append(d)
            # else:
            #     self.model_fit[n]['tau'] = tau_star[n]
            #     self.model_fit[n]['phi'] = int(phi_star[n])
            #     self.model_fit[n]['success'] = True #res['success']




    def fit_reg_linear_elasticNet(self, n, phi_idx, elasticNet_obj, tseries_to_sub=None, D=None, Dinv=None, reg_labels=None):
        '''
        n=cell number
        phi=offset
        tseries_to_sub (optional): pre-subtract a timeseries. Useful for parameter sweeps
        D (optional): provide regressor array. Otherwise defaults to reg array attribute of current object
        Dinv (optional): provide regressor array. Otherwise defaults to inv reg array attribute of current object
        reg_labels (optional): required if providing D, unneeded otherwise. Manually provide fields for dict output
        '''
        dFF_full = self.data_dict[self.activity][n,:]
        L = self.params['L']
        dFF = dFF_full[L:-L]
        if tseries_to_sub is not None:
            dFF -= tseries_to_sub

        # for a given tau, fit for other parameters is linear, solved by pseudoinverse
        if D is None:
            D = self.regressors_array[phi_idx]
        
        dFF = np.expand_dims(dFF, axis=0)
        enf = elasticNet_obj.fit(D.T, dFF.T)
        coeffs = enf.coef_
        dFF_fit = elasticNet_obj.predict(D.T)
        #coeffs = np.squeeze( (dFF@D.T) @ Dinv )
        #dFF_fit = coeffs@D #np.matmul(coeffs,D)
        obj = ((dFF-dFF_fit)**2).sum()

        # split parameters 'coeffs' back into dictionary by labeled regressor
        if reg_labels is None:
            reg_labels = list(self.regressors_dict[phi_idx].keys())
        coeff_dict = {}
        cumulative_tot = 0
        for j in range(len(reg_labels)):
            n_this_reg = self.regressors_dict[phi_idx][ reg_labels[j] ].shape[0]
            # pdb.set_trace()
            coeff_dict[reg_labels[j]] = coeffs[ cumulative_tot+np.arange(n_this_reg) ]
            cumulative_tot += n_this_reg
        coeff_dict['alpha_01'][0] = elasticNet_obj.intercept_/D[0,0] # fixes compatibility with get_dFF_fit method in this library

        
        return coeff_dict, obj



    def fit_reg_model_full_likelihood(self):
        # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
        # this is an extension of ".taulist_shiftlist_gauss", with two additions:
        #       (1) merges gauss/exp kernel methods with option for either
        #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
        
        self.refresh_params()
        data_dict = self.data_dict
        tauList = self.tauList
        phiList = self.phiList
        norm_coeff_list = np.linspace(-1,1,num=3) # ********** change this to 21
        
        tau_star = np.zeros(data_dict[self.activity].shape[0])
        phi_star = np.zeros(data_dict[self.activity].shape[0])
        fn_min = np.inf*np.ones(data_dict[self.activity].shape[0])
        
        # # check how many regressors are being used (** consider redoing without this **)
        self.get_regressors() 
        # initialize P (list of dicts)
        P = [None]*data_dict[self.activity].shape[0] # [] #np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
        self.P_tot = {'alpha_01': np.zeros( ( data_dict[self.activity].shape[0],len(tauList),len(phiList),
                            len(norm_coeff_list),len(norm_coeff_list),
                            len(norm_coeff_list),len(norm_coeff_list),
                            self.regressors_dict['alpha_01'].shape[0] ) ), 
                'trial': np.zeros( ( data_dict[self.activity].shape[0],len(tauList),len(phiList),
                            len(norm_coeff_list),len(norm_coeff_list),
                            len(norm_coeff_list),len(norm_coeff_list),
                            self.regressors_dict['trial'].shape[0] ) )}
        self.obj_tot = np.zeros( ( data_dict[self.activity].shape[0],len(tauList),len(phiList),
                            len(norm_coeff_list),len(norm_coeff_list),
                            len(norm_coeff_list),len(norm_coeff_list) ) )
        
        for i in range(len(tauList)):
            if not np.mod(i,2): print(i, end=' ')
            self.params['tau'] = tauList[i]
            self.get_regressors() 
            
            for j in range(len(phiList)):
                # for n in range(data_dict['rate'].shape[0]):
                    sweep_param_list = [['drink_hunger',0],['drink_hunger',1],['beta_0',0],['beta_0',1]]
                    tot_sweep_params = 4
                    reg_to_sub = np.zeros( (tot_sweep_params, len(self.regressors_array[0,:])) )
                    sweep_regs = {
                        'alpha_01':self.regressors_dict['alpha_01'], 
                        'trial':self.regressors_dict['trial'], 
                    } 
                    sweep_regs_array, sweep_regs_array_inv, _ = self.get_regressor_array(sweep_regs)

                    for k0 in range(len(norm_coeff_list)):
                        reg_to_sub[0,:] = norm_coeff_list[k0]*self.regressors_dict[ sweep_param_list[0][0] ][sweep_param_list[0][1], :]
                        for k1 in range(len(norm_coeff_list)):
                            reg_to_sub[1,:] = norm_coeff_list[k1]*self.regressors_dict[ sweep_param_list[1][0] ][sweep_param_list[1][1], :]
                            for k2 in range(len(norm_coeff_list)):
                                reg_to_sub[2,:] = norm_coeff_list[k2]*self.regressors_dict[ sweep_param_list[2][0] ][sweep_param_list[2][1], :]
                                for k3 in range(len(norm_coeff_list)):
                                    if self.regressors_dict[ sweep_param_list[3][0] ].shape[0]>1:
                                        # if this parameter exists (handles beta_0 case)
                                        reg_to_sub[3,:] = norm_coeff_list[k3]*self.regressors_dict[ sweep_param_list[3][0] ][sweep_param_list[3][1], :]
                                    else:
                                        reg_to_sub[3,:] = 0

                                    
                                    p_tmp, obj_tmp = self.fit_reg_linear_batch(phi=phiList[j],
                                                    tseries_to_sub=reg_to_sub.sum(axis=0), D=sweep_regs_array, 
                                                    Dinv=sweep_regs_array_inv, reg_labels=['alpha_01','trial'])

                                    self.P_tot['alpha_01'][:,i,j,k0,k1,k2,k3,:] = p_tmp['alpha_01']
                                    self.P_tot['trial'][:,i,j,k0,k1,k2,k3,:] = p_tmp['trial']
                                    self.obj_tot[:,i,j,k0,k1,k2,k3] = obj_tmp

                            # pdb.set_trace()

    def normalize_rows(self,input_mat,quantiles=(.01,.99),fix_nans=True): 
        output_mat = np.zeros(input_mat.shape)
        q0 = np.zeros(input_mat.shape[0])
        q1 = np.zeros(input_mat.shape[0])
        for i in range(input_mat.shape[0]):
            y = input_mat[i,:].copy()
            q0[i] = np.quantile(y,quantiles[0])
            q1[i] = np.quantile(y,quantiles[1])
            y[y<q0[i]] = q0[i]
            y[y>q1[i]] = q1[i]
            y -= q0[i]
            y /= (q1[i]-q0[i])
            output_mat[i,:] = y
            if fix_nans:
                output_mat[i,:][np.isnan(output_mat[i,:])]=0
        return output_mat, q0, q1


    def unnormalize_rows(self,input_mat,q0,q1): 
        # then integrate into data loading in _hunger notebook
        # then integrate into use of residual raster to plot reinflated residuals
        # then set everything to run overnight

        # inverts "normalize_rows", except for cropped extrema
        output_mat = np.zeros(input_mat.shape)
        for i in range(input_mat.shape[0]):
            output_mat[i,:] = input_mat[i,:]*(q1[i]-q0[i]) + q0[i]
        return output_mat
    

    def refresh_params(self):
        data_dict = self.data_dict
        sigLimSec = self.params['sigLimSec'] #100
        phaseLimSec = self.params['phaseLimSec'] #20
        sigLim = sigLimSec*data_dict['scanRate']
        self.params['L'] = int(phaseLimSec*data_dict['scanRate'])
        self.params['M'] = round(-sigLim*np.log(0.1)) #.astype(int)
        self.params['mu'] = .5*self.params['M']/data_dict['scanRate']
        self.phiList = np.linspace(-self.params['L'],self.params['L'], num=2*phaseLimSec+1 ).astype(int)
        self.tauList = np.logspace(-1,np.log10(sigLimSec),num=60)
        self.tauList_feed = np.logspace(-1,np.log10(sigLimSec),num=self.num_tau_feed_steps) #self.tauList.copy()
                

    def get_dFF_fit(self, n):
        self.refresh_params()
        dFF_full = self.data_dict[self.activity][n,:]
        dFF = dFF_full[self.params['L']:-self.params['L']]
        dFF = np.expand_dims(dFF, axis=0)
        self.params['tau'] = self.model_fit[n]['tau']
        self.params['tau_feed'] = self.model_fit[n]['tau_feed']
        if self.elasticNet:
            self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=True)
        else:
            self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=False)
        coeff_dict = self.coeff_dict_from_keys(idx=n)
        coeff_array = self.dict_to_flat_list(coeff_dict)
        dFF_fit = coeff_array@self.regressors_array[0]
        return dFF_fit, dFF


    def get_null_subtracted_raster(self, extra_regs_to_use=None, just_null_model=False):
        """ default is to return fit with just alpha and trial regs. 
        Using extra_regs, option to return fit with any additional regs: [ ['partial_reg', idx], ['full_reg'] ] 
        extra_regs are the regressors TO SUBTRACT """
        self.refresh_params()
        dFF_full = self.data_dict[self.activity]
        dFF = dFF_full[:,self.params['L']:-self.params['L']]
        sl = slice(self.params['L'], dFF_full.shape[1]-self.params['L'])
        # dFF = np.expand_dims(dFF, axis=0)
        dFF_fit = np.zeros(dFF.shape)
        for n in range(dFF.shape[0]):
            if self.model_fit[n]['success']:
                self.params['tau'] = self.model_fit[n]['tau']
                self.params['tau_feed'] = self.model_fit[n]['tau_feed']
                if self.elasticNet:
                    self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=True, just_null_model=just_null_model)
                else:
                    self.get_regressors(phi_input=self.model_fit[n]['phi'], amplify_baseline=False, just_null_model=just_null_model)
                coeff_dict = self.coeff_dict_from_keys(idx=n)
                reg_labels = list(coeff_dict.keys())
                # make null dict by setting params of interest to 0
                for j in range(len(reg_labels)):
                    if (reg_labels[j]=='alpha_01') or (reg_labels[j]=='trial'): continue
                    if extra_regs_to_use is not None:
                        tmp_reg = np.zeros(coeff_dict[reg_labels[j]].shape)
                        for sublist in extra_regs_to_use:
                            if reg_labels[j] in sublist: 
                                if len(sublist) == 1: 
                                    continue # use all elements of this regressor
                                else:
                                    el_to_zero = [z for z in range(len(coeff_dict[reg_labels[j]])) if z != sublist[1]]
                                    coeff_dict[reg_labels[j]][el_to_zero] = 0 #use some elements of this regressor
                    else:
                        coeff_dict[reg_labels[j]] = np.zeros(coeff_dict[reg_labels[j]].shape)
                coeff_array = self.dict_to_flat_list(coeff_dict)
                dFF_fit[n,:] = coeff_array@self.regressors_array[0]
        return dFF_fit, dFF, sl


    def get_smooth_behavior(self, split_behav=False):
        from skimage.restoration import denoise_tv_chambolle

        # filter behavior with TV denoising. requires customized settings for seome datasets
        # pdb.set_trace()
        if not self.expt_id: 
            print('***** warning: Missing exp_id *******')

        tv_params = [.01,.9,.05] #np.array([.01,.9,.05])
        if self.expt_id=='2018_08_24_fly3_run1':
            tv_params[0] = 0.2
            tv_params[2] = 0.01
        elif self.expt_id=='2018_08_24_fly2_run2':
            tv_params[2]=0.01
        elif self.expt_id=='2019_07_01_fly2':
            [] #ok
        elif self.expt_id=='2019_10_14_fly3':
            tv_params[0]=0.05
            tv_params[2]=0.01
        elif self.expt_id=='2019_06_28_fly2':
            tv_params[0]=0.1
        elif self.expt_id=='2019_06_30_fly1':
            [] #ok
        elif self.expt_id=='2019_10_14_fly2':
            tv_params[0]=0.1
        elif self.expt_id=='2019_10_14_fly4':
            tv_params[0]=0.05
        elif self.expt_id=='2019_10_18_fly3':
            tv_params[0]=0.05
        elif self.expt_id=='2019_10_21_fly1':
            [] #ok
        elif self.expt_id=='2019_10_10_fly3':
            tv_params[0]=0.05
        elif self.expt_id=='2019_10_02_fly2':
            tv_params[0]=0.1
        elif self.expt_id=='2019_08_14_fly1':
            [] #ok
        elif self.expt_id=='2019_04_18_fly2':
            tv_params[0]=[.01, 0.12, 0.01]
        elif self.expt_id=='2019_04_22_fly1':
            tv_params[0]=[.01, 0.2, 0.01, 0.01]
        elif self.expt_id=='2019_04_22_fly3':
            [] #ok
        elif self.expt_id=='2019_04_24_fly3':
            [] #ok
        elif self.expt_id=='2019_04_24_fly1':
            [] #ok
        elif self.expt_id=='2019_04_25_fly3':
            tv_params[2]=[.05, .01, .01]
        elif self.expt_id=='2019_05_07_fly1':
            tv_params[2]=[.05, .01, .01]
        elif self.expt_id=='2019_03_12_fly4':
            tv_params[0]=0.15
            tv_params[2]=0.01
        elif self.expt_id=='2019_02_19_fly1':
            tv_params[0]=0.2
            tv_params[2]=0.01
        elif self.expt_id=='2019_02_26_fly1_2':
            tv_params[0]=0.2
            tv_params[2]=0.01
        else:
            print('**** warning: no exp_id match ****')
        
        if split_behav:
            trial_flag = self.data_dict['trialFlag']
            U = np.unique(trial_flag)
            NT = len(U)
            for i in range(NT):
                p=[0,0,0]
                for j in range(3):
                    if isinstance(tv_params[j], list):
                        p[j]=tv_params[j][i]
                    else:
                        p[j]=tv_params[j]

                is_this_trial = np.squeeze(trial_flag==U[i])
                beh = self.data_dict['behavior'][is_this_trial]
                m = beh.min()
                M = beh.max()
                l = m+p[0]*(M-m)
                h = m+p[1]*(M-m)
                beh[beh<l]=l
                beh[beh>h]=h
                beh -= l
                self.data_dict['behavior'][is_this_trial] = denoise_tv_chambolle(beh, weight=p[2])
        else:
            beh = self.data_dict['behavior']
            m = beh.min()
            M = beh.max()
            l = m+tv_params[0]*(M-m)
            h = m+tv_params[1]*(M-m)
            beh[beh<l]=l
            beh[beh>h]=h
            beh -= l
            self.data_dict['behavior'] = denoise_tv_chambolle(beh, weight=tv_params[2])


    def estimate_motion_artifacts(self, inv_cv_thresh=1.0, max_dRR_thresh=0.3, make_hist=False):
        self.inv_cv_thresh = inv_cv_thresh
        self.max_dRR_thresh = max_dRR_thresh
        self.get_regressors(just_null_model=True)
        self.data_dict['dFF'] = self.data_dict['dRR'].copy()
        self.model_fit = []
        for n in range(self.data_dict['dRR'].shape[0]):
            # pdb.set_trace()
            p, _ = self.fit_reg_linear(n=n, phi_idx=0)
            d = copy.deepcopy(p)
            d['tau'] = 1
            d['tau_feed'] = 1
            d['phi'] = 0
            d['success'] = True
            d['activity'] = 'dFF'
            d['kern'] = 'gauss'
            self.model_fit.append(d)    
        R0, dFF, _ = self.get_null_subtracted_raster(just_null_model=True)
        dRR0 = dFF-R0
        v=dRR0.var(axis=1)
        m=R0.mean(axis=1)
        self.mag = dRR0.max(axis=1)-dRR0.min(axis=1)
        self.motion_cv_inv = v/m**2
        self.motion_cv_inv[np.isnan(self.motion_cv_inv)] = np.inf
        if make_hist:
            cv = self.motion_cv_inv.copy()
            plt.figure(figsize=(7,3))
            cv[cv>2*inv_cv_thresh]=2*inv_cv_thresh
            plt.hist(cv,50)
            yl = plt.ylim()
            plt.plot([inv_cv_thresh,inv_cv_thresh], yl, 'r--')
            plt.xlim(-0.01,0.01+2*inv_cv_thresh)
            plt.xlabel(r'CV$^{-1}$ (red)')
            plt.ylabel('count')
            plt.tight_layout()
        return dRR0, R0



    def fit_and_eval_reg_model_extended(self):
        self.fit_reg_model_MLE()
        self.evaluate_reg_model_extended()

