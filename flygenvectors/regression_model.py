import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
import copy
import pdb



class reg_obj:
    def __init__(self):
        self.data_dict = {}
        self.params = {'split_behav':True,
                    'sigLimSec':100,
                    'phaseLimSec': 20,
                    'M':0,
                    'L':0,
                    'tau':0.1,
                    'mu':0
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
        self.phi = 0
        self.p_dict = {} # temp
        self.model_fit = []


    def fit_reg_linear(self, n, phi):
        dFF_full = self.data_dict['rate'][n,:]
        
        # dFF slides past beh with displacement phi
        L = self.params['L']
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]

        # for a given tau, fit for other parameters is linear, solved by pseudoinverse
        D = self.regressors_array
        Dinv = self.regressors_array_inv
        dFF = np.expand_dims(dFF, axis=0)
        # coeffs = np.matmul( np.matmul(dFF,D.T), Dinv )
        coeffs = np.squeeze( (dFF@D.T) @ Dinv )
        dFF_fit = coeffs@D #np.matmul(coeffs,D)
        obj = ((dFF-dFF_fit)**2).sum()

        # split parameters 'coeffs' back into dictionary by labeled regressor
        reg_labels = list(self.regressors_dict.keys())
        coeff_dict = {}
        cumulative_tot = 0
        for j in range(len(reg_labels)):
            n_this_reg = self.regressors_dict[ reg_labels[j] ].shape[0]
            # pdb.set_trace()
            coeff_dict[reg_labels[j]] = coeffs[ cumulative_tot+np.arange(n_this_reg) ]
            cumulative_tot += n_this_reg
        
        return coeff_dict, obj



    def get_regressors(self):
        '''build matrix of all regressors.  Stores output as a dictionary and as a concatenated array.
        Also pre-computes inverse of this array.
        params = {'split_behav':True,
                    'M':M,
                    'L':L,
                    'tau':tau,
                    'mu':mu
                 }''' 
        data_dict = self.data_dict
        params = self.params
        L = params['L']
        time = data_dict['time']-data_dict['time'].mean() #data_dict['time'][0]
        ts_full = time #np.squeeze(time)
        t_exp = np.linspace(1,params['M'],params['M'])/data_dict['scanRate']
        d1 = np.argmax(data_dict['drink'])
        hunger = np.zeros(data_dict['drink'].shape)
        hunger[d1:]=1
        hunger = np.squeeze(hunger)-hunger.mean()
        drink = np.squeeze(data_dict['drink'].astype(float))
        drink -= drink.mean()

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

        # convolve behavior with chosen kernel, either piecewise or all together
        ts = ts_full[L:-L]-ts_full[L:-L].mean()
        kern = np.exp(-0.5*((t_exp-params['mu'])/params['tau'])**2)
        # kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern /= kern.sum()
        if params['split_behav']:
            ball = np.zeros((NT,len(data_dict['behavior'])))
            x_c = np.zeros((NT,-2*params['L']+len(data_dict['behavior'])))
            for i in range(NT):
                is_this_trial = np.squeeze(data_dict['trialFlag']==U[i])
                ball[i,is_this_trial] = data_dict['behavior'][is_this_trial]-data_dict['behavior'][is_this_trial].mean()
                x_c_full = np.convolve(kern,ball[i,:],'same')                      # ******* fix 'same' and 'gauss/exp' option *******
                x_c[i,:] = x_c_full[L:-L]
        else:
            ball = data_dict['behavior']-data_dict['behavior'].mean()
            x_c_full = np.convolve(kern,ball,'same')                      # ******* fix 'same' and 'gauss/exp' option *******
            x_c = x_c_full[L:-L]
        
        alpha_regs = np.concatenate( (np.ones((1,len(ts))), ts.T) )
        # pdb.set_trace()
        # regs = np.concatenate( (alpha_regs, x_c, trial_regressors, np.array([drink[L:-L],hunger[L:-L]])), axis=0 )
        regs = {
            'alpha_01':alpha_regs, 
            'beta_0':x_c, 
            'trial':trial_regressors, 
            'drink_hunger':np.array([drink[L:-L],hunger[L:-L]])
        } 
        # pdb.set_trace()
        self.regressors_dict = regs
        self.regressors_array, self.regressors_array_inv, self.tot_n_regressors = self.get_regressor_array(regs)


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
        reg_labels = list(self.regressors_dict.keys())
        reg_dict = {}
        for j in range(len(reg_labels)):
            reg_dict[reg_labels[j]] = self.model_fit[idx][reg_labels[j]]
        return reg_dict


    def evaluate_reg_model_extended(self):
        #, n, coeff_dict, phi):
        # regenerate fit from best parameters and evaluate model
        null_self = copy.deepcopy(self)
        print('evaluating ')
        for n in range(self.data_dict['rate'].shape[0]):
            if not np.mod(n,round(self.data_dict['rate'].shape[0]/10)): print(n, end=' ')
            self.params['tau'] = self.model_fit[n]['tau']
            self.get_regressors() 

            dFF_full = self.data_dict['rate'][n,:]
            # dFF slides past beh with displacement phi
            phi = self.model_fit[n]['phi']
            L = self.params['L']
            if(phi==L):
                dFF = dFF_full[L+phi:]
            else:
                dFF = dFF_full[L+phi:-(L-phi)]

            # for a given tau, fit for other parameters is linear, solved by pseudoinverse
            D = self.regressors_array
            dFF = np.expand_dims(dFF, axis=0)

            # regenerate concatenated coefficient array
            coeff_dict = self.coeff_dict_from_keys(idx=n)
            coeff_array = self.dict_to_flat_list(coeff_dict)
            reg_labels = list(coeff_dict.keys())

            # pdb.set_trace()
            dFF_fit = coeff_array@D
            SS_res = ( (dFF-dFF_fit)**2 )
            SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
            r_sq = 1-SS_res.sum()/SS_tot.sum()
            
            # for all parameter categories, for all parameters in this category, find fit without this parameter to compute p_val
            stat = copy.deepcopy(coeff_dict)
            
            for i in range(len(reg_labels)):
                stat_list = []
                for j in range(coeff_dict[reg_labels[i]].shape[0]):
                    j_inc = [x for x in range(coeff_dict[reg_labels[i]].shape[0]) if x != j]
                    reg_null = copy.deepcopy(self.regressors_dict)
                    reg_null[reg_labels[i]] = reg_null[reg_labels[i]][j_inc,:]
                    null_self.regressors_dict = reg_null
                    null_self.regressors_array, null_self.regressors_array_inv, _ = self.get_regressor_array(reg_null) # ****** this should be null_self, eliminate arguments
                    p_n_dict, _ = null_self.fit_reg_linear(n=n, phi=phi)
                    p_n = self.dict_to_flat_list(p_n_dict)
                    dFF_fit_null = np.squeeze(p_n@null_self.regressors_array)   
                    SS_res_0 = ( (dFF-dFF_fit_null)**2 )
                    # pdb.set_trace()
                    stat_list.append( stats.wilcoxon(np.squeeze(SS_res_0),np.squeeze(SS_res)) )
                stat[reg_labels[i]] = stat_list
            
            stat['tau'] = stat['beta_0']
            stat['phi'] = stat['beta_0']    
            self.model_fit[n]['r_sq'] = r_sq #1-SS_res.sum()/SS_tot.sum()
            self.model_fit[n]['stat'] = stat



    def flag_cells_by_model_params(self, param_dict, dict_tot, eval_tot):
        print('update/deprecate')
        # # for each dataset, flag candidates
        # # typically used by passing bayes_model_tot_post as dict_tot
        # # and model_eval_tot_post as eval_tot
        
        # # generate distance threshold
        # from scipy.stats import multivariate_normal
        # data_dict = dict_tot['data_tot'][0]['data_dict']
        # tmp = multivariate_normal(dict_tot['data_tot'][0]['data_dict']['aligned_centroids'][0,:], param_dict['sig'] ) 
        # ref_pt = dict_tot['data_tot'][0]['data_dict']['aligned_centroids'][0,:].copy()
        # ref_pt[0] += param_dict['sig']
        # ref_val = tmp.pdf(ref_pt)

        # example_array_tot = []
        # # mag_order_tot = []
        # p_th = 10**-5
        # for nf in range(len(dict_tot['data_tot'])):
        #     data_dict = dict_tot['data_tot'][nf]['data_dict']
        #     model_fit = dict_tot['model_fit'][nf]
            
        #     N = data_dict['aligned_centroids'].shape[0]
        #     ex_ar = []
        #     for n in range(N):
        #         if ( (model_fit[n]['tau']>param_dict['tau_th']) and (model_fit[n]['phi']>param_dict['phi_th']) 
        #             and (model_fit[n]['tau']<param_dict['tau_max_th']) and (eval_tot[nf][n]['stat'][1]<p_th)):
        #             mvn = multivariate_normal(data_dict['aligned_centroids'][n,:], param_dict['sig'] )
        #             if mvn.pdf([param_dict['ref_x'],param_dict['ref_y'],param_dict['ref_z']])>ref_val:
        #                 ex_ar.append(n)
        #     example_array_tot.append( ex_ar )
        #     print(len(ex_ar),end=' ')
            
        #     # # get dynamic range of fits, and sort
        #     # mag = np.zeros(len(ex_ar))
        #     # for j in range(len(ex_ar)):
        #     #     dFF_fit, dFF = get_model_fit_gauss(data_dict, model_fit, ex_ar[j])
        #     #     mag[j] = dFF_fit.max() - dFF_fit.min()
        #     # mag_order = np.argsort(mag)
        #     # mag_order_tot.append( mag_order )
                
        # return example_array_tot   



    def fit_reg_model_extended(self):
        # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
        # this is an extension of ".taulist_shiftlist_gauss", with two additions:
        #       (1) merges gauss/exp kernel methods with option for either
        #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
        

        # extraRegressors = opts['extraRegressors']
        # tot_n_regressors = 3+extraRegressors.shape[0]
        data_dict = self.data_dict
        sigLimSec = self.params['sigLimSec'] #100
        phaseLimSec = self.params['phaseLimSec'] #20
        # use_beta_1 = False
        sigLim = sigLimSec*data_dict['scanRate']
        self.params['L'] = int(phaseLimSec*data_dict['scanRate'])
        self.params['M'] = round(-sigLim*np.log(0.1)).astype(int)
        self.params['mu'] = .5*self.params['M']/data_dict['scanRate']
        # t_exp = np.linspace(1,M,M)/data_dict['scanRate']
        # time = data_dict['time']-data_dict['time'].mean() #data_dict['time'][0]
        # ts_full = np.squeeze(time)
        tauList = np.logspace(-1,np.log10(sigLimSec),num=100)
        phiList = np.linspace(-self.params['L'],self.params['L'], num=2*phaseLimSec+1 ).astype(int)
        
        tau_star = np.zeros(data_dict['rate'].shape[0])
        phi_star = np.zeros(data_dict['rate'].shape[0])
        fn_min = np.inf*np.ones(data_dict['rate'].shape[0])
        
        # # check how many regressors are being used (** consider redoing without this **)
        self.get_regressors() 
        
        # initialize P (list of dicts)
        P = [] #np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
        P_tot = [] #np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList),self.tot_n_regressors))
        obj_tot = np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList)))
        for n in range(data_dict['rate'].shape[0]):
            p, obj = self.fit_reg_linear(n=n, phi=phiList[0])
            P.append(p)
            P_tot_i = []
            for i in range(len(tauList)):
                P_tot_i.append( [None]*len(phiList) )
            P_tot.append(P_tot_i.copy())
        
        # fit model -
        # for each value of tau and phi, check if pInv solution is better than previous
        for i in range(len(tauList)):
            if not np.mod(i,10): print(i, end=' ')
            self.params['tau'] = tauList[i]
            self.get_regressors() 
            
            for j in range(len(phiList)):
                for n in range(data_dict['rate'].shape[0]):
                    p, obj = self.fit_reg_linear(n=n, phi=phiList[j])
                    obj_tot[n,i,j] = obj
                    P_tot[n][i][j] = p
                    if (obj<fn_min[n]):
                        tau_star[n] = tauList[i]
                        phi_star[n] = phiList[j]
                        P[n] = p
                        fn_min[n] = obj

        # collect output ********* redo with dictionary defined at outset to eliminate need for this ---------
        self.model_fit = []
        for n in range(data_dict['rate'].shape[0]):
            d = copy.deepcopy(P[n])
            d['tau'] = tau_star[n]
            d['phi'] = int(phi_star[n])
            d['success'] = True #res['success']
            d['P_tot'] = P_tot[n] #P_tot[n,:,:,:]
            d['obj_tot'] = obj_tot[n] #obj_tot[n,:,:]
            self.model_fit.append(d)


    def fit_and_eval_reg_model_extended(self):
        self.fit_reg_model_extended()
        self.evaluate_reg_model_extended()

