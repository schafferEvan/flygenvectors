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
                    'run_full_sweep':True,
                    'sigLimSec':60,
                    'phaseLimSec': 3,
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
        self.P_tot = []
        self.obj_tot = []



    def fit_reg_linear(self, n, phi, tseries_to_sub=None, D=None, Dinv=None, reg_labels=None):
        '''
        n=cell number
        phi=offset
        tseries_to_sub (optional): pre-subtract a timeseries. Useful for parameter sweeps
        D (optional): provide regressor array. Otherwise defaults to reg array attribute of current object
        Dinv (optional): provide regressor array. Otherwise defaults to inv reg array attribute of current object
        reg_labels (optional): required if providing D, unneeded otherwise. Manually provide fields for dict output
        '''
        dFF_full = self.data_dict['rate'][n,:]
        
        # dFF slides past beh with displacement phi
        L = self.params['L']
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]
        if tseries_to_sub is not None:
            dFF -= tseries_to_sub

        # for a given tau, fit for other parameters is linear, solved by pseudoinverse
        if D is None:
            D = self.regressors_array
        if Dinv is None:
            Dinv = self.regressors_array_inv
        dFF = np.expand_dims(dFF, axis=0)
        # coeffs = np.matmul( np.matmul(dFF,D.T), Dinv )
        coeffs = np.squeeze( (dFF@D.T) @ Dinv )
        dFF_fit = coeffs@D #np.matmul(coeffs,D)
        obj = ((dFF-dFF_fit)**2).sum()

        # split parameters 'coeffs' back into dictionary by labeled regressor
        if reg_labels is None:
            reg_labels = list(self.regressors_dict.keys())
        coeff_dict = {}
        cumulative_tot = 0
        for j in range(len(reg_labels)):
            n_this_reg = self.regressors_dict[ reg_labels[j] ].shape[0]
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
        dFF_full = self.data_dict['rate']
        
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
        hunger /= abs(hunger).max()
        drink = np.squeeze(data_dict['drink'].astype(float))
        drink -= drink.mean()
        drink /= abs(drink).max()

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
            x_c = np.zeros((NT,-2*params['L']+len(data_dict['behavior'])))
            for i in range(NT):
                is_this_trial = np.squeeze(data_dict['trialFlag']==U[i])
                ball[i,is_this_trial] = data_dict['behavior'][is_this_trial]-data_dict['behavior'][is_this_trial].mean()
                x_c_full = np.convolve(kern,ball[i,:],'same')                      # ******* fix 'same' and 'gauss/exp' option *******
                x_c[i,:] = x_c_full[L:-L]-x_c_full[L:-L].mean()
                x_c[i,:] /= abs(x_c[i,:]).max()
        else:
            ball = data_dict['behavior']-data_dict['behavior'].mean()
            x_c_full = np.convolve(kern,ball,'same')                      # ******* fix 'same' and 'gauss/exp' option *******
            x_c = x_c_full[L:-L]-x_c_full[L:-L].mean()
            x_c /= abs(x_c).max()
        
        alpha_regs = np.concatenate( (np.ones((1,len(ts)))/len(ts), ts.T) )
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
        if (self.regressors_array.max()>1) or (self.regressors_array.min()<-1):
            print('WARNING: regressor not normalized')
            # pdb.set_trace()


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




    def fit_reg_model_MLE(self):
        # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
        # this is an extension of ".taulist_shiftlist_gauss", with two additions:
        #       (1) merges gauss/exp kernel methods with option for either
        #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
        
        data_dict = self.data_dict
        sigLimSec = self.params['sigLimSec'] #100
        phaseLimSec = self.params['phaseLimSec'] #20
        sigLim = sigLimSec*data_dict['scanRate']
        self.params['L'] = int(phaseLimSec*data_dict['scanRate'])
        self.params['M'] = round(-sigLim*np.log(0.1)).astype(int)
        self.params['mu'] = .5*self.params['M']/data_dict['scanRate']
        tauList = np.logspace(-1,np.log10(sigLimSec),num=60)
        phiList = np.linspace(-self.params['L'],self.params['L'], num=2*phaseLimSec+1 ).astype(int)
        tau_star = np.zeros(data_dict['rate'].shape[0])
        phi_star = np.zeros(data_dict['rate'].shape[0])
        fn_min = np.inf*np.ones(data_dict['rate'].shape[0])
        
        # fit model -  for each value of tau and phi, check if pInv solution is better than previous
        P = [None]*data_dict['rate'].shape[0] # [] #np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
        for i in range(len(tauList)):
            if not np.mod(i,2): print(i, end=' ')
            self.params['tau'] = tauList[i]
            self.get_regressors() 
            
            for j in range(len(phiList)):
                for n in range(data_dict['rate'].shape[0]):
                    p, obj = self.fit_reg_linear(n=n, phi=phiList[j])
                    if (obj<fn_min[n]):
                        tau_star[n] = tauList[i]
                        phi_star[n] = phiList[j]
                        P[n] = p
                        fn_min[n] = obj
                    
        # collect output ********* redo with dictionary defined at outset to eliminate need for this ---------
        self.model_fit = []
        for n in range(data_dict['rate'].shape[0]):
            # if not self.model_fit:
            d = copy.deepcopy(P[n])
            d['tau'] = tau_star[n]
            d['phi'] = int(phi_star[n])
            d['success'] = True #res['success']
            self.model_fit.append(d)
            # else:
            #     self.model_fit[n]['tau'] = tau_star[n]
            #     self.model_fit[n]['phi'] = int(phi_star[n])
            #     self.model_fit[n]['success'] = True #res['success']





    def fit_reg_model_full_likelihood(self):
        # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
        # this is an extension of ".taulist_shiftlist_gauss", with two additions:
        #       (1) merges gauss/exp kernel methods with option for either
        #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
        
        self.params['phaseLimSec'] = 1
        data_dict = self.data_dict
        sigLimSec = self.params['sigLimSec'] #100
        phaseLimSec = self.params['phaseLimSec'] #20
        sigLim = sigLimSec*data_dict['scanRate']
        self.params['L'] = int(phaseLimSec*data_dict['scanRate'])
        self.params['M'] = round(-sigLim*np.log(0.1)).astype(int)
        self.params['mu'] = .5*self.params['M']/data_dict['scanRate']
        tauList = np.logspace(-1,np.log10(sigLimSec),num=10)
        phiList = np.linspace(-self.params['L'],self.params['L'], num=2*phaseLimSec+1 ).astype(int)
        norm_coeff_list = np.linspace(-1,1,num=3) # ********** change this to 21
        
        tau_star = np.zeros(data_dict['rate'].shape[0])
        phi_star = np.zeros(data_dict['rate'].shape[0])
        fn_min = np.inf*np.ones(data_dict['rate'].shape[0])
        
        # # check how many regressors are being used (** consider redoing without this **)
        self.get_regressors() 
        # initialize P (list of dicts)
        P = [None]*data_dict['rate'].shape[0] # [] #np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
        self.P_tot = {'alpha_01': np.zeros( ( data_dict['rate'].shape[0],len(tauList),len(phiList),
                            len(norm_coeff_list),len(norm_coeff_list),
                            len(norm_coeff_list),len(norm_coeff_list),
                            self.regressors_dict['alpha_01'].shape[0] ) ), 
                'trial': np.zeros( ( data_dict['rate'].shape[0],len(tauList),len(phiList),
                            len(norm_coeff_list),len(norm_coeff_list),
                            len(norm_coeff_list),len(norm_coeff_list),
                            self.regressors_dict['trial'].shape[0] ) )}
        self.obj_tot = np.zeros( ( data_dict['rate'].shape[0],len(tauList),len(phiList),
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
                                    
                    







    def get_dFF_fit(self, n):
        dFF_full = self.data_dict['rate'][n,:]
        phi = self.model_fit[n]['phi']
        L = self.params['L']
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]

        self.params['tau'] = self.model_fit[n]['tau']
        self.get_regressors() 
        coeff_dict = self.coeff_dict_from_keys(idx=n)
        coeff_array = self.dict_to_flat_list(coeff_dict)
        dFF_fit = coeff_array@self.regressors_array
        return dFF_fit, dFF

    def fit_and_eval_reg_model_extended(self):
        self.fit_reg_model_MLE()
        self.evaluate_reg_model_extended()

