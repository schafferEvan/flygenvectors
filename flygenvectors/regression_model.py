import os
import sys
import glob
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import copy
import data as dataUtils
import utils as futils
import plotting
import ssmplotting
import pdb



class reg_obj:
    def __init__(self, activity='dFF', exp_id=None, data_dict={}, fig_dirs={}, split_behav=False, use_beh_labels=None, use_only_valid=False):
        """
        Model:
            'alpha_0': baseline (up to quadratic in time)
            'beta_0':  running (taken from ball motion energy)
            'gamma_0': running state change (smooth derivative of ball motion)
            'delta_0': labels (from arHMM + DGP)
            'tau':     width of kernel (applied to beta and delta)
            'phi':     displacement of kernel (applied to beta and delta)
            'trial_coeffs': # separate baselines for trials (1-N_trials)
        split_behav: treat behavior from separate trials as separate regressors (for run+flail experiments)
        use_only_valid: fit model only using time points where beh label is valid (duration > 1s)
        """
        self.data_dict = copy.deepcopy(data_dict) #protect original dictionary
        self.data_dict_orig = copy.deepcopy(self.data_dict)
        self.data_dict_downsample = copy.deepcopy(self.data_dict)
        self.exp_id = exp_id
        self.fig_dirs = fig_dirs
        self.activity = activity # valid values: {'dFF', 'rate'}, determines what trace is used for fit
        self.kern_type = 'double_exp' # formerly 'gauss'
        self.params = {
            'split_behav':split_behav,
            'use_beh_labels':use_beh_labels,
            'use_only_valid':use_only_valid,
            'sigLimSec':60,
            'M':0,
            'L':0,
            'tau':0.1,
         }
        self.regressors_dict = {
            'alpha_0':np.array([]), 
            'beta_0':np.array([]), 
            'trial':np.array([]), 
            'drink_hunger':np.array([]),
            'beh_lbl':np.array([])
        }
        self.is_downsampled = False
        self.t_exp = []
        self.coeff_label_list = ['alpha_0','beta_0','gamma_0','delta_0','tau','phi','trial_coeffs']
        self.options = {'make_motion_hist':False}
        self.cell_id = 0
        self.model_fit = []
        self.exclude_regressors = None


    def get_model(self, fit_coeffs, just_null_model=False):
        '''use parameters to generate fit''' 
        d = self.coeff_list_to_dict(fit_coeffs)
        if not just_null_model:
            self.get_linear_regressors(d)
        time_regs = self.regressors_dict['alpha_0']
        trial_regressors = self.regressors_dict['trial']
        lin_piece = d['alpha_0']@time_regs + d['trial_coeffs']@trial_regressors
        A = np.array([[1,1],[0,1]])
        if just_null_model:
            dFF_fit =  lin_piece
        elif self.params['split_behav']:
            dFF_fit =  lin_piece + d['beta_0']@self.linear_regressors_dict['beta_0'] + d['gamma_0']@A@self.linear_regressors_dict['gamma_0']
        elif self.params['use_beh_labels']:
            dFF_fit =  lin_piece + d['beta_0']*self.linear_regressors_dict['beta_0'] + d['gamma_0']@A@self.linear_regressors_dict['gamma_0'] + d['delta_0']@self.linear_regressors_dict['delta_0'] 
        else:
            dFF_fit =  lin_piece + d['beta_0']*self.linear_regressors_dict['beta_0'] + d['gamma_0']@A@self.linear_regressors_dict['gamma_0'] 
        return dFF_fit


    def get_linear_regressors(self, d):
        # apply all nonlinearities to self.regressors_dict, so that output regs are linear
        # d is a dictionary of regression coefficients
        kern = self.get_kern(d['phi'], d['tau']) #(1/np.sqrt(tau))*np.exp(-t_exp/tau)
        ball = self.regressors_dict['beta_0']
        
        # ball
        if self.params['split_behav']:
            self.linear_regressors_dict['beta_0'] = np.zeros(ball.shape)
            for i in range(ball.shape[0]):
                self.linear_regressors_dict['beta_0'][i,:] = np.convolve(kern,ball[i,:],'same')
                self.linear_regressors_dict['beta_0'][i,:] -= self.linear_regressors_dict['beta_0'][i,:].mean()
        else:
            self.linear_regressors_dict['beta_0'] = np.convolve(kern,ball,'same')
            self.linear_regressors_dict['beta_0'] -= self.linear_regressors_dict['beta_0'].mean()

        # beh labels
        self.linear_regressors_dict['delta_0'] = np.zeros(self.regressors_dict['delta_0'].shape)
        if self.params['use_beh_labels']:
            for i in range(self.regressors_dict['delta_0'].shape[0]):
                self.linear_regressors_dict['delta_0'][i,:] = np.convolve(kern, self.regressors_dict['delta_0'][i,:], 'same')
                if self.linear_regressors_dict['delta_0'][i,:].mean() != 0:
                    self.linear_regressors_dict['delta_0'][i,:] -= self.linear_regressors_dict['delta_0'][i,:].mean()
    

    def get_kern(self, phi, tau):
        '''make kernel''' 
        t_exp = self.t_exp
        kern = np.zeros(len(t_exp))
        p = t_exp>=phi
        n = t_exp<phi
        kern[p] = np.exp(-(t_exp[p]-phi)/tau)
        kern[n] = np.exp((t_exp[n]-phi)/tau)
        return kern


    def get_objective_fn(self, fit_coeffs):
        '''evaluate fit
        use_only_valid: if True, computes objective only from times with valid behav label, defined in preprocessing as state with duration > 1s
        ''' 
        dFF = self.data_dict[self.activity][self.cell_id,:]
        dFF_fit = self.get_model(fit_coeffs)
        #obj = ((dFF[len(t_exp)-1:]-dFF_fit)**2).sum()
        if self.params['use_only_valid']:
            v = self.data_dict['state_is_valid']
            obj = ((dFF[v]-dFF_fit[v])**2).sum()
        else:
            obj = ((dFF-dFF_fit)**2).sum()
        return obj


    def get_model_mle(self, downsample=True, shifted=None, initial_conds=None, bounds=None):
        """
        downsample: flag creates downsampled dict, or points to it if one is already made.
        shifted: if not None, uses circshifted dict instead of original
        time_cropping: if not None, crops in time (only current self.data_dict, which will be overwritten by downsample)
        """
        self.downsample_in_time()
        # if time_cropping is not None:
        #     self.data_dict['dFF'] = self.data_dict['dFF'][:,time_cropping[0]:time_cropping[1]]
        #     self.data_dict['time'] = self.data_dict['time'][time_cropping[0]:time_cropping[1]]
        #     self.data_dict['trialFlag'] = self.data_dict['trialFlag'][time_cropping[0]:time_cropping[1]]
        #     self.data_dict['behavior'] = self.data_dict['behavior'][time_cropping[0]:time_cropping[1]]
        # **********
        # add call to method here that overwrites data_dict with only some timesteps.
        # NEED call downsample() beforehand to reset the dict.
        # **********
        if self.params['split_behav'] and self.params['use_beh_labels']:
            print('ERROR: SPLIT BEHAV with BEHAV LABELS NOT IMPLEMENTED')
            return

        if initial_conds is None:
            initial_conds = self.get_default_inits()
        self.get_regressors(shifted=shifted)
        if bounds is None:
            bounds = self.get_default_bounds()
        if self.exclude_regressors is not None:
            bounds = self.exclude_regressors_by_bounds(bounds)
        for i in range(self.n_trials-1):
            initial_conds.append(0)    # trial coeffs
            bounds.append([None,None]) # trial coeffs   
        N = self.data_dict[self.activity].shape[0]
        model_fit = [None]*N
        for n in range(N):
            if not np.mod(n,100): print(str(int(100*n/N))+'%', end=' ')
            # self.cell_id = n
            # res = minimize(self.get_objective_fn, initial_conds, method='SLSQP', bounds=bounds)
            # model_fit[n] = self.coeff_list_to_dict(res['x'])
            model_fit[n] = self.get_one_cell_mle(cell_id=n, initial_conds=initial_conds, bounds=bounds, shifted=shifted)
        return model_fit


    def get_one_cell_mle(self, cell_id=None, initial_conds=None, bounds=None, shifted=None):
        self.cell_id = cell_id
        if initial_conds is None:
            initial_conds = self.get_default_inits()
        self.get_regressors(shifted=shifted)
        if bounds is None:
            bounds = self.get_default_bounds()
        if self.exclude_regressors is not None:
            bounds = self.exclude_regressors_by_bounds(bounds)
        for i in range(self.n_trials-1):
            initial_conds.append(0)    # trial coeffs
            bounds.append([None,None]) # trial coeffs   
            
        res = minimize(self.get_objective_fn, initial_conds, method='SLSQP', bounds=bounds)
        cell_fit = self.coeff_list_to_dict(res['x'])
        return cell_fit
        

    def dict_to_flat_list(self, coeff_dict):
        # regenerate coeff dict from list (inverse of below).
        reg_labels = list(coeff_dict.keys())
        coeff_list_nested = []
        for j in range(len(reg_labels)):
            coeff_list_nested.append( coeff_dict[ reg_labels[j] ] )
        coeff_list = [val for sublist in coeff_list_nested for val in sublist]
        return coeff_list


    def coeff_list_to_dict(self, coeff_list):
        # regenerate coeff list from dict (inverse of above)
        coeff_dict = {}
        cumulative_tot = 0
        for idx in enumerate(self.coeff_label_list):
            s = self.get_coeff_shape(idx[1])
            coeff_dict[idx[1]] = coeff_list[ cumulative_tot:(cumulative_tot+s) ]
            cumulative_tot += s
        return coeff_dict


    def get_coeff_shape(self, label):
        if label=='alpha_0':
            s = self.regressors_dict['alpha_0'].shape[0]
        elif label=='beta_0':
            if self.params['split_behav']:
                s = self.regressors_dict['beta_0'].shape[0]
            else:
                s = 1
        elif label=='gamma_0':
            if 'gamma_0' in self.regressors_dict:
                if self.regressors_dict['gamma_0'] is not None:
                    s = self.regressors_dict['gamma_0'].shape[0]
                else:
                    s = 0
            else:
                s = 0
        elif label=='delta_0':
            # if self.params['use_beh_labels']:
            s = self.regressors_dict['delta_0'].shape[0]
            # else:
            #     s = 0
        elif (label=='tau') or (label=='phi'):
            s = 1
        elif label=='trial_coeffs':
            s = self.regressors_dict['trial'].shape[0]
        return s


    def get_regressors(self, just_null_model=False, shifted=None):
        '''build matrix of all regressors.  Stores output as a dictionary and as a concatenated array.
        Also pre-computes inverse of this array.
        params = {'split_behav':True,
                    'M':M,
                    'L':L,
                    'tau':tau,
                    'mu':mu
                 }
        note: 'mu' corrects for offset from convolution. unrelated to phi and not to be changed.
        phi_input: bypasses default sweep through all phi. useful for eval step, ignore for other functions
        amplify_baseline: scale alpha and trial regs to spare them from elastic net penalty
        just_null_model: returns just alpha and trial regs''' 
        self.refresh_params()
        data_dict = self.data_dict # evaluate on orig, even if fit was done on downsampled data
        params = self.params
        if just_null_model: params['split_behav'] = False
        if just_null_model: params['use_beh_labels'] = False
        # L = params['L']
        ts_full = data_dict['time']-data_dict['time'].mean() #data_dict['time'][0]
        ts = ts_full-ts_full.mean()
        ts /= abs(ts).max()
        alpha_regs = np.concatenate( (np.ones((1,len(ts)))/len(ts), ts.T, ts.T**2) )

        # t_exp = np.linspace(1,params['M'],params['M'])/data_dict['scanRate']
        self.t_exp = np.linspace(-params['M']/2, params['M']/2, params['M'])/data_dict['scanRate']

        # define trial flag regressors
        NT = self.n_trials
        # print(str(NT)+' trials')
        trial_regressors = np.zeros((NT-1,len(data_dict['trialFlag'])))
        # pdb.set_trace()
        for i in range(NT-1):
            is_this_trial = data_dict['trialFlag']==self.U[i]
            trial_regressors[i,:] = 1.0*np.squeeze(is_this_trial) #[L:-L]
            trial_regressors[i,:] -= trial_regressors[i,:].mean()
            trial_regressors[i,:] /= abs(trial_regressors[i,:]).max()

        # motion energy from ball
        if params['split_behav']:
            if shifted is None:
                ball = np.zeros((NT,len(data_dict['behavior'])))
                for i in range(NT):
                    is_this_trial = np.squeeze(data_dict['trialFlag']==self.U[i])
                    ball[i,is_this_trial] = data_dict['behavior'][is_this_trial]-data_dict['behavior'][is_this_trial].mean()
            else:
                ball = np.zeros((NT,len(data_dict['behavior'])))
                for i in range(NT):
                    is_this_trial = np.squeeze(data_dict['trialFlag']==self.U[i])
                    ball[i,is_this_trial] = data_dict['circshift_behav'][shifted][is_this_trial]-data_dict['circshift_behav'][shifted][is_this_trial].mean()
        # elif just_null_model:
        #     ball = []
        else:
            if shifted is None:
                ball = data_dict['behavior']-data_dict['behavior'].mean()
            else:
                ball = data_dict['circshift_behav'][shifted] - data_dict['circshift_behav'][shifted].mean()

        # running state transitions
        # start with binary timeseries of behavioral state starts & stops (ignoring brief states)
        if (self.exp_id=='2018_08_24_fly3_run1') or (self.exp_id=='2018_08_24_fly2_run2'):
            if shifted is None:
                is_running = 1*(data_dict['behavior']>.0005)
            else:
                is_running = 1*(data_dict['circshift_behav'][shifted]>.0005)
        else:
            if shifted is None:
                is_running = 1*(data_dict['beh_labels'][:,2]>0.5)
            else:
                is_running = 1*(data_dict['circshift_beh_labels'][shifted][:,2]>0.5)

        run_diff_smooth = self.get_beh_diff(is_running)

        # grooming (beh labels)
        grooming = np.zeros((2,len(data_dict['behavior'])))
        if self.params['use_beh_labels']:
            if shifted is None:
                grooming[0,:] = data_dict['beh_labels'][:,3]
                grooming[1,:] = data_dict['beh_labels'][:,4]
            else:
                grooming[0,:] = data_dict['circshift_beh_labels'][shifted][:,3]
                grooming[1,:] = data_dict['circshift_beh_labels'][shifted][:,4]

        # if just_null_model:
        #     self.regressors_dict = {
        #         'alpha_0':alpha_regs, 
        #         'trial':trial_regressors,
        #         'beta_0':None,
        #         'gamma_0':None,
        #         'delta_0':None 
        #     }
        # else:
        self.regressors_dict = {
            'alpha_0':alpha_regs, 
            'trial':trial_regressors, 
            'beta_0':ball,
            'gamma_0':run_diff_smooth,
            'delta_0':grooming 
        }
        self.linear_regressors_dict = copy.deepcopy(self.regressors_dict)


    def get_beh_diff(self, is_running, sec_th=10, sig_raw=8):
        frame_th = np.round(sec_th*self.data_dict['scanRate']).astype(int)
        
        # start with binary timeseries of behavioral state starts & stops (ignoring brief states)
        rstates = {'states':is_running}
        states_out = self.remove_transient_behaviors(rstates, frame_th=frame_th)

        # timeseries of run starts
        run_diff_pos = np.zeros(len(is_running))
        run_diff_pos[:-1] = np.diff(states_out['states'],axis=0)
        run_diff_neg = run_diff_pos.copy()
        run_diff_pos[run_diff_pos<0] = 0
        run_diff_neg[run_diff_neg>0] = 0
        
        # smoothing
        sig = sig_raw*self.data_dict['scanRate']
        b_lim = 40*self.data_dict['scanRate']
        t_exp = np.linspace(-b_lim, b_lim, 2*int(b_lim)+1)
        kern = ((t_exp[1]-t_exp[0])/(sig*np.sqrt(2*np.pi)))*np.exp(-0.5*(t_exp/sig)**2)
        # run_diff_smooth = np.convolve(kern,run_diff,'same')        
        run_diff_pos_smooth = np.convolve(kern,run_diff_pos,'same')
        run_diff_neg_smooth = np.convolve(kern,run_diff_neg,'same')
        run_diff_pos_smooth = np.expand_dims(run_diff_pos_smooth-run_diff_pos_smooth.mean(), axis=0)
        run_diff_neg_smooth = np.expand_dims(run_diff_neg_smooth-run_diff_neg_smooth.mean(), axis=0)
        run_diff_smooth = np.concatenate( (run_diff_pos_smooth, run_diff_neg_smooth), axis=0 )
        return run_diff_smooth


    def downsample_in_time(self, downsample_factor=2):
        """ downsample data_dict in time for more efficient model fitting
        effective_stepsize: new volumetric rate in Hz
        """
        if self.is_downsampled:
            # if downsampling already happened, just point to this version of the data
            self.data_dict = copy.deepcopy(self.data_dict_downsample)
        else:
            # make downsampled dataset
            self.is_downsampled = True
            self.data_dict['scanRate'] = self.data_dict_orig['scanRate']/downsample_factor #1/effective_stepsize
            sub_fac = downsample_factor #effective_stepsize*self.data_dict_orig['scanRate'] #np.round(effective_stepsize*self.data_dict_orig['scanRate']).astype(int)
            L = np.round(self.data_dict_orig[self.activity].shape[1]/sub_fac).astype(int)
            # self.data_dict[self.activity] = np.zeros((self.data_dict_orig[self.activity].shape[0], L))
            self.data_dict['dFF'] = np.zeros((self.data_dict_orig['dFF'].shape[0], L))
            self.data_dict['dYY'] = np.zeros((self.data_dict_orig['dYY'].shape[0], L))
            self.data_dict['dRR'] = np.zeros((self.data_dict_orig['dRR'].shape[0], L))
            self.data_dict['time'] = np.zeros((L,1))
            self.data_dict['tPl'] = np.zeros((L,1))
            self.data_dict['behavior'] = np.zeros(L)
            self.data_dict['trialFlag'] = np.zeros(L)
            self.data_dict['beh_labels'] = np.zeros( (L, self.data_dict_orig['beh_labels'].shape[1]) )
            self.data_dict['state_is_valid'] = np.zeros(L, dtype=bool)
            for j in range(L):
                sl = slice(np.round(sub_fac*j).astype(int), np.round(sub_fac*(j+1)).astype(int))
                # self.data_dict[self.activity][:, j] = self.data_dict_orig[self.activity][:,sl].mean(axis=1)
                self.data_dict['dFF'][:, j] = self.data_dict_orig['dFF'][:,sl].mean(axis=1)
                self.data_dict['dYY'][:, j] = self.data_dict_orig['dYY'][:,sl].mean(axis=1)
                self.data_dict['dRR'][:, j] = self.data_dict_orig['dRR'][:,sl].mean(axis=1)
                self.data_dict['time'][j,0] = self.data_dict_orig['time'][sl].mean() #sub_fac*j
                self.data_dict['tPl'][j,0] = self.data_dict_orig['tPl'][sl].mean() #sub_fac*j
                self.data_dict['behavior'][j] = self.data_dict_orig['behavior'][sl].mean()
                if self.data_dict_orig['beh_labels'].shape[1]==1:
                    self.data_dict['beh_labels'][j,:] = stats.mode(self.data_dict_orig['beh_labels'][sl, :]).mode
                else:
                    self.data_dict['beh_labels'][j,:] = self.data_dict_orig['beh_labels'][sl, :].mean(axis=0)
                self.data_dict['trialFlag'][j] = stats.mode(self.data_dict_orig['trialFlag'][sl]).mode
                if 'state_is_valid' in self.data_dict_orig:
                    self.data_dict['state_is_valid'][j] = np.max( self.data_dict_orig['state_is_valid'][sl] ) # True if any elem is True
            self.data_dict_downsample = copy.deepcopy(self.data_dict)

        

    def evaluate_model(self, model_fit, reg_labels=None, shifted=None):
        # regenerate fit from best parameters and evaluate model
        # self.data_dict = self.data_dict_orig # don't do this
        self.refresh_params()
        self.get_regressors(shifted=shifted)
        if reg_labels is None:
            if self.params['use_beh_labels']:
                reg_labels=['beta_0','gamma_0','delta_0']
            else:
                reg_labels=['beta_0','gamma_0']
        
        print('evaluating ', end='')
        for n in range(self.data_dict[self.activity].shape[0]):
            if not np.mod(n,round(self.data_dict[self.activity].shape[0]/20)): print('.', end='')
            sys.stdout.flush()
            r_sq, stat, cc, _ = self.evaluate_model_for_one_cell(n, 
                model_fit=model_fit, reg_labels=reg_labels, shifted=shifted, regs_prepped=True)
            model_fit[n]['r_sq'] = r_sq #1-SS_res.sum()/SS_tot.sum()
            model_fit[n]['stat'] = stat
            model_fit[n]['cc'] = cc
        print(' Complete')
        sys.stdout.flush()

    
    
    def evaluate_model_for_one_cell(self, n, model_fit, reg_labels=None, shifted=None, regs_prepped=False):
        # regenerate fit from best parameters and evaluate model
        # self.data_dict = self.data_dict_orig # don't do this
        if not regs_prepped:
            self.refresh_params()
            self.get_regressors(shifted=shifted)
        if reg_labels is None:
            if self.params['use_beh_labels']:
                reg_labels=['beta_0','gamma_0','delta_0']
            else:
                reg_labels=['beta_0','gamma_0']
        
        dFF = self.data_dict[self.activity][n,:].copy()
        coeff_list = self.dict_to_flat_list(model_fit[n])
        dFF_fit = self.get_model(coeff_list)
        coeff_dict = model_fit[n]
        self.get_linear_regressors(coeff_dict)

        # get linpart to subtract from everything
        coeffs_null = copy.deepcopy(model_fit[n])
        for label in reg_labels:
            for j in range(coeffs_null[label].shape[0]):
                coeffs_null[label][j] = 0
        coeff_list = self.dict_to_flat_list(coeffs_null)
        dFF_fit_linpart = self.get_model(coeff_list) 
        dFF -= dFF_fit_linpart
        dFF_fit -= dFF_fit_linpart
        if self.params['use_only_valid']:
            dFF = dFF[self.data_dict['state_is_valid']]
            dFF_fit = dFF_fit[self.data_dict['state_is_valid']]

        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        r_sq_tot = 1-SS_res.sum()/SS_tot.sum()
        
        # for all parameter categories, for all parameters in this category, find fit without this parameter to compute p_val
        stat = {}  
        for label in reg_labels: stat[label] = None
        r_sq = copy.deepcopy(stat)
        cc = copy.deepcopy(stat)
        null_fits = copy.deepcopy(stat)
        r_sq['tot'] = r_sq_tot
        exclude_regressors_backup = self.exclude_regressors.copy()
        for label in reg_labels:
            stat_list = []
            r_sq_list = []
            cc_list = []
            null_fit_list = [None]*coeff_dict[label].shape[0]
            for j in range(coeff_dict[label].shape[0]):
                # j_inc = [x for x in range(coeff_dict[label].shape[0]) if x != j]
                # coeffs_null = copy.deepcopy(model_fit[n])
                # coeffs_null[label][j] = 0
                self.exclude_regressors = exclude_regressors_backup.copy()
                if coeff_dict[label].shape[0]>1:
                    self.exclude_regressors.append( [label,j] )
                else:
                    self.exclude_regressors.append( [label] )
                ics = self.dict_to_flat_list(model_fit[n])
                coeffs_null = get_one_cell_mle(cell_id=n, initial_conds=ics, shifted=shifted)
                coeff_list = self.dict_to_flat_list(coeffs_null)
                dFF_fit_null = self.get_model(coeff_list) 

                dFF_fit_null -= dFF_fit_linpart
                if self.params['use_only_valid']:
                    dFF_fit_null = dFF_fit_null[self.data_dict['state_is_valid']]

                null_fit_list[j] = dFF_fit_null
                SS_res_0 = ( (dFF-dFF_fit_null)**2 )
                # res_var_0 = (dFF-dFF_fit_null).var()
                stat_list.append( stats.wilcoxon(np.squeeze(SS_res_0),np.squeeze(SS_res)) )
                r_sq_list.append( (SS_res_0.sum()-SS_res.sum())/SS_tot.sum() )
                
                # cc between a regressor and fit leaving out regressor
                norm_resid = dFF-dFF_fit_null - (dFF-dFF_fit_null).mean()
                if len( self.linear_regressors_dict[label].shape )==1:
                    norm_reg = self.linear_regressors_dict[label].copy()
                else:
                    norm_reg = self.linear_regressors_dict[label][j,:].copy()
                norm_reg -= norm_reg.mean()
                if self.params['use_only_valid']:
                    norm_reg = norm_reg[self.data_dict['state_is_valid']]
                cc_list.append( (norm_resid*norm_reg).mean()/(norm_resid.std()*norm_reg.std()) )
                # if r_sq_list[-1]<0:
                #     print('shit', end=' ')
            stat[label] = stat_list
            r_sq[label] = r_sq_list
            cc[label] = cc_list
            null_fits[label] = null_fit_list
         
        # if self.params['use_beh_labels'] is not None:
        #     stat['tau_beh_lab'] = stat['beh_lbl']
        #     stat['phi_beh_lab'] = stat['beh_lbl']
        if 'drink_hunger' in list(stat.keys()): stat['tau_feed'] = stat['drink_hunger']    
        self.exclude_regressors = exclude_regressors_backup
        r_sq['tau'] = r_sq['beta_0']
        r_sq['phi'] = r_sq['beta_0']
        stat['tau'] = stat['beta_0']
        stat['phi'] = stat['beta_0']
        cc['tau'] = cc['beta_0']
        cc['phi'] = cc['beta_0']
        return r_sq, stat, cc, null_fits



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


    def normalize_rows_mag(self,input_mat,quantiles=(.01,.99),fix_nans=True): 
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


    def unnormalize_rows_mag(self,input_mat,q0,q1): 
        # inverts "normalize_rows", except for cropped extrema
        output_mat = np.zeros(input_mat.shape)
        for i in range(input_mat.shape[0]):
            output_mat[i,:] = input_mat[i,:]*(q1[i]-q0[i]) + q0[i]
        return output_mat


    def convert_dFF_to_ROC(self, input_mat, classifier_tseries=None):
        """
        convert activity to ROC using a binary classifier to define null.
        Default classifier uses stationary state vs anything nonstationary
        """
        if classifier_tseries is None:
            classifier_tseries = self.data_dict['beh_labels']!=0
        output_mat = np.zeros(input_mat.shape)
        l = np.logical_not(classifier_tseries).sum()
        for i in range(input_mat.shape[0]):
            null_dist = input_mat[i, np.logical_not(classifier_tseries)]
            for t in range(input_mat.shape[1]):
                output_mat[i,t] = (input_mat[i,t]>null_dist).sum()/l
        return output_mat
    

    def refresh_params(self):
        data_dict = self.data_dict
        sigLimSec = self.params['sigLimSec'] #100
        sigLim = sigLimSec*data_dict['scanRate']
        self.params['M'] = np.round(-sigLim*np.log(0.1)).astype(int)      
        self.U = np.unique(data_dict['trialFlag'])
        self.n_trials = len(self.U)


    def get_model_sig_from_shifted_fit(self, param, param_idx=0, sig_th=0.05, update_model_fit=True):
        # uses model_fit and model_fit_shifted to assess significance of fits
        # output is number of standard deviations real fit is away from shifted fit
        f = plotting.get_model_fit_as_dict(self.model_fit)
        rsq = plotting.get_model_fit_as_dict(f['r_sq'])
        # fs_ = []
        # for i in range(len(self.model_fit_shifted)):
        #     fs_.extend( plotting.get_model_fit_as_dict(self.model_fit_shifted[i]) )
        # fs = plotting.get_model_fit_as_dict(fs_)
        # rsq_shift = plotting.array_of_dicts_to_dict_of_arrays(fs['r_sq'])
        tot_shift = plotting.array_of_dicts_to_dict_of_arrays(self.model_fit_shifted)
        rsq_shift = plotting.array_of_dicts_to_dict_of_arrays(tot_shift['r_sq'])

        p_vals = np.zeros(len(rsq[param]))
        for n in range(len(rsq[param])):
            if isinstance(rsq[param][n],np.float64):
                p_vals[n] = 1 - (rsq[param][n]>rsq_shift[param][:,n,param_idx]).sum()/rsq_shift[param].shape[0]
            else:
                p_vals[n] = 1 - (rsq[param][n,param_idx]>rsq_shift[param][:,n,param_idx]).sum()/rsq_shift[param].shape[0]
            # m=rsq_shift[param][:,n,param_idx].mean()
            # s=rsq_shift[param][:,n,param_idx].std()
            # std_devs[n] = (rsq[param][n]-m)/s
        # is_sig = std_devs>sig_th
        if update_model_fit:
            for n in range(len(self.model_fit)):
                self.model_fit[n]['stat'][param][param_idx] = [None, p_vals[n]]
        return p_vals


    def remove_transient_behaviors(self, states, frame_th=35):
        state_copy = copy.deepcopy(states)
        indexing_list = ssmplotting._get_state_runs(states=[states['states']])
        b=0
        for k in range(indexing_list[b].shape[0]):
            if (indexing_list[b][k][2] - indexing_list[b][k][1]) < frame_th:
                state_copy['states'][indexing_list[b][k][1]:indexing_list[b][k][2]] = 1
        return state_copy


    def get_null_subtracted_raster(self, extra_regs_to_use=None, just_null_model=False):
        """ default is to return fit with just alpha and trial regs. 
        Using extra_regs, option to return fit with any additional regs: [ ['partial_reg', idx], ['full_reg'] ] 
        extra_regs are the regressors TO SUBTRACT """
        self.get_regressors(just_null_model=just_null_model)
        dFF = copy.deepcopy(self.data_dict[self.activity])
        dFF_fit = np.zeros(dFF.shape)
        for n in range(dFF.shape[0]):
            # if just_null_model:
            #     # get linpart to subtract from everything
            #     coeffs_null = copy.deepcopy(self.model_fit[n])
            #     for label in ['beta_0', 'gamma_0', 'delta_0']:
            #         for j in range(coeffs_null[label].shape[0]):
            #             coeffs_null[label][j] = 0
            #     coeff_list = self.dict_to_flat_list(coeffs_null)
            # else:
            coeff_list = self.dict_to_flat_list(self.model_fit[n])
            
            dFF_fit[n,:] = self.get_model(coeff_list, just_null_model=just_null_model)

        dFF -= dFF_fit
        return dFF_fit, dFF


    def get_smooth_behavior(self):
        from skimage.restoration import denoise_tv_chambolle

        split_behav=self.params['split_behav']
        self.data_dict['behavior'] = self.data_dict['ball'].flatten().copy()

        # filter behavior with TV denoising. requires customized settings for seome datasets
        # pdb.set_trace()
        if not self.exp_id: 
            print('***** warning: Missing exp_id *******')

        tv_params = [.01,.9,.05] #np.array([.01,.9,.05])
        if self.exp_id=='2018_08_24_fly3_run1':
            tv_params[0] = 0.2 #0.25
            tv_params[2] = 0.0001 #0.002
        elif self.exp_id=='2018_08_24_fly2_run2':
            tv_params[0] = 0.025
            tv_params[2]=0.002
        elif self.exp_id=='2019_07_01_fly2':
            [] #ok
        elif self.exp_id=='2019_10_14_fly3':
            tv_params[0]=0.05
            tv_params[2]=0.01
        elif self.exp_id=='2019_06_28_fly2':
            tv_params[0]=0.1
        elif self.exp_id=='2019_06_30_fly1':
            [] #ok
        elif self.exp_id=='2019_10_14_fly2':
            tv_params[0]=0.1
        elif self.exp_id=='2019_10_14_fly4':
            tv_params[0]=0.05
        elif self.exp_id=='2019_10_18_fly3':
            tv_params[0]=0.05
        elif self.exp_id=='2019_10_21_fly1':
            [] #ok
        elif self.exp_id=='2019_10_10_fly3':
            tv_params[0]=0.05
        elif self.exp_id=='2019_10_02_fly2':
            tv_params[0]=0.1
        elif self.exp_id=='2019_08_14_fly1':
            [] #ok
        elif self.exp_id=='2019_04_18_fly2':
            tv_params[0]=[.01, 0.12, 0.01]
        elif self.exp_id=='2019_04_22_fly1':
            tv_params[0]=[.01, 0.2, 0.01, 0.01]
        elif self.exp_id=='2019_04_22_fly3':
            [] #ok
        elif self.exp_id=='2019_04_24_fly3':
            [] #ok
        elif self.exp_id=='2019_04_24_fly1':
            [] #ok
        elif self.exp_id=='2019_04_25_fly3':
            tv_params[2]=[.05, .01, .01]
        elif self.exp_id=='2019_05_07_fly1':
            tv_params[2]=[.05, .01, .01]
        elif self.exp_id=='2019_03_12_fly4':
            tv_params[0]=0.15
            tv_params[2]=0.01
        elif self.exp_id=='2019_02_19_fly1':
            tv_params[0]=0.2
            tv_params[2]=0.01
        elif self.exp_id=='2019_02_26_fly1_2':
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
        # manual cleanup (compressing scale)
        if self.exp_id == '2018_08_24_fly3_run1':
            # self.data_dict['behavior'][self.data_dict['behavior']>.0032] = .0032
            self.data_dict['behavior'][self.data_dict['behavior']>.005] = .005
        elif self.exp_id == '2018_08_24_fly2_run2':
            self.data_dict['behavior'][self.data_dict['behavior']>.006] = .006
        elif self.exp_id == '2019_07_01_fly2':
            self.data_dict['behavior'][self.data_dict['behavior']>.022] = .022


    def estimate_motion_artifacts(self, inv_cv_thresh=1.0, max_dRR_thresh=0.75, make_hist=False):
        """
        original value of max_dRR_thresh=0.3
        """
        self.motion = {'inv_cv_thresh':inv_cv_thresh, 'max_dRR_thresh':max_dRR_thresh}
        self.get_regressors(just_null_model=True)
        
        # fit null model to dRR to get dRR0
        self.activity = 'dRR'
        self.model_fit = self.get_model_mle()
    
        R0, dRR0 = self.get_null_subtracted_raster(just_null_model=True)
        v=dRR0.var(axis=1)
        m=R0.mean(axis=1)
        self.motion['mag'] = dRR0.max(axis=1)-dRR0.min(axis=1)
        self.motion['motion_cv_inv'] = v/m**2
        self.motion['motion_cv_inv'][np.isnan(self.motion['motion_cv_inv'])] = np.inf
        self.motion['cvisgood'] = self.motion['motion_cv_inv']<self.motion['inv_cv_thresh']
        self.motion['rmagisgood'] = self.motion['mag']<self.motion['max_dRR_thresh']
        self.motion['isgood'] = np.logical_and(self.motion['cvisgood'], self.motion['rmagisgood'])

        if make_hist:
            cv = self.motion['motion_cv_inv'].copy()
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


    def get_train_test_data(self, trial_len=100, rng_seed=0, trials_tr=10, trials_val=2, trials_test=0, trials_gap=0, activity='dFF'):
        """
        split into train/test trials
        trial_len: length of pseudo-trials
        """
        data_neural = self.data_dict[activity].T        
        n_trials = np.floor(data_neural.shape[0] / trial_len)
        indxs = dataUtils.split_trials(
            n_trials, rng_seed=rng_seed, trials_tr=trials_tr, trials_val=trials_val, trials_test=trials_test, trials_gap=trials_gap)
        data = {}
        for dtype in ['train', 'test', 'val']:
            data_segs = []
            for indx in indxs[dtype]:
                data_segs.append(data_neural[(indx*trial_len):(indx*trial_len + trial_len)])
            data[dtype] = data_segs
        data['train_all'] = np.concatenate(data['train'], axis=0)
        data['val_all'] = np.concatenate(data['val'], axis=0)
        self.data_dict['train_all'] = data['train_all']
        self.data_dict['val_all'] = data['val_all']


    # def get_train_test_indices(self, trial_len=100, rng_seeds=[0,100,1000], trials_tr=10, trials_val=2, trials_test=0, trials_gap=0):
    #     """
    #     split into train/test trials
    #     trial_len: length of pseudo-trials
    #     """
    #     n_trials = np.floor( self.data_dict['time'] / trial_len)
    #     self.data_dict['train_test_indices'] = []
    #     for n in range(len(rng_seeds)):
    #         indxs = dataUtils.split_trials(
    #             n_trials, rng_seed=rng_seeds[n], trials_tr=trials_tr, trials_val=trials_val, trials_test=trials_test, trials_gap=trials_gap)
    #         block_indices = {}
    #         for dtype in ['train', 'test', 'val']:
    #             data_segs = []
    #             for indx in indxs[dtype]:
    #                 data_segs.append( [(indx*trial_len):(indx*trial_len + trial_len)] )
    #             block_indices[dtype] = data_segs
    #         self.data_dict['train_test_indices'].append( block_indices )


    def get_circshift_behav_data(self, abs_min_shift=0.33, rng_seed=0, n_perms=5):
        np.random.seed(rng_seed)
        self.data_dict['circshift_behav'] = [None]*n_perms
        self.data_dict['circshift_beh_labels'] = [None]*n_perms
        low_idx = int( abs_min_shift * len(self.data_dict['behavior']) )
        high_idx = int( (1-abs_min_shift) * len(self.data_dict['behavior']) )
        perms = np.random.randint(low=low_idx, high=high_idx, size=n_perms)
        for n in range(n_perms):
            self.data_dict['circshift_behav'][n] = np.roll(self.data_dict['behavior'], perms[n])
            self.data_dict['circshift_beh_labels'][n] = np.roll(self.data_dict['beh_labels'], perms[n])
        self.data_dict_orig['circshift_behav'] = copy.deepcopy( self.data_dict['circshift_behav'] )
        self.data_dict_downsample['circshift_behav'] = copy.deepcopy( self.data_dict['circshift_behav'] )
        self.data_dict_orig['circshift_beh_labels'] = copy.deepcopy( self.data_dict['circshift_beh_labels'] )
        self.data_dict_downsample['circshift_beh_labels'] = copy.deepcopy( self.data_dict['circshift_beh_labels'] )


    def preprocess(self, do_ICA=False, get_behav_from_ball=False):
        if get_behav_from_ball:
            self.get_smooth_behavior()
        self.data_dict_orig['behavior'] = copy.deepcopy(self.data_dict['behavior'])
        self.data_dict_downsample['behavior'] = copy.deepcopy(self.data_dict['behavior'])
        motion_obj = copy.deepcopy(self) # i'm not proud of this
        motion_obj.estimate_motion_artifacts(make_hist=self.options['make_motion_hist'])
        self.motion = motion_obj.motion
        if self.options['make_motion_hist']:
            plt.savefig(self.fig_dirs['fig_folder'] + self.exp_id +'_motion_artifacts.pdf',transparent=False, bbox_inches='tight')
        isgood = self.motion['isgood']
        self.data_dict['dFF'] = self.data_dict['dFF'][isgood,:]
        self.data_dict['dYY'] = self.data_dict['dYY'][isgood,:]
        self.data_dict['dRR'] = self.data_dict['dRR'][isgood,:]
        self.data_dict['aligned_centroids'] = self.data_dict['aligned_centroids'][isgood,:]
        self.data_dict['A'] = self.data_dict['A'][:,isgood]
        if do_ICA: 
            self.data_dict['dFF'] = dataUtils.get_dFF_ica(self.data_dict)
        # self.data_dict['behavior'] = self.data_dict['behavior'].copy()
        self.data_dict['dt'] = self.data_dict['time'][1]-self.data_dict['time'][0]
        self.data_dict['tPl'] = self.data_dict['time'][0]+np.linspace(0,self.data_dict['dt']*len(self.data_dict['time']),len(self.data_dict['time']))
        self.data_dict['dFF_unnormalized'] = self.data_dict['dFF'].copy()
        self.data_dict['dFF_mag_norm'], self.data_dict['q0'], self.data_dict['q1'] = self.normalize_rows_mag(self.data_dict['dFF_unnormalized'])
        self.data_dict['dFF'], self.data_dict['mu'], self.data_dict['sig'] = self.normalize_rows_euc(self.data_dict['dFF_unnormalized'])
        # self.data_dict['dFF_unnormalized'] = self.unnormalize_rows(self.data_dict['dFF'],self.data_dict['q0'],self.data_dict['q1'])
        self.get_train_test_data()
        return self.data_dict


    def get_default_inits(self):
        if self.params['split_behav']:
            initial_conds=[0,.0001,.0001] # alpha
            U = np.unique(self.data_dict['trialFlag']) # beta
            for i in range(len(U)):
                initial_conds.append(.0001)
            initial_conds.extend([0,0.5,5,0]) # gamma, tau, phi
        else:
            # elif self.params['use_beh_labels']:
            initial_conds=[0,.0001,.0001]        # alpha
            initial_conds.append(.0001)          # beta
            initial_conds.extend([0,0.5])        # gamma
            initial_conds.extend([.0001,.0001])  # delta
            initial_conds.extend([5,0])          # tau, phi
        # else:
        #     initial_conds=[0,.0001,.0001,.0001,0,0.5,5,0]
        return initial_conds


    def get_default_bounds(self):
        if self.params['split_behav']:
            bounds=[[None,None],[None,None],[None,None]] # alpha
            U = np.unique(self.data_dict['trialFlag'])   # beta
            for i in range(len(U)):
                bounds.append([None,None])
            bounds.extend([[None,None],[-.05,.05],[1,59],[-59,59]]) # gamma, tau, phi
        else:
            # elif self.params['use_beh_labels']:
            bounds=[[None,None],[None,None],[None,None]]    # alpha
            bounds.append([None,None])                      # beta
            bounds.extend([[None,None],[-.05,.05]])         # gamma
            bounds.extend([[None,None],[None,None]])        # delta
            bounds.extend([[1,59],[-59,59]])                # tau, phi
        # else:
        #     # bound for gamma_1 = +/-.05
        #     bounds=[[None,None],[None,None],[None,None],[None,None],[None,None],[-.05,.05],[1,59],[-59,59]]
        return bounds


    def exclude_regressors_by_bounds(self, bounds):
        for reg in self.exclude_regressors:
            if reg == 'beta_0':
                bounds[3] = [-1e-12,1e-12]
            elif reg == 'gamma_0':
                bounds[4] = [-1e-12,1e-12]
                bounds[5] = [-1e-12,1e-12]
            elif reg == ['delta_0',0]:
                bounds[6] = [-1e-12,1e-12]
            elif reg == ['delta_0',1]:
                bounds[7] = [-1e-12,1e-12]
        return bounds


    def get_model_mle_with_many_inits(self, shifted=None, tau_inits=[8,10,12,15,18,22,25,30,35,42,50], exclude_regressors=None):
        initial_conds = self.get_default_inits()
        self.exclude_regressors = exclude_regressors
        model_fit = self.get_model_mle(shifted=shifted, initial_conds=initial_conds.copy())
        # pdb.set_trace()
        self.evaluate_model(model_fit=model_fit, shifted=shifted)
        for i in enumerate(tau_inits):
            initial_conds[-2] = i[1]
            model_fit_new = self.get_model_mle(shifted=shifted, initial_conds=initial_conds.copy())
            self.evaluate_model(model_fit=model_fit_new, shifted=shifted)
            for n in range(len(model_fit)):
                if model_fit[n]['r_sq']['tot'] < model_fit_new[n]['r_sq']['tot']:
                    model_fit[n] = copy.deepcopy( model_fit_new[n] )
        return model_fit


    def fit_and_eval_reg_model_extended(self, n_perms=10, exclude_regressors=None):
        """
        exclude_regressors: (list) test model with some coefficients set to zero 
        """
        self.exclude_regressors = exclude_regressors
        self.model_fit = self.get_model_mle_with_many_inits(shifted=None, exclude_regressors=exclude_regressors)
        print('Testing model on circshifted data')
        # pdb.set_trace()
        self.get_circshift_behav_data(n_perms=n_perms)
        self.model_fit_shifted = [None]*n_perms
        for n in range(n_perms):
            print('Perm '+str(n))
            try:
                self.model_fit_shifted[n] = self.get_model_mle_with_many_inits(shifted=n, exclude_regressors=exclude_regressors)
            except:
                pdb.set_trace()


def get_summary_dict(expt_id, split_behav, data_dict, model_fit, model_fit_shifted, use_beh_labels):
    """
    summary preprocessing step to either plot regression output or feed it into other analyses
    """
    ro = reg_obj(exp_id=expt_id, 
                   data_dict=data_dict,
                   fig_dirs=futils.get_fig_dirs(expt_id),
                   split_behav=split_behav,
                   use_beh_labels=use_beh_labels)
    ro.downsample_in_time()
    ro.model_fit = model_fit #copy.deepcopy(model_fit)
    ro.model_fit_shifted = model_fit_shifted    

    f = plotting.get_model_fit_as_dict(ro.model_fit)
    rsq = plotting.get_model_fit_as_dict(f['r_sq'])
    cc = plotting.get_model_fit_as_dict(f['cc'])
    #pdb.set_trace()
    # get full and null-subtracted fit
    null_fit, dFF_tot = ro.get_null_subtracted_raster(just_null_model=True)
    ro.params['use_beh_labels'] = True
    dFF_fit, resid_tot = ro.get_null_subtracted_raster(just_null_model=False)
    #pdb.set_trace()
    
    # rescale fits to original dR/R
    dFF_tot_base = ro.unnormalize_rows_euc(dFF_tot, 0*ro.data_dict['mu'], ro.data_dict['sig'])
    dFF_fit_base = ro.unnormalize_rows_euc(dFF_fit-null_fit, 0*ro.data_dict['mu'], ro.data_dict['sig'])
    dFF_resid_base = ro.unnormalize_rows_euc(resid_tot, 0*ro.data_dict['mu'], ro.data_dict['sig'])

    # show rescaled null-subtracted fit
    #reload(plotting)
    dict_tmp = copy.deepcopy(ro.data_dict)
    dict_tmp['dFF'] = dFF_tot_base
    dict_tmp['dFF_fit'] = dFF_fit_base
    dict_tmp['dFF_resid_base'] = dFF_resid_base
    dict_tmp['dFF_resid'] = resid_tot
    dict_tmp['behavior'][dict_tmp['behavior']>0] = 1
    
    summary_dict = {'expt_id':expt_id, 'ro':ro, 'data_dict':dict_tmp, 'f':f, 'rsq':rsq, 'cc':cc}
    return summary_dict

