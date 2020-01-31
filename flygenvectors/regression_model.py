import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
import pdb



def null_reg_model(params, data):
    a0 = params[0]
    a1 = params[1]
    [t_exp, time, ball, dFF] = data
    lin_piece = a0 + a1*time
    dFF_fit =  np.squeeze(lin_piece[:-len(t_exp)+1])
    return dFF_fit

def fit_reg_null(params, data):
    [t_exp, time, ball, dFF] = data
    dFF_fit = null_reg_model(params, data)
    obj = ((dFF[len(t_exp)-1:]-dFF_fit)**2).sum()
    return obj



def tau_reg_model(params, data):
    a0 = params[0]
    a1 = params[1]
    b0 = params[2]
    b1 = params[3]
    tau = params[4]
    [t_exp, time, ball, dFF] = data
    kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
    x_c = np.convolve(kern,ball,'valid')
    ts = np.squeeze(time[:len(x_c)]/time[len(x_c)])

    lin_piece = a0 + a1*time
    dFF_fit =  np.squeeze(lin_piece[:len(x_c)]) + b0*x_c + b1*(x_c*ts )
    return dFF_fit

def fit_reg(params, data):
    [t_exp, time, ball, dFF] = data
    dFF_fit = tau_reg_model(params, data)
    obj = ((dFF[len(t_exp)-1:]-dFF_fit)**2).sum()
    return obj

def estimate_SINGLE_neuron_behav_reg_model(dFF, ball, time, scanRate, initial_conds=[]):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    tauLim = 100*scanRate
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/scanRate

    ball = ball
    time = time-time[0]
    # bnds = ((-10, 10),(-10, 10),(-10, 10),(-10, 10), (0, 100))
    tau_max = 100

    if not initial_conds:
        initial_conds=[[0,.0001,0.5,-1,1],
                        [0,.0001,0.5,-1,99],
                        [0,.0001,1,-0.5,1]]

    data = [t_exp, time, ball, dFF]
    fn_min = np.inf
    for k in range(len(initial_conds)):
        res_tmp = minimize(fit_reg, initial_conds[k], data, method='SLSQP') #, bounds=bnds)
        if ((res_tmp.fun<fn_min) and (res_tmp['x'][-1]>0)):
            res = res_tmp.copy()
            fn_min = res['fun']
            if (res['x'][-1]>tau_max): res['x'][-1]=tau_max
    # pdb.set_trace()
    return res['x']


def estimate_neuron_behav_reg_model(data_dict, initial_conds=[]):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    tauLim = 100*data_dict['scanRate']
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']

    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    # bnds = ((-10, 10),(-10, 10),(-10, 10),(-10, 10), (0, 100))
    tau_max = 100

    if not initial_conds:
        initial_conds=[[0,.0001,0.5,-1,1],
                        [0,.0001,0.5,-1,99],
                        [0,.0001,0.5,-1,99],
                        [0,.0001,1,-0.5,20],
                        [0,.0001,1,-0.5,1]]

    model_fit = []
    # null_fit = []
    for j in range(data_dict['dFF'].shape[0]):
        if not np.mod(j,100):
            print(j,end=' ')
        dFF = data_dict['dFF'][j,:]
        data = [t_exp, time, ball, dFF]
        
        fn_min = np.inf
        for k in range(len(initial_conds)):
            res_tmp = minimize(fit_reg, initial_conds[k], data, method='SLSQP') #, bounds=bnds)
            if ((res_tmp.fun<fn_min) and (res_tmp['x'][-1]>0)):
                res = res_tmp.copy()
                fn_min = res['fun']
                if (res['x'][-1]>tau_max): res['x'][-1]=tau_max
        res_null = minimize(fit_reg_null, initial_conds[0][:2], data, method='SLSQP') #, bounds=bnds[:2])

        dFF_fit = tau_reg_model(res['x'], data)
        dFF_fit_null = null_reg_model(res_null.x, data)
        dFF_fit_linpart = null_reg_model(res['x'][:2], data)
        

        # r squared
        SS_tot = ( (dFF[M-1:]-dFF.mean()-dFF_fit_linpart)**2 )
        SS_res = ( (dFF[M-1:]-dFF_fit)**2 )
        SS_tot_0 = ( (dFF[M-1:]-dFF.mean())**2 )
        SS_res_0 = ( (dFF[M-1:]-dFF_fit_null)**2 )
        stat = stats.wilcoxon(SS_res_0,SS_res)

        # CC
        CC = np.corrcoef(dFF[M-1:], dFF_fit)[0,1] 
        CC_null = np.corrcoef(dFF[M-1:], dFF_fit_null)[0,1] 

        # collecting
        d = {}
        d['alpha_0_null'] = res_null.x[0]
        d['alpha_1_null'] = res_null.x[1]
        d['alpha_0'] = res['x'][0]
        d['alpha_1'] = res['x'][1]
        d['beta_0'] = res['x'][2]
        d['beta_1'] = res['x'][3]
        d['tau'] = res['x'][4]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        d['success'] = res['success']
        model_fit.append(d)
    return model_fit
