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

def fit_reg_taulist(params_hat, data_hat):
    [t_exp, time, ball, dFF, tau] = data_hat
    data = [t_exp, time, ball, dFF]
    params = np.concatenate((params_hat,[tau]))
    dFF_fit = tau_reg_model(params, data)
    obj = ((dFF[len(t_exp)-1:]-dFF_fit)**2).sum()
    return obj

def fit_reg_linear(data):
    # for a given tau, fit for other parameters is linear, solved by pseudoinverse
    [D, Dinv, dFF] = data
    dFF = np.expand_dims(dFF, axis=0)
    p = np.matmul( np.matmul(dFF,D.T), Dinv )
    dFF_fit = np.matmul(p,D)
    obj = ((dFF-dFF_fit)**2).sum()
    return p, obj

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


def estimate_neuron_behav_reg_model_taulist(data_dict):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    tauLimSec = 100
    tauLim = tauLimSec*data_dict['scanRate']
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    ts = np.squeeze(time[:-M+1])
    tauList = np.logspace(0,np.log10(tauLimSec),num=200)
    tau_star = np.zeros(data_dict['dFF'].shape[0])
    fn_min = np.inf*np.ones(data_dict['dFF'].shape[0])
    P = np.zeros((data_dict['dFF'].shape[0],4))

    for i in range(len(tauList)):
        if not np.mod(i,10): print(i, end=' ')
        # [t_exp, time, ball, dFF] = data
        tau = tauList[i]
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        x_c = np.convolve(kern,ball,'valid')
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        Dinv = np.linalg.inv( np.matmul(D,D.T))

        for j in range(data_dict['dFF'].shape[0]):
            dFF = data_dict['dFF'][j,M-1:]
            data = [D, Dinv, dFF]
            p, obj = fit_reg_linear(data)
            if (obj<fn_min[j]):
                tau_star[j] = tauList[i]
                P[j,:] = p
                fn_min[j] = obj

    # IN LOOP BELOW, NEED TO USE TAU_STAR[j] AND P TO COMPUTE STUFF AND SAVE IN CORRECT FORMAT

    model_fit = []
    for j in range(data_dict['dFF'].shape[0]):
        dFF = data_dict['dFF'][j,M-1:]
        p = P[j,:]
        tau = tau_star[j]
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        x_c = np.convolve(kern,ball,'valid')
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        dFF_fit = np.matmul(p,D)
        p_lin = p.copy()
        p_lin[2:] = 0
        dFF_fit_linpart = np.matmul(p_lin,D)

        # null model
        D_n = np.array( [np.ones(len(x_c)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF])
        dFF_fit_null = np.squeeze(np.matmul(p_n,D_n))
        
        # r squared
        # pdb.set_trace()
        # print(dFF_fit_linpart.shape)
        # print(dFF.shape)
        # print(dFF_fit.shape)
        # print(dFF_fit_null.shape)
        SS_tot = ( (dFF-dFF.mean()-dFF_fit_linpart)**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = ( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF-dFF_fit_null)**2 )
        stat = stats.wilcoxon(SS_res_0,SS_res)

        # CC
        CC = np.corrcoef(dFF, dFF_fit)[0,1] 
        CC_null = np.corrcoef(dFF, dFF_fit_null)[0,1] 

        # collecting
        d = {}
        d['alpha_0_null'] = p_n[0,0]
        d['alpha_1_null'] = p_n[0,1]
        d['alpha_0'] = p[0]
        d['alpha_1'] = p[1]
        d['beta_0'] = p[2]
        d['beta_1'] = p[3]
        d['tau'] = tau_star[j]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        d['success'] = True #res['success']
        model_fit.append(d)
    return model_fit


