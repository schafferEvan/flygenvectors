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

    if (tau>=0):
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        x_c = np.convolve(kern,ball,'valid')
    else:
        kern = (1/np.sqrt(-tau))*np.exp(t_exp/tau)
        x_c = np.convolve(kern[::-1],ball,'valid')
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
    # not used -------------
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
    # p = np.matmul( np.matmul(dFF,D.T), Dinv )
    p = (dFF@D.T) @ Dinv
    dFF_fit = p@D #np.matmul(p,D)
    obj = ((dFF-dFF_fit)**2).sum()
    return p, obj

def get_model_fit(data_dict, model_fit, n):
    d = model_fit[n]
    p = [d['alpha_0'], d['alpha_1'], d['beta_0'], d['beta_1']]
    tau = d['tau']
    phi = int(d['phi'])

    tauLimSec = 30
    phaseLimSec = 20
    tauLim = tauLimSec*data_dict['scanRate']
    L = int(phaseLimSec*data_dict['scanRate'])
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    ts_full = np.squeeze(time[:-M+1])
    kern = (1/np.sqrt(abs(tau)))*np.exp(-t_exp/abs(tau))
    if (tau>=0):
        x_c_full = np.convolve(kern,ball,'valid')
    else:
        x_c_full = np.convolve(kern[::-1],ball,'valid')
    x_c = x_c_full[L:-L]
    ts = ts_full[L:-L]-ts_full[L]
    D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
    # Dinv = np.linalg.inv( np.matmul(D,D.T))
    if (tau>=0):
        dFF_full = data_dict['dFF'][n,M-1:]
    else:
        dFF_full = data_dict['dFF'][n,:-M+1]
    # dFF slides past beh with displacement phi
    if(phi==L):
        dFF = dFF_full[L+phi:]
    else:
        dFF = dFF_full[L+phi:-(L-phi)]
    dFF_fit = np.matmul(p,D)
    return dFF_fit, dFF


def get_model_fit_gauss(data_dict, model_fit, n):
    d = model_fit[n]
    p = [d['alpha_0'], d['alpha_1'], d['beta_0'], d['beta_1']]
    tau = d['tau']
    phi = int(d['phi'])

    tauLimSec = 100
    phaseLimSec = 20
    tauLim = tauLimSec*data_dict['scanRate']
    L = int(phaseLimSec*data_dict['scanRate'])
    M = round(-tauLim*np.log(0.1)).astype(int)
    mu = .5*M/data_dict['scanRate']
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']-data_dict['behavior'].mean()
    time = data_dict['time']-data_dict['time'][0]
    ts_full = np.squeeze(time)

    kern = np.exp(-0.5*((t_exp-mu)/tau)**2)
    kern /= kern.sum()
    x_c_full = np.convolve(kern, ball, 'same')

    x_c = x_c_full[L:-L]
    ts = ts_full[L:-L]-ts_full[L]
    D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )

    rate_full = data_dict['rate'][n,:]
    if(phi==L):
        rate = rate_full[L+phi:]
    else:
        rate = rate_full[L+phi:-(L-phi)]
    rate_fit = np.matmul(p,D)
    return rate_fit, rate


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


def estimate_neuron_behav_reg_model_taulist(data_dict,both_directions=True):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    tauLimSec = 100
    tauLim = tauLimSec*data_dict['scanRate']
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    ts = np.squeeze(time[:-M+1])
    tauList = np.logspace(0,np.log10(tauLimSec),num=200)
    if both_directions:
        tauList = np.concatenate((-tauList[::-1],tauList))

    tau_star = np.zeros(data_dict['dFF'].shape[0])
    fn_min = np.inf*np.ones(data_dict['dFF'].shape[0])
    P = np.zeros((data_dict['dFF'].shape[0],4))

    for i in range(len(tauList)):
        if not np.mod(i,10): print(i, end=' ')
        # [t_exp, time, ball, dFF] = data
        tau = abs(tauList[i])
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        if (tauList[i]>=0):
            x_c = np.convolve(kern,ball,'valid')
        else:
            x_c = np.convolve(kern[::-1],ball,'valid')
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        Dinv = np.linalg.inv( np.matmul(D,D.T))

        for j in range(data_dict['dFF'].shape[0]):
            if (tauList[i]>=0):
                dFF = data_dict['dFF'][j,M-1:]
            else:
                dFF = data_dict['dFF'][j,:-M+1]
            data = [D, Dinv, dFF]
            p, obj = fit_reg_linear(data)
            if (obj<fn_min[j]):
                tau_star[j] = tauList[i]
                P[j,:] = p
                fn_min[j] = obj

    model_fit = []
    for j in range(data_dict['dFF'].shape[0]):
        tau = abs(tau_star[j])
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        if (tau_star[j]>=0):
            dFF = data_dict['dFF'][j,M-1:]
            x_c = np.convolve(kern,ball,'valid')
        else:
            dFF = data_dict['dFF'][j,:-M+1]
            x_c = np.convolve(kern[::-1],ball,'valid')
        p = P[j,:]
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        dFF_fit = np.matmul(p,D)
        p_lin = p.copy()
        p_lin[2:] = 0
        dFF_fit_linpart = np.matmul(p_lin,D)
        dFF_without_linpart = dFF-dFF_fit_linpart

        # null model (linear fit, after subtracting linear part of full model)
        D_n = np.array( [np.ones(len(x_c)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF_without_linpart])
        dFF_fit_null = np.squeeze(np.matmul(p_n,D_n))
        
        # r squared: comparing model to null (linear)
        SS_tot = ( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = SS_tot #( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF_without_linpart-dFF_fit_null)**2 )
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


def estimate_neuron_behav_reg_model_taulist_shiftlist(data_dict,both_directions=True):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    tauLimSec = 100
    phaseLimSec = 20
    use_beta_1 = False
    tauLim = tauLimSec*data_dict['scanRate']
    L = int(phaseLimSec*data_dict['scanRate'])
    M = round(-tauLim*np.log(0.2)).astype(int)
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']
    time = data_dict['time']-data_dict['time'][0]
    # print(-M-L+1)
    ts_full = np.squeeze(time[:-M+1])
    tauList = np.logspace(0,np.log10(tauLimSec),num=100)
    phiList = np.linspace(-L,L, num=2*phaseLimSec+1 ).astype(int)
    if both_directions:
        tauList = np.concatenate((-tauList[::-1],tauList))

    tau_star = np.zeros(data_dict['dFF'].shape[0])
    phi_star = np.zeros(data_dict['dFF'].shape[0])
    fn_min = np.inf*np.ones(data_dict['dFF'].shape[0])
    P = np.zeros((data_dict['dFF'].shape[0],4))

    # fit model -
    # for each value of tau and phi, check if pInv solution is better than previous
    for i in range(len(tauList)):
        if not np.mod(i,10): print(i, end=' ')
        # [t_exp, time, ball, dFF] = data
        tau = abs(tauList[i])
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        if (tauList[i]>=0):
            x_c_full = np.convolve(kern,ball,'valid')
        else:
            x_c_full = np.convolve(kern[::-1],ball,'valid')
        
        x_c = x_c_full[L:-L]
        ts = ts_full[L:-L]-ts_full[L]
        if use_beta_1:
            D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        else:
            D = np.array( [np.ones(len(x_c)), ts, x_c] )
        Dinv = np.linalg.inv( np.matmul(D,D.T))

        for j in range(len(phiList)):
            phi = phiList[j]
            for n in range(data_dict['dFF'].shape[0]):
                if (tauList[i]>=0):
                    dFF_full = data_dict['dFF'][n,M-1:]
                else:
                    dFF_full = data_dict['dFF'][n,:-M+1]
                
                # dFF slides past beh with displacement phi
                if(phi==L):
                    dFF = dFF_full[L+phi:]
                else:
                    dFF = dFF_full[L+phi:-(L-phi)]
                data = [D, Dinv, dFF]
                p, obj = fit_reg_linear(data)
                if (obj<fn_min[n]):
                    tau_star[n] = tauList[i]
                    phi_star[n] = phiList[j]
                    if use_beta_1:
                        P[n,:] = p
                    else:
                        P[n,:-1] = p
                    fn_min[n] = obj

    # regenerate fit from best parameters and evaluate model
    model_fit = []
    for n in range(data_dict['dFF'].shape[0]):
        tau = abs(tau_star[n])
        phi = int(phi_star[n])
        kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        if (tau_star[n]>=0):
            dFF_full = data_dict['dFF'][n,M-1:]
            x_c_full = np.convolve(kern,ball,'valid')
        else:
            dFF_full = data_dict['dFF'][n,:-M+1]
            x_c_full = np.convolve(kern[::-1],ball,'valid')
        p = P[n,:]
        x_c = x_c_full[L:-L]
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        dFF_fit = np.matmul(p,D)
        p_lin = p.copy()
        p_lin[2:] = 0
        dFF_fit_linpart = np.matmul(p_lin,D)
        dFF_without_linpart = dFF-dFF_fit_linpart

        # null model (linear fit, after subtracting linear part of full model)
        D_n = np.array( [np.ones(len(x_c)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF_without_linpart])
        dFF_fit_null = np.squeeze(np.matmul(p_n,D_n))
        
        # r squared: comparing model to null (linear)
        SS_tot = ( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = SS_tot #( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF_without_linpart-dFF_fit_null)**2 )
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
        d['tau'] = tau_star[n]
        d['phi'] = phi_star[n]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        d['success'] = True #res['success']
        model_fit.append(d)
    return model_fit


def get_conv_array(data_dict, dict_var='behavior', kern_type='gauss', conv_params=[] ):
    # generate lookup table matrix of convolved output for any time and any kernel shape
    if not conv_params:
        conv_params = {
            'sigLimSec': 100,
            'sig_min_pow': -1,
            'sig_num': 100}
    sigLim = conv_params['sigLimSec']*data_dict['scanRate']
    M = round(-sigLim*np.log(0.01)).astype(int)
    conv_array = np.zeros( (conv_params['sig_num'], len(data_dict[dict_var])+M-1) )
    mu = .5*M/data_dict['scanRate']
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    tseries = np.squeeze(data_dict[dict_var].astype(float))
    tseries -= tseries.mean()
    time = data_dict['time']-data_dict['time'][0]
    tauList = np.logspace(conv_params['sig_min_pow'], np.log10(conv_params['sigLimSec']),num=conv_params['sig_num'])
    for i in range(conv_params['sig_num']):
        tau = tauList[i]
        if kern_type=='gauss':
            kern = np.exp(-0.5*((t_exp-mu)/tau)**2)
        elif kern_type=='exp':
            kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern /= kern.sum()
        conv_array[i,:] = np.convolve(kern,tseries,'full')
    return conv_array, tauList


def get_poly_reg_coeffs(conv_array, tauList, ex_num, poly_order=8):
    # linear regression to approximate time slice of conv_array with a polynomial
    y = conv_array[:,ex_num]
    X = np.ones((poly_order+1,len(tauList)))
    for i in range(poly_order):
        X[i,:] = tauList**(poly_order-i)
    A = y@X.T@np.linalg.inv(X@X.T)
    return A, X


def estimate_neuron_behav_reg_model_taulist_shiftlist_gauss(data_dict):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    sigLimSec = 100
    phaseLimSec = 20
    use_beta_1 = False
    sigLim = sigLimSec*data_dict['scanRate']
    L = int(phaseLimSec*data_dict['scanRate'])
    M = round(-sigLim*np.log(0.1)).astype(int)
    mu = .5*M/data_dict['scanRate']
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']-data_dict['behavior'].mean()
    time = data_dict['time']-data_dict['time'][0]
    # print(-M-L+1)
    ts_full = np.squeeze(time)
    tauList = np.logspace(-1,np.log10(sigLimSec),num=100)
    phiList = np.linspace(-L,L, num=2*phaseLimSec+1 ).astype(int)
    
    tau_star = np.zeros(data_dict['rate'].shape[0])
    phi_star = np.zeros(data_dict['rate'].shape[0])
    fn_min = np.inf*np.ones(data_dict['rate'].shape[0])
    P = np.zeros((data_dict['rate'].shape[0],4))
    P_tot = np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList),4))
    obj_tot = np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList)))

    # fit model -
    # for each value of tau and phi, check if pInv solution is better than previous
    for i in range(len(tauList)):
        if not np.mod(i,10): print(i, end=' ')
        # [t_exp, time, ball, dFF] = data
        tau = tauList[i]
        # kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern = np.exp(-0.5*((t_exp-mu)/tau)**2)
        kern /= kern.sum()
        x_c_full = np.convolve(kern,ball,'same')
        
        x_c = x_c_full[L:-L]
        ts = ts_full[L:-L]-ts_full[L]
        if use_beta_1:
            D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        else:
            D = np.array( [np.ones(len(x_c)), ts, x_c] )
        Dinv = np.linalg.inv( np.matmul(D,D.T))

        for j in range(len(phiList)):
            phi = phiList[j]
            for n in range(data_dict['rate'].shape[0]):
                dFF_full = data_dict['rate'][n,:]
                
                # dFF slides past beh with displacement phi
                if(phi==L):
                    dFF = dFF_full[L+phi:]
                else:
                    dFF = dFF_full[L+phi:-(L-phi)]
                data = [D, Dinv, dFF]
                p, obj = fit_reg_linear(data)
                obj_tot[n,i,j] = obj
                if use_beta_1:
                    P_tot[n,i,j,:] = p
                else:
                    P_tot[n,i,j,:-1] = p
                if (obj<fn_min[n]):
                    tau_star[n] = tauList[i]
                    phi_star[n] = phiList[j]
                    if use_beta_1:
                        P[n,:] = p
                    else:
                        P[n,:-1] = p
                    fn_min[n] = obj

    # regenerate fit from best parameters and evaluate model
    model_fit = []
    for n in range(data_dict['rate'].shape[0]):
        tau = tau_star[n]
        phi = int(phi_star[n])
        #kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern = np.exp(-0.5*((t_exp-mu)/tau)**2)
        kern /= kern.sum()
        dFF_full = data_dict['rate'][n,:]
        x_c_full = np.convolve(kern,ball,'same')
        p = P[n,:]
        x_c = x_c_full[L:-L]
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]
        D = np.array( [np.ones(len(x_c)), ts, x_c, ts*x_c/ts[-1]] )
        dFF_fit = np.matmul(p,D)
        p_lin = p.copy()

        # dFF is now assumed to be de-trended already from low-rank model
        # p_lin[2:] = 0
        # dFF_fit_linpart = np.matmul(p_lin,D)
        # dFF_without_linpart = dFF-dFF_fit_linpart

        # null model (linear fit, after subtracting linear part of full model)
        D_n = np.array( [np.ones(len(x_c)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF]) #dFF_without_linpart])
        dFF_fit_null = np.squeeze(np.matmul(p_n,D_n))
        
        # r squared: comparing model to null (linear)
        SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = SS_tot #( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF-dFF_fit_null)**2 ) #( (dFF_without_linpart-dFF_fit_null)**2 )
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
        d['tau'] = tau_star[n]
        d['phi'] = phi_star[n]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        d['success'] = True #res['success']
        d['P_tot'] = P_tot[n,:,:,:]
        d['obj_tot'] = obj_tot[n,:,:]
        model_fit.append(d)
    return model_fit


def estimate_neuron_behav_reg_model_extended(data_dict, opts):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    # this is an extension of ".taulist_shiftlist_gauss", with two additions:
    #       (1) merges gauss/exp kernel methods with option for either
    #       (2) allows for arbitrarily many special timeseries inputs. previously just time, now time, feeding, "binary hunger", etc
    extraRegressors = opts['extraRegressors']
    tot_n_regressors = 3+extraRegressors.shape[0]
    sigLimSec = opts['sigLimSec'] #100
    phaseLimSec = opts['phaseLimSec'] #20
    # use_beta_1 = False
    sigLim = sigLimSec*data_dict['scanRate']
    L = int(phaseLimSec*data_dict['scanRate'])
    M = round(-sigLim*np.log(0.1)).astype(int)
    mu = .5*M/data_dict['scanRate']
    t_exp = np.linspace(1,M,M)/data_dict['scanRate']
    ball = data_dict['behavior']-data_dict['behavior'].mean()
    time = data_dict['time']-data_dict['time'][0]
    # print(-M-L+1)
    ts_full = np.squeeze(time)
    tauList = np.logspace(-1,np.log10(sigLimSec),num=100)
    phiList = np.linspace(-L,L, num=2*phaseLimSec+1 ).astype(int)
    
    tau_star = np.zeros(data_dict['rate'].shape[0])
    phi_star = np.zeros(data_dict['rate'].shape[0])
    fn_min = np.inf*np.ones(data_dict['rate'].shape[0])
    P = np.zeros((data_dict['rate'].shape[0],tot_n_regressors))
    P_tot = np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList),tot_n_regressors))
    obj_tot = np.zeros((data_dict['rate'].shape[0],len(tauList),len(phiList)))

    # fit model -
    # for each value of tau and phi, check if pInv solution is better than previous
    for i in range(len(tauList)):
        if not np.mod(i,10): print(i, end=' ')
        # [t_exp, time, ball, dFF] = data
        tau = tauList[i]
        # kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern = np.exp(-0.5*((t_exp-mu)/tau)**2)
        kern /= kern.sum()
        x_c_full = np.convolve(kern,ball,'same')                      # ******* fix 'same' and 'gauss/exp' option *******
        
        x_c = x_c_full[L:-L]
        ts = ts_full[L:-L]-ts_full[L]
        
        D = np.zeros((3+extraRegressors.shape[0], len(x_c)))
        D[:3,:] = np.array( [np.ones(len(x_c)), ts, x_c] )
        for j in range(extraRegressors.shape[0]):
            D[3+j,:] = extraRegressors[j,L:-L]
        Dinv = np.linalg.inv( np.matmul(D,D.T))

        for j in range(len(phiList)):
            phi = phiList[j]
            for n in range(data_dict['rate'].shape[0]):
                dFF_full = data_dict['rate'][n,:]                      #  ****** this is fine if detrending is removed from hunger processing
                
                # dFF slides past beh with displacement phi
                if(phi==L):
                    dFF = dFF_full[L+phi:]
                else:
                    dFF = dFF_full[L+phi:-(L-phi)]
                data = [D, Dinv, dFF]
                p, obj = fit_reg_linear(data)
                obj_tot[n,i,j] = obj
                P_tot[n,i,j,:] = p
                if (obj<fn_min[n]):
                    tau_star[n] = tauList[i]
                    phi_star[n] = phiList[j]
                    P[n,:] = p
                    fn_min[n] = obj

    # regenerate fit from best parameters and evaluate model
    model_fit = []
    for n in range(data_dict['rate'].shape[0]):
        tau = tau_star[n]
        phi = int(phi_star[n])
        #kern = (1/np.sqrt(tau))*np.exp(-t_exp/tau)
        kern = np.exp(-0.5*((t_exp-mu)/tau)**2)                                 # ******* kern options
        kern /= kern.sum()
        dFF_full = data_dict['rate'][n,:]
        x_c_full = np.convolve(kern,ball,'same')                                # fix 'same'
        p = P[n,:]
        x_c = x_c_full[L:-L]
        if(phi==L):
            dFF = dFF_full[L+phi:]
        else:
            dFF = dFF_full[L+phi:-(L-phi)]
        D = np.zeros((3+extraRegressors.shape[0], len(x_c)))
        D[:3,:] = np.array( [np.ones(len(x_c)), ts, x_c] )
        for j in range(extraRegressors.shape[0]):
            D[3+j,:] = extraRegressors[j,L:-L]
        dFF_fit = np.matmul(p,D)
        p_lin = p.copy()

        # null model (linear fit, after subtracting linear part of full model)
        D_n = np.array( [np.ones(len(x_c)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF]) #dFF_without_linpart])
        dFF_fit_null = np.squeeze(np.matmul(p_n,D_n))
        
        # r squared: comparing model to null (linear)
        SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = SS_tot #( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF-dFF_fit_null)**2 ) #( (dFF_without_linpart-dFF_fit_null)**2 )
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
        d['beta_1'] = p[3:]
        d['tau'] = tau_star[n]
        d['phi'] = phi_star[n]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        d['success'] = True #res['success']
        d['P_tot'] = P_tot[n,:,:,:]
        d['obj_tot'] = obj_tot[n,:,:]
        model_fit.append(d)
    return model_fit


def get_model_rsq_and_pval(data_dict, model_fit):
    # recompute rsq and pval
    import copy
    model_fit_null = copy.deepcopy( model_fit )
    phaseLimSec = 20
    L = int(phaseLimSec*data_dict['scanRate'])
    time = data_dict['time']-data_dict['time'][0]
    ts_full = np.squeeze(time)  
    ts = ts_full[L:-L]-ts_full[L]

    model_eval = []
    for n in range(data_dict['dFF'].shape[0]):

        # compute fit for full model
        # p = [model_fit[n]['alpha_0'], model_fit[n]['alpha_1'], model_fit[n]['beta_0'], model_fit[n]['beta_1'] ]
        # tau = model_fit[n]['tau']
        # phi = int(model_fit[n]['phi'])
        dFF_fit, dFF = get_model_fit_gauss(data_dict, model_fit, n)

        # recompute null model and get fit
        D_n = np.array( [np.ones(len(ts)), ts] )
        Dinv_n = np.linalg.inv( np.matmul(D_n,D_n.T))
        p_n, obj = fit_reg_linear([D_n, Dinv_n, dFF]) #dFF_without_linpart])
        model_fit_null[n]['alpha_0'] = p_n[0,0]
        model_fit_null[n]['alpha_1'] = p_n[0,1]
        model_fit_null[n]['beta_0'] = 0
        model_fit_null[n]['beta_1'] = 0
        dFF_fit_null, dFF = get_model_fit_gauss(data_dict, model_fit_null, n)
        
        # r squared: comparing model to null (linear)
        SS_tot = ( (dFF-dFF.mean())**2 ) #( (dFF_without_linpart-dFF_without_linpart.mean())**2 )
        SS_res = ( (dFF-dFF_fit)**2 )
        SS_tot_0 = SS_tot #( (dFF-dFF.mean())**2 )
        SS_res_0 = ( (dFF-dFF_fit_null)**2 ) #( (dFF_without_linpart-dFF_fit_null)**2 )
        stat = stats.wilcoxon(SS_res_0,SS_res)

        # CC
        CC = np.corrcoef(dFF, dFF_fit)[0,1] 
        CC_null = np.corrcoef(dFF, dFF_fit_null)[0,1] 

        d = {}
        d['alpha_0_null'] = p_n[0,0]
        d['alpha_1_null'] = p_n[0,1]
        d['r_sq'] = 1-SS_res.sum()/SS_tot.sum()
        d['r_sq_null'] = 1-SS_res_0.sum()/SS_tot_0.sum()
        d['CC'] = CC
        d['CC_null'] = CC_null
        d['stat'] = stat
        model_eval.append(d)
        
    return model_eval


def flag_cells_by_model_params(param_dict, dict_tot, eval_tot):
    # for each dataset, flag candidates
    # typically used by passing bayes_model_tot_post as dict_tot
    # and model_eval_tot_post as eval_tot
    
    # generate distance threshold
    from scipy.stats import multivariate_normal
    data_dict = dict_tot['data_tot'][0]['data_dict']
    tmp = multivariate_normal(dict_tot['data_tot'][0]['data_dict']['aligned_centroids'][0,:], param_dict['sig'] ) 
    ref_pt = dict_tot['data_tot'][0]['data_dict']['aligned_centroids'][0,:].copy()
    ref_pt[0] += param_dict['sig']
    ref_val = tmp.pdf(ref_pt)

    example_array_tot = []
    # mag_order_tot = []
    p_th = 10**-5
    for nf in range(len(dict_tot['data_tot'])):
        data_dict = dict_tot['data_tot'][nf]['data_dict']
        model_fit = dict_tot['model_fit'][nf]
        
        N = data_dict['aligned_centroids'].shape[0]
        ex_ar = []
        for n in range(N):
            if ( (model_fit[n]['tau']>param_dict['tau_th']) and (model_fit[n]['phi']>param_dict['phi_th']) 
                and (model_fit[n]['tau']<param_dict['tau_max_th']) and (eval_tot[nf][n]['stat'][1]<p_th)):
                mvn = multivariate_normal(data_dict['aligned_centroids'][n,:], param_dict['sig'] )
                if mvn.pdf([param_dict['ref_x'],param_dict['ref_y'],param_dict['ref_z']])>ref_val:
                    ex_ar.append(n)
        example_array_tot.append( ex_ar )
        print(len(ex_ar),end=' ')
        
        # # get dynamic range of fits, and sort
        # mag = np.zeros(len(ex_ar))
        # for j in range(len(ex_ar)):
        #     dFF_fit, dFF = get_model_fit_gauss(data_dict, model_fit, ex_ar[j])
        #     mag[j] = dFF_fit.max() - dFF_fit.min()
        # mag_order = np.argsort(mag)
        # mag_order_tot.append( mag_order )
            
    return example_array_tot



