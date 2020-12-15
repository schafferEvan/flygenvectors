import os
import sys
# sys.path.insert(0, '../flygenvectors/')
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import copy
import data as dataUtils
import flygenvectors.utils as futils
import regression_model as reg_model
import pdb



class lr_obj:
    def __init__(self, activity='dFF', exp_id=None, data_dict={}, fig_dirs={}, split_behav=False, iters=5, n_components=100, use_p2=False):
        self.data_dict = data_dict
        self.exp_id = exp_id
        self.fig_dirs = fig_dirs
        self.activity=activity # valid values: {'dFF', 'rate'}, determines what trace is used for fit
        self.reg_obj = reg_model.reg_obj(exp_id=exp_id, 
                        data_dict=data_dict,
                        fig_dirs=fig_dirs,
                        split_behav=split_behav)

        self.iters = iters #1500
        self.n_components = n_components
        self.use_p2 = use_p2
        self.model_fit = {}


    def fit_model(self):
        '''
        Fits model of the form F = AT + UV
        '''
        self.model_fit = {}
        self.model_fit['a1'] = np.zeros((self.data_dict[self.activity].shape[0],3))
        self.model_fit['a2'] = np.zeros((self.data_dict[self.activity].shape[0],2))
        self.model_fit['U'] = np.zeros((self.data_dict[self.activity].shape[0],self.n_components))
        self.model_fit['V'] = np.zeros((self.data_dict[self.activity].shape[1],self.n_components))
        self.model_fit = self.fit_p1(self.data_dict[self.activity], self.model_fit)

        pca = PCA(n_components=self.n_components)
        F = self.get_F_for_PCA(self.data_dict[self.activity], self.model_fit)
        pca.fit(F.T)
        self.model_fit['V'] = pca.transform(F.T)
        self.model_fit['U'] = pca.components_.T
        # model_fit['UV'] = np.matmul(U,V.T) #pca.inverse_transform(LR).T
        self.var_exp_init = pca.explained_variance_ratio_

        if(self.use_p2): self.model_fit = self.fit_p2(self.data_dict[self.activity], self.model_fit)

        self.mse = np.zeros(self.iters)
        for i in range(self.iters):
            print(i,end=' ')
            self.model_fit = self.update_P1_and_U(self.data_dict[self.activity], self.model_fit)
            if(self.use_p2):
                self.model_fit = self.update_P1_and_P2(self.data_dict[self.activity], self.model_fit)
                for m in range(self.n_components):
                    self.model_fit['U'][:,m] /= np.sqrt((self.model_fit['U'][:,m]**2).sum())
                self.model_fit = self.update_V(self.data_dict[self.activity], self.model_fit)
            else:
                self.model_fit = self.update_V_at_once(self.data_dict[self.activity], self.model_fit)
            
            self.dFF_fit, self.mse[i] = self.model_eval(self.data_dict[self.activity], self.model_fit)
        print('\r MSE after 1 iter = '+str(self.mse[0]))
        print('MSE after '+str(self.iters-1)+' iters = '+str(self.mse[-2]))
        print('MSE after '+str(self.iters)+' iters = '+str(self.mse[-1]))
            


    def fit_p1(self, dFF, model_fit):
        a2 = model_fit['a2']
        UV = np.matmul(model_fit['U'],model_fit['V'].T) #model_fit['UV']
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        #     atime = np.array([np.ones(t.shape), t])
        Xinv = np.linalg.inv( np.matmul(atime,atime.T))
        for i in range(dFF.shape[0]):
            f = dFF[i,:]
            P2i = 1 + np.matmul( a2[i,:], time )
            y = P2i*UV[i,:]
            fhat = f-y
            model_fit['a1'][i,:] = np.matmul( np.matmul(fhat, atime.T), Xinv )
            #model_fit['a1'][i,:2] = np.matmul( np.matmul(fhat, atime.T), Xinv )
        return model_fit


    def get_F_for_PCA(self, dFF, model_fit):
        a1 = model_fit['a1']
        a2 = model_fit['a2']
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        F = np.zeros(dFF.shape)
        for i in range(dFF.shape[0]):
            f = dFF[i,:]
            P1i = np.matmul( a1[i,:], atime )
            P2i = 1 + np.matmul( a2[i,:], time )
            F[i,:] = (f-P1i)/P2i
        return F


    def fit_UV(self, dFF, model_fit):
        a1 = model_fit['a1']
        a2 = model_fit['a2']
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        F = np.zeros(dFF.shape)
        for i in range(dFF.shape[0]):
            f = dFF[i,:]
            P1i = np.matmul( a1[i,:], atime )
            P2i = 1 + np.matmul( a2[i,:], time )
            F[i,:] = (f-P1i)/P2i
        return model_fit


    def fit_p2(self, dFF, model_fit, order=1):
        a1 = model_fit['a1']
        UV = np.matmul(model_fit['U'],model_fit['V'].T) #model_fit['UV']
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        for i in range(dFF.shape[0]):
            f = dFF[i,:]
            P1i = np.matmul( a1[i,:], atime )
            fhat = f-P1i-UV[i,:]
            if(order==1): 
                x = np.expand_dims(UV[i,:]*t,axis=0)
            else:
                x = np.array([UV[i,:]*t, UV[i,:]*t**2])
            Xinv = np.linalg.inv( np.matmul(x,x.T))
            if(order==1): 
                model_fit['a2'][i,0] = np.matmul( np.matmul(fhat, x.T), Xinv )
            else:
                model_fit['a2'][i,:] = np.matmul( np.matmul(fhat, x.T), Xinv )
            #model_fit['a2'][i,0] = np.matmul( np.matmul(fhat, x.T), Xinv )
        return model_fit


    def model_eval(self, dFF, model_fit):
        a1 = model_fit['a1']
        a2 = model_fit['a2']
        UV = np.matmul(model_fit['U'],model_fit['V'].T) #model_fit['UV']
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        dFF_fit = np.zeros(dFF.shape)
        for i in range(dFF.shape[0]):
            P1i = np.matmul( a1[i,:], atime )
            P2i = 1 + np.matmul( a2[i,:], time )
            dFF_fit[i,:] = P1i + P2i*UV[i,:]
        mse = ((dFF-dFF_fit)**2).sum()/np.prod(dFF.shape)
        return dFF_fit, mse


    def update_P1_and_U(self, dFF, model_fit):
        a2 = model_fit['a2']
        V = model_fit['V'].T
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        P2iV = np.zeros(V.shape)
        for i in range(dFF.shape[0]):
            f = dFF[i,:]
            P2i = 1 + np.matmul( a2[i,:], time )
            for j in range(V.shape[0]):
                P2iV[j,:] = P2i*V[j,:]
            X = np.concatenate((atime,P2iV),axis=0)
            Xinv = np.linalg.inv( np.matmul(X,X.T))
            phi = np.matmul( np.matmul(f, X.T), Xinv )
            model_fit['a1'][i,:] = phi[:3]
            model_fit['U'][i,:] = phi[3:]
        return model_fit


    def update_P1_and_P2(self, dFF, model_fit, p2_order=1):
        UV = np.matmul(model_fit['U'],model_fit['V'].T)
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        atime = np.array([np.ones(t.shape), t, t**2])
        for i in range(dFF.shape[0]):
            fhat = dFF[i,:]-UV[i,:]
            if(p2_order==1): 
                tUV = np.expand_dims(UV[i,:]*t,axis=0)
            else:
                tUV = np.array([UV[i,:]*t, UV[i,:]*t**2])
            X = np.concatenate((atime,tUV),axis=0)
            Xinv = np.linalg.inv( np.matmul(X,X.T))
            alpha = np.matmul( np.matmul(fhat, X.T), Xinv )
            model_fit['a1'][i,:] = alpha[:3]
            if(p2_order==1):
                model_fit['a2'][i,0] = alpha[3:]
            else:
                model_fit['a2'][i,:] = alpha[3:]
        return model_fit


    def opt_P1_given_V(self, a, params):
        [dFF, model_fit] = params
        U = model_fit['U']
        a2 = model_fit['a2']
        M = U.shape[1]
        N = dFF.shape[0]
        a1 = np.reshape(a, (N,3))
        
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        H = np.array([np.ones(t.shape), t, t**2])
        P2 = 1 + (a2@time)
        W = U @ np.linalg.inv(U.T@U) @ U.T  
        
        offset = dFF@H.T - (P2*( W@(dFF/P2) ))@H.T
        eq1 = a1@H@H.T
        eq2 = (P2*( W@( (a1@H)/P2 ) ))@H.T
        obj = ((  offset-eq1+eq2  )**2).sum()
        return obj


    def update_P1_and_V(self, dFF, model_fit):
        params = [dFF, model_fit]
        a0 = np.reshape(model_fit['a1'], (3*N,1))
        res = minimize(opt_P1_given_V, a0, params, method='SLSQP')
        
        
        model_fit['V'] = np.matmul( np.matmul( np.linalg.inv(W), X ), np.linalg.inv(Y)  ).T
        UV = np.matmul(model_fit['U'],model_fit['V'].T)
        Fhat = dFF-P2*UV
        model_fit['a1'] = np.matmul(Fhat, Q)
        return model_fit


    def update_V_at_once(self, dFF, model_fit):
        a1 = model_fit['a1']
        a2 = model_fit['a2']
        U = model_fit['U']
        UV = np.matmul(model_fit['U'],model_fit['V'].T)
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        P1i = np.matmul( a1, atime )
        P2i = 1 + np.matmul( a2, time )
        F_hat = (dFF-P1i)/P2i
        model_fit['V'] = ( np.linalg.inv(U.T@U)@U.T@F_hat ).T
        return model_fit


    def update_V(self, dFF, model_fit):
        a1 = model_fit['a1']
        a2 = model_fit['a2']
        U = model_fit['U']
        UV = np.matmul(model_fit['U'],model_fit['V'].T)
        t = np.arange(0,dFF.shape[1],1)/dFF.shape[1]
        time = np.array([t, t**2])
        atime = np.array([np.ones(t.shape), t, t**2])
        P1i = np.matmul( a1, atime )
        P2i = 1 + np.matmul( a2, time )

        V_hat = np.zeros(model_fit['V'].T.shape)
        for tp in range(len(t)):
            U_hat = np.zeros(U.shape)
            for i in range(U.shape[0]):
                U_hat[i,:] = U[i,:]*P2i[i,tp]
            f_hat = data_dict['dFF'][:,tp]-P1i[:,tp]
            V_hat[:,tp] = np.linalg.inv(U_hat.T@U_hat)@U_hat.T@f_hat
        model_fit['V'] = V_hat.T
        return model_fit

