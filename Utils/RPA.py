'''
#####################################################################################################################
Date        : 1st, Jun., 2023
---------------------------------------------------------------------------------------------------------------------
Descriptions:  

This code is a modification of the RPA source code, with examples adapted for the KU dataset. The code implements two functions, 
RCT and ROT, with the ROT function requiring a semi-supervised condition. The source code can be found in the following link:

https://github.com/plcrodrigues/RPA/tree/cfcddb3d31b482941a23353dfbe46dffb118d02d

#######################################################################################################################
'''


from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.estimation import Covariances, Shrinkage


import pymanopt
from pymanopt.manifolds import Rotations
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.tools.autodiff import (AutogradBackend, TheanoBackend)


from scipy.linalg import eigh
from functools import partial
from load_data import FilterBank
import numpy as np


def gen_symm(n):
    A = np.random.randn(n,n)
    return A + A.T

def gen_spd(n):
    A   = gen_symm(n)
    w,v = eigh(A)
    w   = np.diag(np.random.rand(len(w)))
    return np.dot(v, np.dot(w, v.T))

def gen_anti(n):
    A = np.random.randn(n,n)
    return A - A.T

def gen_orth(n):

    A   = gen_symm(n)
    _,Q = eigh(A)

    return Q

def cost_function_pair_euc(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return np.linalg.norm(t1 - t2)**2

def cost_function_pair_rie(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return distance_riemann(t1, t2)**2

def cost_function_full(Q, M, Mtilde, weights=None, dist=None):

    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)
        
    if dist is None:
        dist = 'euc'
        
    cost_function_pair        = {}
    cost_function_pair['euc'] = cost_function_pair_euc
    cost_function_pair['rie'] = cost_function_pair_rie    
        
    c = []
    for Mi, Mitilde in zip(M, Mtilde):
        ci = cost_function_pair[dist](Mi, Mitilde, Q)
        c.append(ci)
    c = np.array(c)
    
    return np.dot(c, weights)

def egrad_function_pair_rie(M, Mtilde, Q):

    Mtilde_invsqrt = invsqrtm(Mtilde)
    M_sqrt         = sqrtm(M)
    term_aux       = np.dot(Q, np.dot(M, Q.T))
    term_aux       = np.dot(Mtilde_invsqrt, np.dot(term_aux, Mtilde_invsqrt))

    return 4 * np.dot(np.dot(Mtilde_invsqrt, logm(term_aux)), np.dot(M_sqrt, Q))

def egrad_function_full_rie(Q, M, Mtilde, weights=None):

    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)

    g = []
    for Mi, Mitilde, wi in zip(M, Mtilde, weights):
        gi = egrad_function_pair_rie(Mi, Mitilde, Q)
        g.append(gi * wi)
    g = np.sum(g, axis=0)        
    
    return g

def get_rotation_matrix(M, Mtilde, weights=None, dist=None):
    
    if dist is None:
        dist = 'euc'
    
    n = M[0].shape[0]
        
    manifold = Rotations(n)
    
    if dist == 'euc':
        cost    = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    elif dist == 'rie':
        cost    = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        egrad   = partial(egrad_function_full_rie, M=M, Mtilde=Mtilde, weights=weights) 
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
        
    solver = SteepestDescent(mingradnorm=1e-3)   
    
    Q_opt  = solver.solve(problem)    
    
    return Q_opt

def parallel_transport_covariance_matrix(C, R):
    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):
    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)

def transform_org2rct(source_org, target_org_train, target_org_test):

    source_rct = {}
    T_source   = np.stack([mean_riemann(source_org)] * len(source_org))
    source_rct = parallel_transport_covariances(source_org, T_source)

    M_target   = mean_riemann(target_org_train)

    target_rct_train = {}
    T_target_train   = np.stack([M_target]*len(target_org_train))
    target_rct_train = parallel_transport_covariances(target_org_train, T_target_train)

    target_rct_test = {}
    T_target_test   = np.stack([M_target]*len(target_org_test))
    target_rct_test = parallel_transport_covariances(target_org_test, T_target_test)

    return source_rct, target_rct_train, target_rct_test

def transform_rct2str(source, target_train, target_test, pcoeff=False):

    covs_source       = source
    covs_target_train = target_train
    covs_target_test  = target_test

    source_pow = source

    n           = covs_source.shape[1]
    disp_source = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_source]) / len(covs_source)
    disp_target = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_target_train]) / len(covs_target_train)
    p           = np.sqrt(disp_target / disp_source)

    target_pow_train = np.stack([powm(covi, 1.0/p) for covi in covs_target_train])
    target_pow_test  = np.stack([powm(covi, 1.0/p) for covi in covs_target_test])

    if pcoeff:
        return source_pow , target_pow_train, target_pow_test, p
    else:
        return source_pow , target_pow_train, target_pow_test

def transform_rct2rot(source, source_label, target_train, target_train_label, target_test, target_test_label, weights=None, distance='euc'):

    source_rot = source

    M_source = []
    for i in np.unique(source_label):
        M_source_i = mean_riemann(source[source_label == i])
        M_source.append(M_source_i)

    M_target_train = []
    for j in np.unique(target_train_label):
        M_target_train_j = mean_riemann(target_train[target_train_label == j])
        M_target_train.append(M_target_train_j)

    R = get_rotation_matrix(M=M_source, Mtilde=M_target_train, weights=weights, dist=distance)

    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_train])
    target_rot_train  = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_test])
    target_rot_test  = covs_target_test

    return source_rot, target_rot_train, target_rot_test

def transform_str2rot(source, target_train, target_test):
    return transform_rct2rot(source, target_train, target_test)


class Problem(object):
    def __init__(self, manifold, cost, egrad=None, ehess=None, grad=None,
                 hess=None, arg=None, precon=None, verbosity=2):
        self.manifold = manifold
        self._cost    = None
        self._original_cost = cost
        self._egrad   = egrad
        self._ehess   = ehess
        self._grad    = grad
        self._hess    = hess
        self._arg     = arg
        self._backend = None

        if precon is None:
            def precon(x, d):
                return d
        self.precon   = precon

        self.verbosity= verbosity

        self._backends= list(
            filter(lambda b: b.is_available(), [
                TheanoBackend(),
                AutogradBackend(),
                ]))
        self._backend  = None

    @property
    def backend(self):
        if self._backend is None:
            for backend in self._backends:
                if backend.is_compatible(self._original_cost, self._arg):
                    self._backend = backend
                    break
            else:
                backend_names = [str(backend) for backend in self._backends]
                if self.verbosity >= 1:
                    print(backend_names)
                raise ValueError(
                    "Cannot determine autodiff backend from cost function of "
                    "type `{:s}`. Available backends are: {:s}".format(
                        self._original_cost.__class__.__name__,
                        ", ".join(backend_names)))
        return self._backend

    @property
    def cost(self):
        if (self._cost is None and callable(self._original_cost) and
                not AutogradBackend().is_available()):
            self._cost = self._original_cost

        elif self._cost is None:
            if self.verbosity >= 1:
                print("Compiling cost function...")
            self._cost = self.backend.compile_function(self._original_cost,
                                                       self._arg)

        return self._cost

    @property
    def egrad(self):
        if self._egrad is None:
            if self.verbosity >= 1:
                print("Computing gradient of cost function...")
            egrad = self.backend.compute_gradient(self._original_cost,
                                                  self._arg)
            self._egrad = egrad
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            egrad = self.egrad

            def grad(x):
                return self.manifold.egrad2rgrad(x, egrad(x))
            self._grad = grad
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            if self.verbosity >= 1:
                print("Computing Hessian of cost function...")
            ehess = self.backend.compute_hessian(self._original_cost,
                                                 self._arg)
            self._ehess = ehess
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            ehess = self.ehess

            def hess(x, a):
                return self.manifold.ehess2rhess(
                    x, self.egrad(x), ehess(x, a), a)
            self._hess = hess
        return self._hess


if __name__ == '__main__':

    channel_index = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    x_train = np.load('Dataset/sess1_sub1_x.npy')[:   , channel_index, :]
    x_val   = np.load('Dataset/sess2_sub1_x.npy')[:100, channel_index, :]
    x_test  = np.load('Dataset/sess2_sub1_x.npy')[100:, channel_index, :]
    
    fbank   = FilterBank(fs=1000)
    _       = fbank.get_filter_coeff()

    x_train = fbank.filter_data(x_train, window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    x_val   = fbank.filter_data(x_val,   window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    x_test  = fbank.filter_data(x_test,  window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
    
    def temporal_sig2cov(x):

        temp_stack  = []
        for i in range(x.shape[0]):
            cov_stack = []
            for j in range(x.shape[1]):
              cov_stack.append(Shrinkage(1e-2).transform(Covariances().transform(x[i,j])))
            temp_stack.append(np.stack(cov_stack, axis = 0))

        return np.stack(temp_stack, axis = 0)
    

    def stack(data, interval):

        data_record = []
        for [a, b] in interval:
            data_record.append(np.expand_dims(data[:, :, :, a:b], axis = 1))

        return temporal_sig2cov(np.concatenate(data_record, axis = 1))

    interval = [[0, 2500]]  
    
    source_org          = stack(x_train, interval).reshape(200, 9, 20, 20)
    target_org_train    = stack(x_val, interval).reshape(100, 9, 20, 20)
    target_org_test     = stack(x_test, interval).reshape(100, 9, 20, 20)

    source_label        = np.load('Dataset/sess1_sub1_y.npy')
    target_train_label  = np.load('Dataset/sess2_sub1_y.npy')[:100]                                             
    target_test_label   = np.load('Dataset/sess2_sub1_y.npy')[100:]

    source_rct=source_pow=source_rot=np.zeros(source_org.shape)
    target_rct_train=target_pow_train=target_rot_train=np.zeros(target_org_train.shape)
    target_rct_test=target_pow_test=target_rot_test=np.zeros(target_org_test.shape) 

    for channel in range(source_org.shape[1]):     
          source_rct[:,channel,:,:], target_rct_train[:,channel,:,:], target_rct_test[:,channel,:,:] = transform_org2rct(source_org[:,channel,:,:], target_org_train[:,channel,:,:], target_org_test[:,channel,:,:])
          source_pow[:,channel,:,:], target_pow_train[:,channel,:,:], target_pow_test[:,channel,:,:] = transform_rct2str(source_rct[:,channel,:,:], target_rct_train[:,channel,:,:], target_rct_test[:,channel,:,:], pcoeff=False)
          source_rot[:,channel,:,:], target_rot_train[:,channel,:,:], target_rot_test[:,channel,:,:] = transform_rct2rot(source_pow[:,channel,:,:], source_label, 
                                                                                                                        target_pow_train[:,channel,:,:], target_train_label, 
                                                                                                                        target_pow_test[:,channel,:,:], target_test_label, 
                                                                                                                        weights=None, distance='rie')
        
