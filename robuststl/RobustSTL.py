from cvxopt import matrix

import numpy as np 
import math
import os

def denoise_step(sample, H=3, dn1=1., dn2=1.):
    def get_denoise_value(idx):
        start_idx, end_idx = get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(lambda j: bilateral_filter(j, idx, sample[j], sample[idx], dn1, dn2), idxs)))
        return np.sum(weight_sample * weights)/np.sum(weights)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)))
    return denoise_sample

def trend_extraction(sample, season_len, reg1=10., reg2=0.5):
    sample_len = len(sample)
    season_diff = sample[season_len:] - sample[:-season_len]
    assert len(season_diff) == (sample_len - season_len)
    q = np.concatenate([season_diff, np.zeros([sample_len*2-3])])
    q = np.reshape(q, [len(q),1])
    q = matrix(q)

    M = get_toeplitz([sample_len-season_len, sample_len-1], np.ones([season_len]))
    D = get_toeplitz([sample_len-2, sample_len-1], np.array([1,-1]))
    P = np.concatenate([M, reg1*np.eye(sample_len-1), reg2*D], axis=0)
    P = matrix(P)
    
    delta_trends = l1(P,q)
    relative_trends = get_relative_trends(delta_trends)

    return sample-relative_trends, relative_trends

def seasonality_extraction(sample, season_len=10, K=2, H=5, ds1=50., ds2=1.):
    sample_len = len(sample)
    idx_list = np.arange(sample_len)

    def get_season_value(idx):
        idxs = get_season_idx(sample_len, idx, season_len, K, H)
        if idxs.size == 0:
            return sample[idx]

        weight_sample = sample[idxs]
        #t_idxs = [idx - (int((idx -j)/season_len)+1)*season_len for j in idxs]
        #weights = np.array(list(map(lambda j, t: bilateral_filter(j, t, sample[j], sample[t], ds1, ds2), idxs, t_idxs)))
        weights = np.array(list(map(lambda j: bilateral_filter(j, idx, sample[j], sample[idx], ds1, ds2), idxs)))
        season_value = np.sum(weight_sample * weights)/np.sum(weights)
        return season_value

    seasons_tilda = np.array(list(map(get_season_value, idx_list)))
    return seasons_tilda

def adjustment(sample, relative_trends, seasons_tilda, season_len):
    num_season = int(len(sample)/season_len)
    trend_init = np.mean(seasons_tilda[:season_len*num_season])
    
    trends_hat = relative_trends + trend_init
    seasons_hat = seasons_tilda - trend_init
    remainders_hat = sample - trends_hat - seasons_hat
    return [trends_hat, seasons_hat, remainders_hat]

def check_converge_criteria(prev_remainders, remainders):
    diff = np.sqrt(np.mean(np.square(remainders-prev_remainders)))
    if diff < 1e-10:
        return True
    else:
        return False

def _RobustSTL(input, season_len, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.):
    '''
    args:
    - reg1: first order regularization parameter for trend extraction
    - reg2: second order regularization parameter for trend extraction
    - K: number of past season samples in seasonaility extraction
    - H: number of neighborhood in seasonality extraction
    - dn1, dn2 : hyperparameter of bilateral filter in denoising step.
    - ds1, ds2 : hypterparameter of bilarteral filter in seasonality extraction step.
    '''
    sample = input
    trial = 1
    patient=0
    while True:
    #for hey in range(10):
        #step1: remove noise in input via bilateral filtering
        denoise_sample =\
                denoise_step(sample, H, dn1, dn2)

        #step2: trend extraction via LAD loss regression 
        detrend_sample, relative_trends =\
                trend_extraction(denoise_sample, season_len, reg1, reg2)

        #step3: seasonality extraction via non-local seasonal filtering
        seasons_tilda =\
                seasonality_extraction(detrend_sample, season_len, K, H, ds1, ds2)

        #step4: adjustment of trend and season
        trends_hat, seasons_hat, remainders_hat =\
                adjustment(sample, relative_trends, seasons_tilda, season_len)

        #step5: repreat step1 - step4 until remainders are converged
        if trial != 1:            
            converge = check_converge_criteria(previous_remainders, remainders_hat)
            if converge:
                return [input, trends_hat, seasons_hat, remainders_hat]
        
        trial+=1
        print("[!] ", trial, "iteration will strat")
        previous_remainders = remainders_hat[:]
        sample = trends_hat + seasons_hat + remainders_hat
    return [input, trends_hat, seasons_hat, remainders_hat]

def RobustSTL(input, season_len, reg1=10.0, reg2= 0.5, K=2, H=5, dn1=1., dn2=1., ds1=50., ds2=1.):
        '''
    args:
    - input: time series as a numpy array
    - season_len: length of seasonal period
    - reg1: first order regularization parameter for trend extraction
    - reg2: second order regularization parameter for trend extraction
    - K: number of past season samples in seasonaility extraction
    - H: number of neighborhood in seasonality extraction
    - dn1, dn2 : hyperparameter of bilateral filter in denoising step.
    - ds1, ds2 : hypterparameter of bilarteral filter in seasonality extraction step.
    '''
    if np.ndim(input) < 2:
        return _RobustSTL(input, season_len, reg1, reg2, K, H, dn1, dn2, ds1, ds2)
    
    elif np.ndim(input)==2 and np.shape(input)[1] ==1:
        return _RobustSTL(input[:,0], season_len, reg1, reg2, K, H, dn1, dn2, ds1, ds2)
    
    elif np.ndim(input)==2 or np.ndim(input)==3:
        if np.ndim(input)==3 and np.shape(input)[2] > 1:
            print("[!] Valid input series shape: [# of Series, # of Time Steps] or [# of series, # of Time Steps, 1]")
            raise
        elif np.ndim(input)==3:
            input = input[:,:,0]
        num_series = np.shape(input)[0]
            
        input_list = [input[i,:] for i in range(num_series)]
        
        from pathos.multiprocessing import ProcessingPool as Pool
        p = Pool(num_series)
        def run_RobustSTL(_input):
            return _RobustSTL(_input, season_len, reg1, reg2, K, H, dn1, dn2, ds1, ds2)
        result = p.map(run_RobustSTL, input_list)

        return result
    else:
        print("[!] input series error")
        raise

###### UTILS ######
from scipy.linalg import toeplitz
import numpy as np
import math

def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    idx1 = -1.0 * (math.fabs(j-t) **2.0)/(2.0*delta1**2)
    idx2 = -1.0 * (math.fabs(y_j-y_t) **2.0)/(2.0*delta2**2)
    weight = (math.exp(idx1)*math.exp(idx2))
    #print('args: ', j, t, y_j, y_t, weight,math.exp(idx1),math.exp(idx2) )
    return weight

def get_neighbor_idx(total_len, target_idx, H=3):
    '''
    Let i = target_idx.
    Then, return i-H, ..., i, ..., i+H, (i+H+1)
    '''
    return [np.max([0, target_idx-H]), np.min([total_len, target_idx+H+1])]

def get_neighbor_range(total_len, target_idx, H=3):
    start_idx, end_idx = get_neighbor_idx(total_len, target_idx, H)
    return np.arange(start_idx, end_idx)

def get_season_idx(total_len, target_idx, T=10, K=2, H=5):
    num_season = np.min([K, int(target_idx/T)])
    if target_idx < T:
        key_idxs = target_idx + np.arange(0, num_season+1)*(-1*T)
    else:        
        key_idxs = target_idx + np.arange(1, num_season+1)*(-1*T)
    
    idxs = list(map(lambda idx: get_neighbor_range(total_len, idx, H), key_idxs))
    season_idxs = []
    for item in idxs:
        season_idxs += list(item)
    season_idxs = np.array(season_idxs)
    return season_idxs

def get_relative_trends(delta_trends):
    init_value = np.array([0])
    idxs = np.arange(len(delta_trends))
    relative_trends = np.array(list(map(lambda idx: np.sum(delta_trends[:idx]), idxs)))
    relative_trends = np.concatenate([init_value, relative_trends])
    return relative_trends

def get_toeplitz(shape, entry):
    h, w = shape
    num_entry = len(entry)
    assert np.ndim(entry) < 2
    if num_entry < 1:
        return np.zeros(shape)
    row = np.concatenate([entry[:1], np.zeros(h-1)])
    col = np.concatenate([np.array(entry), np.zeros(w-num_entry)])
    return toeplitz(row, col)

###### L1 ######
from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spdiag, mul, div, sparse 
from cvxopt import spmatrix, sqrt, base

def l1(P, q):
    m, n = P.size
    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        if trans == 'N':
            u = P*x[:n]
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]
        else:
            y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta*y[:n]
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]

    def Fkkt(W): 
        d1, d2 = W['d'][:m], W['d'][m:]
        D = 4*(d1**2 + d2**2)**-1
        A = P.T * spdiag(D) * P
        lapack.potrf(A)

        def f(x, y, z):
            x[:n] += P.T * ( mul( div(d2**2 - d1**2, d1**2 + d2**2), x[n:]) 
                + mul( .5*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            u = P*x[:n]
            x[n:] =  div( x[n:] - div(z[:m], d1**2) - div(z[m:], d2**2) + 
                mul(d1**-2 - d2**-2, u), d1**-2 + d2**-2 )

            z[:m] = div(u-x[n:]-z[:m], d1)
            z[m:] = div(-u-x[n:]-z[m:], d2)
        return f

    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q 

    x0 = matrix( [uls[:n],  1.1*abs(rls)] ) 
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1) 

    if max(abs(rls)) > 1e-10:  
        w = .9/max(abs(rls)) * rls
    else: 
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,  
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n]
