import cvxpy as cp
import numpy as np
import tqdm
import scipy
import math
from scipy.special import xlogy

# CVXOPT for optimal distribution
def calculate_optimal_distribution(n, m, P, sum_x=1):
    '''
    copied from https://www.cvxpy.org/examples/applications/Channel_capacity_BV4.57.html
    '''

    # n is the number of different input values
    # m is the number of different output values
    if n*m == 0:
        print('The range of both input and output values must be greater than zero')
        return 'failed', np.nan, np.nan

    # x is probability distribution of the input signal X(t)
    x = cp.Variable(shape=n)

    # y is the probability distribution of the output signal Y(t)
    # P is the channel transition matrix
    y = P@x

    # I is the mutual information between x and y
    c = np.sum(np.array((xlogy(P, P) / math.log(2))), axis=0)
    I = c@x + cp.sum(cp.entr(y) / math.log(2))

    # Channel capacity maximised by maximising the mutual information
    obj = cp.Maximize(I)
    constraints = [cp.sum(x) == sum_x,x >= 0]

    # Form and solve problem
    prob = cp.Problem(obj,constraints)
    prob.solve()
    if prob.status=='optimal':
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan

# Quantizer functions for exhaustive search
def calculate_I(px, Q, M, Phi, H_thres):
    Ax = np.zeros((M, Q))
    for m in range(Q):
        for n in range(M):
            Ax[n, m] = Phi[m].cdf(H_thres[n+1]) - Phi[m].cdf(H_thres[n])
            
    py = np.matmul(Ax, px)
    Hy = -np.sum(np.array((xlogy(py, py) / math.log(2))), axis=0)
    
    c = np.sum(np.array((xlogy(Ax, Ax) / math.log(2))), axis=0)
    Hyx = -np.sum(px*c)
    
    I = Hy - Hyx
    
    return I

def exhaustive_search_quantization_3_level(Px, N, Q, M, S, Phi):
    search_thresh = S[1:-1]
    data = {}
    current_best = -np.inf
    current_best_thres = None
    for idx_h1, h1 in tqdm.tqdm(enumerate(search_thresh[:-1])):
        for idx_h2, h2 in enumerate(search_thresh[idx_h1+1:]):
            data[(h1, h2)] = calculate_I(Px, Q, M, Phi, [S[0], h1, h2, S[-1]])
            if data[(h1, h2)] > current_best:
                current_best = data[(h1, h2)]
                current_best_thres = (h1, h2)
                
    return current_best, current_best_thres, data

def exhaustive_search_quantization_4_level(Px, N, Q, M, S, Phi):
    search_thresh = S[1:-1]
    data = {}
    current_best = -np.inf
    current_best_thres = None
    for idx_h1, h1 in tqdm.tqdm(enumerate(search_thresh[:-2])):
        for idx_h2, h2 in enumerate(search_thresh[idx_h1+1:-1]):
            for idx_h3, h3 in enumerate(search_thresh[idx_h2+1:]):
                data[(h1, h2, h3)] = calculate_I(Px, Q, M, Phi, [S[0], h1, h2, h3, S[-1]])
                if data[(h1, h2, h3)] > current_best:
                    current_best = data[(h1, h2, h3)]
                    current_best_thres = (h1, h2, h3)
                
    return current_best, current_best_thres, data

def exhaustive_search_quantization_5_level(Px, N, Q, M, S, Phi):
    search_thresh = S[1:-1]
    data = {}
    current_best = -np.inf
    current_best_thres = None
    for idx_h1, h1 in tqdm.tqdm(enumerate(search_thresh[:-3])):
        for idx_h2, h2 in enumerate(search_thresh[idx_h1+1:-2]):
            for idx_h3, h3 in enumerate(search_thresh[idx_h2+1:-1]):
                for idx_h4, h4 in enumerate(search_thresh[idx_h3+1:]):
                    data[(h1, h2, h3, h4)] = calculate_I(Px, Q, M, Phi, [S[0], h1, h2, h3, h4, S[-1]])
                    if data[(h1, h2, h3, h4)] > current_best:
                        current_best = data[(h1, h2, h3, h4)]
                        current_best_thres = (h1, h2, h3, h4)
                
    return current_best, current_best_thres, data

# Quantizer functions for DP
def calculate_transition_matrix(Px, N, Q, S, Phi):
    Ayx = np.zeros((N, Q))

    for j in range(Q):
        for i in range(N):
            Ayx[i, j] = Phi[j].cdf(S[i+1]) - Phi[j].cdf(S[i])

    Axy = np.zeros((Q, N))
    for m in range(Q):
        for n in range(N):
            Axy[m, n] = Px[m]*Ayx[n, m]/np.sum(Px*Ayx[n,:])

    # fix nan values by repeating nearest row
    Axy_cp = Axy.T.copy()

    nan_index = np.arange(N)[np.any(np.isnan(Axy.T), axis=1) == True]
    upper_half = nan_index[nan_index<N/2]
    lower_half = nan_index[nan_index>=N/2]

    if len(upper_half) > 0:
        upper_half_idx = upper_half[-1]
        Axy_cp[:upper_half_idx+1,:] = Axy_cp[upper_half_idx+1,:]

    if len(lower_half) > 0:
        lower_half_idx = lower_half[0]
        Axy_cp[lower_half_idx:,:] = Axy_cp[lower_half_idx-1,:]

    Axy = Axy_cp.T

    Py = np.matmul(Ayx, Px)

    Pxy = Axy*Py
    
    return Ayx, Axy, Py, Pxy

def calculate_cost_w(l, r, Pxy, Py, Q):
    tmp = []
    dem = np.sum(Py[l:r+1])
    for k in range(l, r+1):
        tmp_tmp = []
        for i in range(Q):
            num = np.sum(Pxy[i,l:r+1])
            ent = xlogy(num/dem, num/dem)
            # print(num, dem, ent)
            tmp_tmp.append(ent)
            # print(num, dem, ent)
        tmp.append(Py[k]*sum(tmp_tmp))
    return -np.sum(tmp)

def dp_optimal_quantizer(N, M, Pxy, Py, Q):
    DP = np.zeros((N, M))
    SOL = np.zeros((N, M))

    for n in range(N):
        DP[n, 0] = calculate_cost_w(0, n, Pxy, Py, Q)
        SOL[n, 0] = 0

    for m in range(1, M):
        for n in np.arange(m, N-M+m+1)[::-1]:
            tmp = []
            for t in range(m-1, n):
                tmp.append(DP[t, m-1] + calculate_cost_w(t+1, n, Pxy, Py, Q))
            # SOL[n, m] = np.argmin(tmp)
            SOL[n, m] = np.arange(m-1, n)[np.argmin(tmp)]
            t = int(SOL[n, m])
            DP[n, m] = DP[t, m-1] + calculate_cost_w(t+1, n, Pxy, Py, Q)
            
    H = []
    h_prev = N
    H.append(h_prev)
    for m in np.arange(M)[::-1]:
        h_prev = int(SOL[h_prev-1, m]) + 1
        H.append(h_prev)
    H[-1] -= 1
    H = H[::-1]
    
    return H, DP[N-1, M-1]

