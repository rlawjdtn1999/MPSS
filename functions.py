import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy.stats import qmc
from polybasis import compute_M_matrix

# QMC normal sample generator
def generate_qmc_normal_samples(mean, cov, n_samples):
    sampler = qmc.Sobol(d=len(mean), scramble=True)
    u = sampler.random(n_samples)
    u = np.clip(u, 1e-10, 1-1e-10)  # For numerical stability
    normal_samples = norm.ppf(u)  # Transform to standard normal
    L = np.linalg.cholesky(cov)
    return mean + normal_samples @ L.T

# fit_surrogate
def fit_surrogate( basis_terms, W, y_func, Z0_samples): 
    M = compute_M_matrix(Z0_samples, basis_terms)
    Psi = M @ W.T
    y_samples = y_func(Z0_samples)
    c, *_ = np.linalg.lstsq(Psi, y_samples, rcond=None)
    # mean_ref = c[0]
    # var_ref = np.sum(c[1:]**2)
    return c, Psi

#update c
def rebase_surrogate(d, basis_terms, W, c_old, Psi_old, X_samples):
    Z = X_samples - d 
    M = compute_M_matrix(Z, basis_terms)      
    Psi_new = M @ W.T                         

    A = Psi_old.T @ Psi_old                   
    B = Psi_old.T @ (Psi_new @ c_old)         
    c_new = np.linalg.solve(A, B)            

    return c_new, Psi_new

# Design functions
def y1(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + X1**2 * X2 / 20

def y2(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + (X1 + X2 - 5)**2/30 + (X1 - X2 - 12)**2/ 120 

def y3(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + 80 /(X1**2 +8 *X2 + 5)