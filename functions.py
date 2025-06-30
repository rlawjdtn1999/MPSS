import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy.stats import qmc
from polybasis import compute_M_matrix

d_init    = np.array([5.0, 5.0])

# QMC normal sample generator
def generate_qmc_normal_samples(mean, cov, n_samples):
    sampler = qmc.Sobol(d=len(mean), scramble=True)
    u = sampler.random(n_samples)
    u = np.clip(u, 1e-10, 1-1e-10)  # For numerical stability
    normal_samples = norm.ppf(u)  # Transform to standard normal
    L = np.linalg.cholesky(cov)
    return mean + normal_samples @ L.T

# scoreGauss
def scoreGauss(X, mu, cov):
    return np.array([ np.linalg.solve(cov, x - mu) for x in X ])


# fit_surrogate
def fit_surrogate( d, basis_terms, W, y_func, Z0_samples): 
    Z_samples = Z0_samples + d_init - d
    M = compute_M_matrix(Z_samples, basis_terms)
    Psi = M @ W.T
    y_samples = y_func(Z_samples)
    c, *_ = np.linalg.lstsq(Psi, y_samples, rcond=None)
    mean_ref = c[0]
    var_ref = np.sum(c[1:]**2)
    return c, mean_ref, var_ref ,Psi

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