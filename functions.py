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

# update_c
def rebase_surrogate(d, basis_terms, W, c_old, Psi_old, X_samples):
    Z = X_samples - d 
    M = compute_M_matrix(Z, basis_terms)      
    Psi_new = M @ W.T                         

    A = Psi_old.T @ Psi_old                   
    B = Psi_old.T @ (Psi_new @ c_old)         
    c_new = np.linalg.solve(A, B)            

    return c_new, Psi_new

 # fit_surrogate for multiple functions
def fit_all_surrogates(basis_terms, W, y_funcs, Z0_samples):
     # Psi is calculated only ONCE, as it's independent of y_funcs.
    M = compute_M_matrix(Z0_samples, basis_terms)
    Psi = M @ W.T

    c_dict = {}
    for y_func in y_funcs:
        y_samples = y_func(Z0_samples)
        c, *_ = np.linalg.lstsq(Psi, y_samples, rcond=None)
        # Store the coefficient vector with the function name as the key
        c_dict[y_func.__name__] = c

    return c_dict, Psi

# rebase_surrogate for multiple functions
def rebase_all_surrogates(d, basis_terms, W, c_old_dict, Psi_old, X_samples):
    # Calculate new Psi only ONCE.
    Z = X_samples - d
    M = compute_M_matrix(Z, basis_terms)
    Psi_new = M @ W.T

    # Pre-calculate the A matrix for solving
    A = Psi_old.T @ Psi_old

    c_new_dict = {}
    # Iterate through the dictionary of old coefficients
    for name, c_old in c_old_dict.items():
        # Calculate the B matrix using the fake output data
        B = Psi_old.T @ (Psi_new @ c_old)
        c_new = np.linalg.solve(A, B)
        c_new_dict[name] = c_new

    return c_new_dict, Psi_new
    
# design_functions
def y1(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + (X1**2 * X2)/ 20

def y2(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + ((X1 + X2 - 5)**2) / 30 + ((X1 - X2 - 12)**2) / 120

def y3(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return -1 + 80 /(X1**2 +8 *X2 + 5)