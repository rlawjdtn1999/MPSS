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


# Exponential autocorrelation funtion
def compute_correlation_matrix(X, X_ref, theta):
    diff = (X[:, None, :] - X_ref[None, :, :]) / theta
    return np.exp(-np.sum(np.abs(diff), axis=2))

# LOO objective
def loo_cv_objective(theta, X, y):
    R = compute_correlation_matrix(X, X, theta)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros_like(theta)
    # LOOCV 잔차 u_i = (R⁻¹y)_i / (R⁻¹)_{ii}
    b         = y.reshape(-1, 1)
    a         = R_inv.dot(b).ravel()
    diag_invR = np.diag(R_inv)
    u         = a / diag_invR
    # 목적함수 값
    obj = np.sum(u**2)

    # analytic gradient
    grad = np.zeros_like(theta)
    for k in range(theta.shape[0]):
        # ∂R/∂θ_k  (exponential kernel)
        Dk   = np.abs(X[:,k][:,None] - X[:,k][None,:])
        dR   = -Dk * R
        # ∂(R⁻¹) = -R⁻¹·dR·R⁻¹
        dInv = - R_inv.dot(dR).dot(R_inv)

        da    = dInv.dot(b).ravel()      # ∂a
        ddiag = np.diag(dInv)            # ∂diag_invR
        du    = (da * diag_invR - a * ddiag) / (diag_invR**2)

        grad[k] = 2 * np.dot(u, du)

    return obj, grad

# Design functions
def y0(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return (X1 - 4)**3 + (X1 - 3)**8 + (X2 - 5)**4 + 10

def y1(X):
    X1 = X[..., 0]
    X2 = X[..., 1]
    return X1 + X2 - 6.45