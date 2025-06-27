import numpy as np
from itertools import combinations
from polybasis import compute_M_matrix


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
 # Step 1: Sample generation & model output
X_samples = generate_qmc_normal_samples(mean, cov, n_samples)
y_lf = hf_model(X_samples)

# Step 2: Basis & whitening
M_matrix = compute_M_matrix(X_samples, basis_terms)
W = np.load('W.npy')
A_matrix = M_matrix @ W.T

# Step 3: Optimize theta
pairwise_dists = pdist(X_samples)              
median_dist = np.median(pairwise_dists)       
theta0 = np.full(N, median_dist)                
bounds = [(1e-5, 100.0)] * N   
res = minimize(loo_cv_objective, theta0, args=(X_samples, y_lf), jac=True, bounds=bounds, method='L-BFGS-B')
theta_star = res.x
R_opt = compute_correlation_matrix(X_samples, X_samples, theta_star)
R_inv = np.linalg.inv(R_opt)

# Step 4: GP regression
b_vector = y_lf.reshape(-1, 1)
ARinv = A_matrix.T @ R_inv
c_hat = np.linalg.solve(ARinv @ A_matrix, ARinv @ b_vector)
residual = b_vector - A_matrix @ c_hat
sigma2 = (residual.T @ R_inv @ residual) / n_samples

# Step 5: Prediction function 생성
def predict(X):
    Mx = compute_M_matrix(X, basis_terms)
    Psi = Mx @ W.T
    r_vec = compute_correlation_matrix(X, X_samples, theta_star)
    mu = Psi @ c_hat + r_vec @ (R_inv @ (b_vector - A_matrix @ c_hat))
    t1 = 1 - np.sum(r_vec @ R_inv * r_vec, axis=1, keepdims=True)
    delta = (Psi - (A_matrix.T @ R_inv @ r_vec.T).T)  
    M_inv = np.linalg.inv(A_matrix.T @ R_inv @ A_matrix)
    t2 = np.einsum('ij,jk,ik->i', delta, M_inv, delta)
    var = sigma2 * (t1 + t2.reshape(-1, 1))
    var = np.maximum(var, 0)
    return mu.ravel(), np.sqrt(var.ravel())

# Step 6: Risk region & CVaR 추정 (Algorithm 2 기반 + 무작위 샘플링)
X_test = generate_qmc_normal_samples(mean, cov, n_test)
mu_preds, std_preds = predict(X_test)
epsilons = z_alpha * std_preds


mu_preds, std_preds = predict(X_test)
