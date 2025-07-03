import numpy as np
from scipy.optimize import minimize
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples , fit_surrogate, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
import Parameters   

# 파라미터 설정
N         = Parameters.N  #2
S         = Parameters.S  #1
m         = Parameters.m  #[3 , 4, 2]
m1        = Parameters.m[0]
m2        = Parameters.m[1]
m3        = Parameters.m[2]
mean      = np.array(Parameters.mean)
cov      = np.array(Parameters.cov1)

d_init    = np.array([5.0, 5.0])    # 초기 설계 변수
d_bounds = [(0.0, 10.0), (0.0, 10.0)]  

beta_init = [0.3, 0.3]
err1      = Parameters.err1
err2      = Parameters.err2
err3      = Parameters.err3
err4      = Parameters.err4
err5      = Parameters.err5
err6      = Parameters.err6
err7      = Parameters.err7
err8      = Parameters.err8
err_vals = {
    "err3": err3,
    "err4": err4,
    "err5": err5,
    "err6": err6,
    "err7": err7
}
threshold   = 0

basis_terms1 = generate_monomial_basis(N, S, m1)
basis_terms2 = generate_monomial_basis(N, S, m2)
basis_terms3 = generate_monomial_basis(N, S, m3)

mean_0 = mean - d_init # [0 0]s
W1 = compute_whitening_matrix(N, S, m1, mean_0, cov )
W2 = compute_whitening_matrix(N, S, m2, mean_0, cov )
W3 = compute_whitening_matrix(N, S, m3, mean_0, cov )

def objective(d):
    d1 = d[..., 0]
    d2 = d[..., 1]
    return -d1 + d2

def constraint1(d):
    Z_samples = X_samples - d
    M1 = compute_M_matrix(Z_samples, basis_terms1)
    Psi_new = M1 @ W1.T

    A = Psi_1.T @ Psi_1
    B = Psi_1.T @ (Psi_new @ c_1 )
    c1_new = np.linalg.solve(A,B)
    y_new   = Psi_new @ c1_new  
    P_failure = np.sum(y_new < threshold) / len(y_new)
    return   P_failure - 1.35e-3

def constraint2(d):
    Z_samples = X_samples - d
    M2 = compute_M_matrix(Z_samples, basis_terms2)
    Psi_new = M2 @ W2.T

    A = Psi_2.T @ Psi_2
    B = Psi_2.T @ (Psi_new @ c_2 )
    c2_new = np.linalg.solve(A,B)
    y_new   =  Psi_new @ c2_new
    P_failure = np.sum(y_new < threshold) / len(y_new)
    return  P_failure - 1.35e-3

def constraint3(d):
    Z_samples = X_samples - d
    M3 = compute_M_matrix(Z_samples, basis_terms3)
    Psi_new = M3 @ W3.T

    A = Psi_3.T @ Psi_3
    B = Psi_3.T @ (Psi_new @ c_3 )
    c3_new = np.linalg.solve(A,B)
    y_new   =  Psi_new @ c3_new
    P_failure = np.sum(y_new < threshold) / len(y_new)
    return  P_failure - 1.35e-3


################     MPSS   q = 1   ################################
X_samples = generate_qmc_normal_samples(mean, cov, 128)
d0 = d_init
d_old = np.array([10.0, 10.0])

Z0_samples = X_samples - d0
c_1, mean_y1_ref, var_y1_ref, Psi_1 = fit_surrogate(d0, basis_terms1, W1, y1, Z0_samples)
c_2, mean_y2_ref, var_y2_ref, Psi_2 = fit_surrogate(d0, basis_terms2, W2, y2, Z0_samples)
c_3, mean_y3_ref, var_y3_ref, Psi_3 = fit_surrogate(d0, basis_terms3, W3, y3, Z0_samples)

subregion_bounds = get_subregion_bounds(d0, beta_init, d_bounds)

result1 = minimize(
    fun=objective,
    x0=d0,
    bounds=subregion_bounds,
    constraints=[
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
    ],
    method='SLSQP',
    options={'disp': True, 'ftol': 1e-3}
)

d_old = d0
d0 = result1.x
objective_value_old = objective(d_old)
objective_value_new = objective(d0)

################     MPSS   q >= 2  ################################
while (
    np.linalg.norm(d0 - d_old) >= err1 and 
    np.abs(objective_value_new - objective_value_old) >= err2
):

    Z0_samples = X_samples - d0
    c_1, mean_y1_ref, var_y1_ref, Psi_1 = fit_surrogate(d0, basis_terms1, W1, y1, Z0_samples)
    c_2, mean_y2_ref, var_y2_ref, Psi_2 = fit_surrogate(d0, basis_terms2, W2, y2, Z0_samples)
    c_3, mean_y3_ref, var_y3_ref, Psi_3 = fit_surrogate(d0, basis_terms3, W3, y3, Z0_samples)

    c_new = np.array([constraint1(d0), constraint2(d0), constraint3(d0)])
    c_old = np.array([constraint1(d_old), constraint2(d_old), constraint3(d_old)])
    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)

    result1 = minimize(
        fun=objective,
        x0=d0,
        bounds= subregion_bounds,
        constraints=[
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3},
        ],
        method='SLSQP',
        options={
            'disp': True,
            'ftol': 1e-3,      # 함수 수렴 허용오차
        }
    )

    d_old = d0
    d0 = result1.x
    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)

print(d0)