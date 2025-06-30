import numpy as np
from scipy.optimize import minimize
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples , fit_surrogate, y1, y2, y3
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

basis_terms1 = generate_monomial_basis(N, S, m1)
basis_terms2 = generate_monomial_basis(N, S, m2)
basis_terms2 = generate_monomial_basis(N, S, m3)

mean_0 = mean - d_init # [0 0]s
W1 = compute_whitening_matrix(N, S, m1, mean_0, cov )
W2 = compute_whitening_matrix(N, S, m2, mean_0, cov )

X_samples = generate_qmc_normal_samples(mean, cov, 128)
d0 = d_init

##          MPSS에서는 여기서 부터 루프문

Z0_samples = X_samples - d0
c_1, mean_y1_ref, var_y1_ref, Psi_1 = fit_surrogate(d0, basis_terms1, W1, y1, Z0_samples)
c_2, mean_y2_ref, var_y2_ref, Psi_2 = fit_surrogate(d0, basis_terms2, W2, y2, Z0_samples)

def objective(d):
    d1 = d[..., 0]
    d2 = d[..., 1]
    return -d1 + d2

def constraint(d):
    Z_samples = X_samples - d
    M1 = compute_M_matrix(Z_samples, basis_terms1)
    Psi_new = M1 @ W1.T

    A = Psi_1.T @ Psi_1
    B = Psi_1.T @ (Psi_new @ c_1 )
    c1_new = np.linalg.solve(A,B)
    y_new   = c1_new @ Psi_new 

    return  y_new

result = minimize(
    fun=objective,
    x0=np.array([5.0, 5.0]),
    bounds=[(0.0, 10.0), (0.0, 10.0)],
    constraints=[{
        'type': 'ineq',
        'fun': constraint,
    }],
    method='SLSQP',
    options={
        'disp': True,
        'ftol': 1e-3,      # 함수 수렴 허용오차
    }
)

d0 = result.x
print("최적의 d:", result.x)
print("최적의 목적함수 값:", result.fun)


