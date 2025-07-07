import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
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

beta = [0.3, 0.3]
beta_init = [0.3, 0.3]

err1      = Parameters.err1 #1e-3
err2      = Parameters.err2 #1e-3
err3      = Parameters.err3 #0.01
err4      = Parameters.err4 #0.07
err5      = Parameters.err5 #0.01
err6      = Parameters.err6 #0.5
err7      = Parameters.err7 #0.05
err8      = Parameters.err8 #0.3
err_vals = {
    "err3": err3,
    "err4": err4,
    "err5": err5,
    "err6": err6,
    "err7": err7
}
threshold   = 0
mean_0 = mean - d_init # [0 0]

c1_new_saved = None
c2_new_saved = None
c3_new_saved = None

Psi1_new_saved = None
Psi2_new_saved = None
Psi3_new_saved = None

basis_terms1 = generate_monomial_basis(N, S, m1)
basis_terms2 = generate_monomial_basis(N, S, m2)
basis_terms3 = generate_monomial_basis(N, S, m3)

W1 = compute_whitening_matrix(N, S, m1, mean_0, cov )
W2 = compute_whitening_matrix(N, S, m2, mean_0, cov )
W3 = compute_whitening_matrix(N, S, m3, mean_0, cov )

def objective(d):
    d1 = d[..., 0]
    d2 = d[..., 1]
    return -d1 + d2

def constraint1(d):
    global c1_new_saved
    global Psi1_new_saved

    Z_samples = X_samples - d
    M1 = compute_M_matrix(Z_samples, basis_terms1)
    Psi1_new = M1 @ W1.T

    A = Psi1.T @ Psi1
    B = Psi1.T @ (Psi1_new @ c1 )
    c1_new = np.linalg.solve(A,B)
    y_new   = Psi1_new @ c1_new  
    P_failure = np.sum(y_new < threshold) / len(y_new)

    c1_new_saved = c1_new
    Psi1_new_saved = Psi1_new

    return   1.35e-3 - P_failure 

def constraint2(d):
    global c2_new_saved
    global Psi2_new_saved

    Z_samples = X_samples - d
    M2 = compute_M_matrix(Z_samples, basis_terms2)
    Psi2_new = M2 @ W2.T

    A = Psi2.T @ Psi2
    B = Psi2.T @ (Psi2_new @ c2 )
    c2_new = np.linalg.solve(A,B)
    y_new   =  Psi2_new @ c2_new
    P_failure = np.sum(y_new < threshold) / len(y_new)

    c2_new_saved = c2_new
    Psi2_new_saved = Psi2_new

    return  1.35e-3 - P_failure 

def constraint3(d):
    global c3_new_saved
    global Psi3_new_saved

    Z_samples = X_samples - d
    M3 = compute_M_matrix(Z_samples, basis_terms3)
    Psi3_new = M3 @ W3.T

    A = Psi3.T @ Psi3
    B = Psi3.T @ (Psi3_new @ c3 )
    c3_new = np.linalg.solve(A,B)
    y_new   =  Psi3_new @ c3_new
    P_failure = np.sum(y_new < threshold) / len(y_new)

    c3_new_saved = c3_new
    Psi3_new_saved = Psi3_new

    return  1.35e-3 - P_failure 

def constraint_vector(d):
    return np.array([constraint1(d), constraint2(d), constraint3(d)])

################     MPSS   q = 1   ################################

X_samples = generate_qmc_normal_samples(mean, cov, 128)
d0 = d_init
d_old = np.array([10.0, 10.0])

Z0_samples = X_samples - d0
c1, mean_y1_ref, var_y1_ref, Psi1 = fit_surrogate( basis_terms1, W1, y1, Z0_samples)
c2, mean_y2_ref, var_y2_ref, Psi2 = fit_surrogate( basis_terms2, W2, y2, Z0_samples)
c3, mean_y3_ref, var_y3_ref, Psi3 = fit_surrogate( basis_terms3, W3, y3, Z0_samples)

subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
nonlinear_constraint = NonlinearConstraint(constraint_vector, lb=0, ub=np.inf)

result1 = differential_evolution(
    func=objective,
    bounds=subregion_bounds,             # 경계 조건 유지
    constraints=(nonlinear_constraint,), # ✅ 튜플/리스트로 묶기
    strategy='best1bin',
    maxiter=1000,
    popsize=15,
    tol=1e-3,
    disp=False
)

d_old = d0 # [5 5]
d0 = result1.x # [3.2 4.4]

constraint1(d0)
constraint2(d0)
constraint3(d0)

c1 = c1_new_saved
c2 = c2_new_saved 
c3 = c3_new_saved

Psi1 = Psi1_new_saved
Psi2 = Psi2_new_saved
Psi3 = Psi3_new_saved

objective_value_old = objective(d_old)
objective_value_new = objective(d0)

################     MPSS   q > 1  ################################
d_history = [d0.copy()]

while (
    np.linalg.norm(d0 - d_old) > err1 and 
    np.abs(objective_value_new - objective_value_old) > err2
):

    Z0_samples = X_samples - d0

    c_new = np.array([constraint1(d0), constraint2(d0), constraint3(d0)])
    c_old = np.array([constraint1(d_old), constraint2(d_old), constraint3(d_old)])

    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)
    print(beta)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
    print(subregion_bounds)

    nonlinear_constraint = NonlinearConstraint(constraint_vector, lb=0, ub=np.inf)

    result1 = differential_evolution(
        func=objective,
        bounds=subregion_bounds,             
        constraints=(nonlinear_constraint,), 
        strategy='best1bin',
        maxiter=1000,
        popsize=50,
        tol=1e-3,
        disp=False
    )

    d_old = d0 
    d0 = result1.x 

    constraint1(d0)
    constraint2(d0)
    constraint3(d0)

    c1   = c1_new_saved
    c2   = c2_new_saved
    c3   = c3_new_saved
    
    Psi1 = Psi1_new_saved
    Psi2 = Psi2_new_saved
    Psi3 = Psi3_new_saved


    d_history.append(d0.copy())
    print("d0 iteration history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")

    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)



print("d0 iteration history:")
for i, d in enumerate(d_history):
    print(f"iter {i}: {d}")
print(d0)