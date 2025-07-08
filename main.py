import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples , fit_surrogate, rebase_surrogate, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
import Parameters   

# 파라미터 설정
N         = Parameters.N  #2
S         = Parameters.S  #1
m         = Parameters.m  #[3, 3, 4]
m1        = Parameters.m[0]
m2        = Parameters.m[1]
m3        = Parameters.m[2]
mean      = np.array(Parameters.mean)
cov      = np.array(Parameters.cov1)

d_init    = np.array([5.0, 5.0])    # 초기 설계 변수
d_bounds  = [(0.0, 10.0), (0.0, 10.0)]  

beta      = [0.3, 0.3]


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


basis_terms1 = generate_monomial_basis(N, S, m1)
basis_terms2 = generate_monomial_basis(N, S, m2)
basis_terms3 = generate_monomial_basis(N, S, m3)

W1 = compute_whitening_matrix(N, S, m1, mean_0, cov )
W2 = compute_whitening_matrix(N, S, m2, mean_0, cov )
W3 = compute_whitening_matrix(N, S, m3, mean_0, cov )

def objective(d):
    return -d[0] + d[1]

def constraint1(d):
    # Psi1, c1 이 old 다, W 는 고정
    Z0_samples = X0_samples - d     # 128
    M = compute_M_matrix(Z0_samples, basis_terms1)
    Psi1_new = M @ W1.T
    A = Psi1.T @ Psi1
    B = Psi1.T @ (Psi1_new @ c1 )
    c1_new = np.linalg.solve(A,B)    # d0 부터 c1 을 구한다

    Z_samples = X_samples - d       # 1e6
    M1 = compute_M_matrix(Z_samples, basis_terms1)
    Psi1_test = M1 @ W1.T            # Psi 를 구한다

    y_new   = Psi1_test @ c1_new
    P_failure = np.sum(y_new <= threshold) / len(y_new)

    return   P_failure -  1.35e-3 

def constraint2(d):
    # Psi2, c2 이 old 다 
    Z0_samples = X0_samples - d
    M = compute_M_matrix(Z0_samples, basis_terms2)
    Psi2_new = M @ W2.T
    A = Psi2.T @ Psi2
    B = Psi2.T @ (Psi2_new @ c2 )
    c2_new = np.linalg.solve(A,B)

    Z_samples = X_samples - d       # 1e6
    M2 = compute_M_matrix(Z_samples, basis_terms2)
    Psi2_test = M2 @ W2.T   

    y_new   =  Psi2_test @ c2_new
    P_failure = np.sum(y_new <= threshold) / len(y_new)

    return  P_failure -  1.35e-3 

def constraint3(d):
    # Psi3, c3 이 old 다 
    Z_samples = X_samples - d
    M = compute_M_matrix(Z_samples, basis_terms3)
    Psi3_new = M @ W3.T
    A = Psi3.T @ Psi3
    B = Psi3.T @ (Psi3_new @ c3 )
    c3_new = np.linalg.solve(A,B)

    Z_samples = X_samples - d       # 1e6
    M3 = compute_M_matrix(Z_samples, basis_terms3)
    Psi3_test = M3 @ W3.T   

    y_new   =  Psi3_test @ c3_new
    P_failure = np.sum(y_new <= threshold) / len(y_new)

    return  P_failure -  1.35e-3 

def constraint_vector(d):
    return np.array([constraint1(d), constraint2(d), constraint3(d)])

nlc = NonlinearConstraint(constraint_vector, -np.inf, 0.0)

cons = [constraint1, constraint2, constraint3]

def is_feasible(d):
    # 모든 c_i(d) ≤ 0 이면 feasible
    return all(c(d) <= 0 for c in cons)

def obj_feas(d):
    # c_i(d)=P_fail−threshold 이므로, max(0, c_i) 는 위반량
    return sum(max(0, c(d)) for c in cons)

def obj_piecewise(d,
                  gamma1=1e4,    # 제약 1개 위반 시 고정 페널티
                  gamma2=1e5,    # 제약 2개 위반 시 고정 페널티
                  gamma3=1e6,    # 제약 3개 위반 시 고정 페널티
                  gamma_size=1e5 # 위반량 크기에 대한 가중 페널티
                 ):
    # 각 제약별 위반량 계산
    viols = [max(0, c(d)) for c in cons]
    sum_viols = sum(viols)
    num_viol  = sum(1 for v in viols if v > 0)

    # 위반 개수별 기본 페널티 결정
    if num_viol == 0:
        base_pen = 0.0
    elif num_viol == 1:
        base_pen = gamma1
    elif num_viol == 2:
        base_pen = gamma2
    else:  # num_viol == 3
        base_pen = gamma3

    # 최종 페널티 = base_pen + gamma_size * sum_viols
    penalty = base_pen + gamma_size * sum_viols

    # feasible 이면 순수 objective, 아니면 objective + penalty
    return objective(d) + penalty


################     MPSS   q = 1   ################################

X0_samples  = generate_qmc_normal_samples(mean, cov, 128)
X_samples   = generate_qmc_normal_samples(mean, cov, int(1e6))

d0 = d_init
Z0_samples = X0_samples - d0

c1, Psi1 = fit_surrogate( basis_terms1, W1, y1, Z0_samples)
c2, Psi2 = fit_surrogate( basis_terms2, W2, y2, Z0_samples)
c3, Psi3 = fit_surrogate( basis_terms3, W3, y3, Z0_samples)

subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
print(subregion_bounds)

result1 = differential_evolution(
    func= obj_feas,
    bounds=subregion_bounds,            
    strategy='best1bin',
    maxiter=20,
    popsize=30,
    tol=1e-3,
    workers=-1,     
    disp=True
)

d_old = d0 # [5 5]
d0    = result1.x #e.g. [3.2 4.4]
print(d0)

c1, Psi1 = rebase_surrogate(d0, basis_terms1, W1, c1, Psi1, X0_samples)
c2, Psi2 = rebase_surrogate(d0, basis_terms2, W2, c2, Psi2, X0_samples)
c3, Psi3 = rebase_surrogate(d0, basis_terms3, W3, c3, Psi3, X0_samples)

objective_value_old = objective(d_old)
objective_value_new = objective(d0)

################    find feasible      ################################
d_history = [d0.copy()]

while not is_feasible(d0):

    Z0_samples = X_samples - d0

    c_new = np.array([constraint1(d0), constraint2(d0), constraint3(d0)])
    c_old = np.array([constraint1(d_old), constraint2(d_old), constraint3(d_old)])

    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)

    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)

    print(subregion_bounds)

    result2 = differential_evolution(
        func= obj_feas,
        bounds=subregion_bounds,            
        strategy='best1bin',
        maxiter=20,
        popsize=30,
        tol=1e-4,
        workers=1,      
        disp=True
    )

    d_old = d0 
    d0 = result2.x     
    print(d0)

    c1, Psi1 = rebase_surrogate(d0, basis_terms1, W1, c1, Psi1, X0_samples)
    c2, Psi2 = rebase_surrogate(d0, basis_terms2, W2, c2, Psi2, X0_samples)
    c3, Psi3 = rebase_surrogate(d0, basis_terms3, W3, c3, Psi3, X0_samples)

    d_history.append(d0.copy())
    print("find feasible :")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")

    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)


################    afet feasible hard    ################################)

d_history = [d0.copy()]

while (
    np.linalg.norm(d0 - d_old) > err1 and 
    np.abs(objective_value_new - objective_value_old) > err2
):

    Z0_samples = X_samples - d0

    c_new = np.array([constraint1(d0), constraint2(d0), constraint3(d0)])
    c_old = np.array([constraint1(d_old), constraint2(d_old), constraint3(d_old)])

    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)

    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)

    print(subregion_bounds)

    result3 = differential_evolution(
        func=objective,
        bounds=subregion_bounds,            
        strategy='best1bin',
        constraints=(nlc,),
        maxiter=10,
        popsize=30,
        tol=1e-4,
        workers=1,      
        disp=True
    )

    d_old = d0 
    d0 = result3.x     
    print(d0)

    c1, Psi1 = rebase_surrogate(d0, basis_terms1, W1, c1, Psi1, X0_samples)
    c2, Psi2 = rebase_surrogate(d0, basis_terms2, W2, c2, Psi2, X0_samples)
    c3, Psi3 = rebase_surrogate(d0, basis_terms3, W3, c3, Psi3, X0_samples)

    d_history.append(d0.copy())
    print("d0 iteration history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")

    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)

