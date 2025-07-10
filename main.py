import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples , fit_surrogate, rebase_surrogate, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
import Parameters   

# 파라미터 설정
N         = Parameters.N  #2
S         = Parameters.S  #2
m         = Parameters.m  #[4, 4, 4]
m1        = Parameters.m[0] # 효율을 위해 하나로 통일함
mean      = np.array(Parameters.mean)
cov       = np.array(Parameters.cov1)

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


basis_terms = generate_monomial_basis(N, S, m1)

W = compute_whitening_matrix(N, S, m1, mean_0, cov )

def objective(d):
    return -d[0] + d[1]

def unified_constraint_vector(d):
    # 1. 가장 무거운 계산을 단 한 번만 수행
    Z0_samples = X0_samples - d
    M = compute_M_matrix(Z0_samples, basis_terms)
    Psi_rebase = M @ W.T

    # 2. 각 제약조건에 대한 c_new 계산 (이 부분은 비교적 가벼움)
    B1 = Psi1.T @ (Psi_rebase @ c1)
    c1_new = np.linalg.solve(A1, B1)

    B2 = Psi2.T @ (Psi_rebase @ c2)
    c2_new = np.linalg.solve(A2, B2)

    B3 = Psi3.T @ (Psi_rebase @ c3)
    c3_new = np.linalg.solve(A3, B3)

    # 3. 100만개 샘플에 대한 Psi_test 계산도 단 한 번만 수행
    Z_samples_test = X_samples - d
    M_test = compute_M_matrix(Z_samples_test, basis_terms)
    Psi_test = M_test @ W.T

    # 4. 각 제약조건의 파괴 확률 계산
    y1_new = Psi_test @ c1_new
    P_failure1 = np.sum(y1_new <= threshold) / len(y1_new)

    y2_new = Psi_test @ c2_new
    P_failure2 = np.sum(y2_new <= threshold) / len(y2_new)

    y3_new = Psi_test @ c3_new
    P_failure3 = np.sum(y3_new <= threshold) / len(y3_new)

    # 5. 세 제약조건의 결과를 벡터(배열)로 반환
    return np.array([
        P_failure1 - 1.35e-3,
        P_failure2 - 1.35e-3,
        P_failure3 - 1.35e-3
    ])


nlc_unified = NonlinearConstraint(unified_constraint_vector, -np.inf, 0.0)

################     MPSS   q = 1   ################################

X0_samples  = generate_qmc_normal_samples(mean, cov, 64)
X_samples   = generate_qmc_normal_samples(mean, cov, int(1e6))

d0 = d_init
Z0_samples = X0_samples - d0

c1, Psi1 = fit_surrogate( basis_terms, W, y1, Z0_samples)
c2, Psi2 = fit_surrogate( basis_terms, W, y2, Z0_samples)
c3, Psi3 = fit_surrogate( basis_terms, W, y3, Z0_samples)

A1 = Psi1.T @ Psi1
A2 = Psi2.T @ Psi2
A3 = Psi3.T @ Psi3


subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
print(subregion_bounds)


result1 = differential_evolution(
    func = objective, # 또는 obj_feas
    bounds = subregion_bounds,
    constraints = nlc_unified, # 수정된 부분
    maxiter=15,
    popsize=20,
    mutation=(0.7, 1.5),
    recombination=0.7,
    tol=1e-3,
    workers=-1,     
    disp=True
)

d_old = d0 # [5 5]
d0    = result1.x #e.g. [3.2 4.4]
print(d0)

c1, Psi1 = rebase_surrogate(d0, basis_terms, W, c1, Psi1, X0_samples)
c2, Psi2 = rebase_surrogate(d0, basis_terms, W, c2, Psi2, X0_samples)
c3, Psi3 = rebase_surrogate(d0, basis_terms, W, c3, Psi3, X0_samples)

objective_value_old = objective(d_old)
objective_value_new = objective(d0)

################    after feasible     ################################)
d_history = [d0.copy()]

while (
    np.linalg.norm(d0 - d_old) > err1 and 
    np.abs(objective_value_new - objective_value_old) > err2
):

    Z0_samples = X_samples - d0

    A1 = Psi1.T @ Psi1
    A2 = Psi2.T @ Psi2
    A3 = Psi3.T @ Psi3

    c_new = unified_constraint_vector(d0)
    c_old = unified_constraint_vector(d_old)

    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)

    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)

    print(subregion_bounds)

    result3 = differential_evolution(
        func = objective, # 또는 obj_feas
        bounds = subregion_bounds,
        constraints = nlc_unified, # 수정된 부분
        maxiter=15,
        popsize=20,
        mutation=(0.7, 1.5),
        recombination=0.7,
        tol=1e-3,
        workers=-1,     
        disp=True
    )


    d_old = d0 
    d0 = result3.x     
    print(d0)

    c1, Psi1 = rebase_surrogate(d0, basis_terms, W, c1, Psi1, X0_samples)
    c2, Psi2 = rebase_surrogate(d0, basis_terms, W, c2, Psi2, X0_samples)
    c3, Psi3 = rebase_surrogate(d0, basis_terms, W, c3, Psi3, X0_samples)

    d_history.append(d0.copy())
    print("d0 iteration history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")

    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)

