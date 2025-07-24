import numpy as np
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples, fit_all_surrogates, rebase_all_surrogates, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
from scipy.optimize import differential_evolution, NonlinearConstraint
import Parameters   
import matplotlib.pyplot as plt

##########################      Parameters        ########################## 
N         = Parameters.N  #2
S         = Parameters.S  #2 , GPCE와 똑같음
m         = Parameters.m  #[4, 3, 2 ]
m1        = Parameters.m[0] # 효율을 위해 하나로 통일

d_init    = np.array([5.0, 5.0])    # 초기 설계 변수
mean      = d_init
cov       = np.array(Parameters.cov1)

d_bounds  = [(0.0, 10.0), (0.0, 10.0)]  
beta      = [0.3, 0.3]

penalty_coeff = 1e9     # 패널티 가중치 
threesigma    = 1.35e-3    # 실패 확률 예제는 3시그마

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
mean_0      = [0 , 0]

##########################      func    ##########################
def objective(d):
    return -d[0] + d[1]

# single_step
def unified_constraint_vector(d):
    # 100만개 샘플에 대한 Psi_test 계산 한 번만 수행
    Z_samples_test = Z_samples - d0 + d
    M_test = compute_M_matrix(Z_samples_test, basis_terms)
    Psi_test = M_test @ W.T

    # 각 제약조건의 실패 확률 계산
    c1, c2, c3 = c_dict['y1'], c_dict['y2'], c_dict['y3']

    y1_new = Psi_test @ c1
    P_failure1 = np.sum(y1_new <= threshold) / len(y1_new)

    y2_new = Psi_test @ c2
    P_failure2 = np.sum(y2_new <= threshold) / len(y2_new)

    y3_new = Psi_test @ c3
    P_failure3 = np.sum(y3_new <= threshold) / len(y3_new)

    # 세 제약조건의 결과를 벡터(배열)로 반환
    return np.array([
        P_failure1 - threesigma,
        P_failure2 - threesigma,
        P_failure3 - threesigma
    ])

def obj_with_penalty(d):
    # global last_violations 
    f = objective(d)
    c = unified_constraint_vector(d)  # [P1-α, P2-α, P3-α]  
    violation = np.maximum(0, c)
    return f + penalty_coeff  * np.sum(violation**2)

def status_report(d, convergence):

    violations = unified_constraint_vector(d)
    # 2) 보기 좋게 출력
    print(f"d = [{d[0]:.4f}, {d[1]:.4f}], violations = {violations}")
    
#####################        q = 1         ################################
X_samples_coef  = generate_qmc_normal_samples(d_init, cov, 64)
X_samples   = generate_qmc_normal_samples(d_init, cov, int(1e6))
Z_samples_coef = X_samples_coef - d_init    # 평균 0
Z_samples = X_samples - d_init              # 평균 0


basis_terms = generate_monomial_basis(N, S, m1)
W = compute_whitening_matrix(N, S, m1, mean_0, cov )
y_functions = [y1, y2, y3]

d0    = d_init

c_dict = fit_all_surrogates(basis_terms, W, X_samples_coef, Z_samples_coef, y_functions)

c_old  = unified_constraint_vector(0)

subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
print(subregion_bounds)

result1 = differential_evolution(
    func = obj_with_penalty,  #objective
    bounds = subregion_bounds, 
    strategy='rand1bin',
    maxiter=5,
    popsize=50,
    mutation=(0.7, 1.5),
    recombination=0.7,
    tol=1e-2,
    workers=1,
    polish=False,   # 효율을 위해서    
    callback = status_report, ## status_report , plot_callback
    disp=True
)

d_old = d0        # [5 5]
d0    = result1.x # [3.2 4.4]

print(d_old)
print(d0)

c_new  = unified_constraint_vector(d0)
c_dict = rebase_all_surrogates(basis_terms, W, Z_samples_coef, d0, d_old, c_dict)
objective_value_old = objective(d_old)
objective_value_new = objective(d0)

###########################    after_feasible     ################################
d_history = [d0.copy()]
c_dict_history = [c_dict.copy()]
popsize = 20

while (
    np.linalg.norm(d0 - d_old) > err1 and 
    np.abs(objective_value_new - objective_value_old) > err2
):
   
    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)

    lb = np.array([b[0] for b in subregion_bounds])
    ub = np.array([b[1] for b in subregion_bounds])
    print(subregion_bounds)

    warm_pop = np.vstack([
        d0, 
        np.random.default_rng(123).uniform(lb, ub, size=(popsize-1, len(d0)))
    ])

    print("=== result3 시작: ")
    result3 = differential_evolution(
        func = obj_with_penalty, 
        bounds = subregion_bounds,
        strategy='best1bin',
        maxiter=15,
        popsize = 20,
        mutation=(0.7, 1.5),
        recombination=0.7,
        tol=1e-3,
        workers=1,
        init = warm_pop,  
        callback    = status_report,   ## status_report , plot_callback   
        polish=False,
        disp=True
    )

    d_old = d0 
    d0 = result3.x     

    c_new = unified_constraint_vector(d0)
    c_old = unified_constraint_vector(d_old) 

    objective_value_new = objective(d0)
    objective_value_old = objective(d_old)

    print("Final design: ", d0)
    c_dict = rebase_all_surrogates(basis_terms, W, Z_samples_coef, d0, d_old, c_dict)
    c_dict_history.append(c_dict.copy())
    d_history.append(d0.copy())

    print("d0 iteration history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")
    print("c_dict history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")