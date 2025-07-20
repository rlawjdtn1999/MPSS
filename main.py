import numpy as np
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples, fit_all_surrogates, rebase_all_surrogates, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
from scipy.optimize import differential_evolution, NonlinearConstraint
import Parameters   
import matplotlib.pyplot as plt


##########################      파라미터 설정    ########################## 
N         = Parameters.N  #2
S         = Parameters.S  #2 , GPCE와 똑같음
m         = Parameters.m  #[4, 4, 4]
m1        = Parameters.m[0] # 효율을 위해 하나로 통일
mean      = np.array(Parameters.mean)
cov       = np.array(Parameters.cov1)

d_init    = np.array([5.0, 5.0])    # 초기 설계 변수
d_bounds  = [(0.0, 10.0), (0.0, 10.0)]  

beta      = [0.3, 0.3]
penalty_coeff = 1e2     # 패널티 가중치
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

def unified_constraint_vector(d):
    # 3. 100만개 샘플에 대한 Psi_test 계산도 단 한 번만 수행
    Z_samples_test = X_samples - d
    M_test = compute_M_matrix(Z_samples_test, basis_terms)
    Psi_test = M_test @ W.T

    # 4. 각 제약조건의 파괴 확률 계산
    c1, c2, c3 = c_dict['y1'], c_dict['y2'], c_dict['y3']

    y1_new = Psi_test @ c1
    P_failure1 = np.sum(y1_new <= threshold) / len(y1_new)

    y2_new = Psi_test @ c2
    P_failure2 = np.sum(y2_new <= threshold) / len(y2_new)

    y3_new = Psi_test @ c3
    P_failure3 = np.sum(y3_new <= threshold) / len(y3_new)

    # 5. 세 제약조건의 결과를 벡터(배열)로 반환
    return np.array([
        P_failure1 - threesigma,
        P_failure2 - threesigma,
        P_failure3 - threesigma
    ])

def obj_with_penalty(d):
    f = objective(d)
    c = unified_constraint_vector(d)  # [P1-α, P2-α, P3-α]
    violation = np.maximum(0, c)
    return f + penalty_coeff  * np.sum(violation**2)

##########################   d 추적   ##########################
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(d_bounds[0])
ax.set_ylim(d_bounds[1])

# 빈 산점도 & 텍스트 리스트
scat = ax.scatter([], [], s=30)
texts = []
history = []

def plot_callback(xk, convergence):
    history.append(xk.copy())
    H = np.array(history)
    xs, ys = H[:,0], H[:,1]

    # 1) 산점도 갱신
    scat.set_offsets(np.c_[xs, ys])

    # 2) 이전 텍스트 지우기
    for txt in texts:
        txt.remove()
    texts.clear()

    # 3) 각 점에 좌표 텍스트 추가
    for xi, yi in zip(xs, ys):
        txt = ax.text(
            xi, yi,
            f"({xi:.2f}, {yi:.2f})",
            fontsize=8,
            ha='left', va='bottom'
        )
        texts.append(txt)

    # 4) 화면 갱신
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
################     MPSS   q = 1   ################################
# global_data
basis_terms = generate_monomial_basis(N, S, m1)
W = compute_whitening_matrix(N, S, m1, mean_0, cov )
X_samples_coef  = generate_qmc_normal_samples(mean, cov, 64)
X_samples   = generate_qmc_normal_samples(mean, cov, int(1e6))
y_functions = [y1, y2, y3]

d0 = d_init
Z0_samples = X_samples_coef - d0

c_dict, Psi = fit_all_surrogates(y_functions)

subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
print(subregion_bounds)

result1 = differential_evolution(
    func = obj_with_penalty, 
    bounds = subregion_bounds,
    # constraints = nlc_unified, 
    strategy='rand1bin',
    maxiter=15,
    popsize=20,
    mutation=(0.7, 1.5),
    recombination=0.7,
    tol=1e-2,
    workers=1,
    polish=False,   # 효율을 위해서    
    callback    = plot_callback,        
    disp=True
)

d_old = d0 # [5 5]
d0    = result1.x #e.g. [3.2 4.4]
print(d0)

c_dict, Psi = rebase_all_surrogates(d0, c_dict, Psi)

objective_value_old = objective(d_old)
objective_value_new = objective(d0)

c_new = unified_constraint_vector(d0)

###########################    find_feasible2     ################################
while ( np.any(c_new > 0)                   # <-- 하나라도 위반이 있으면(True) 계속
):

    c_new = unified_constraint_vector(d0) 
    c_old = unified_constraint_vector(d_old)

    print("Constraint violations:", c_new)
    print("All ≤0? ", np.all(c_new <= 0))
    print("  → constraint values:", c_new, "  all feasible?", np.all(c_new <= 0))
    
    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
    print(subregion_bounds)

    print("=== result2 시작: ")
    result2 = differential_evolution(
        func = obj_with_penalty, 
        bounds = subregion_bounds,
        strategy='rand1bin',
        maxiter=15,
        popsize=20,
        mutation=(0.7, 1.5),
        recombination=0.7,
        tol=1e-3,
        workers=1,  
        polish=False,  
        callback    = plot_callback, 
        disp=True
    )

    d_old = d0 
    d0 = result2.x 
    print("Final design: ", d0)

    c_dict, Psi = rebase_all_surrogates(d0, c_dict, Psi)
    
    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)


###########################    after_feasible     ################################
d_history = [d0.copy()]

while (
    np.linalg.norm(d0 - d_old) > err1 and 
    np.abs(objective_value_new - objective_value_old) > err2
):

    c_new = unified_constraint_vector(d0)
    c_old = unified_constraint_vector(d_old)    

    beta = update_beta(beta, d0, d_old, c_new, c_old, d_bounds, err_vals)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
    print(subregion_bounds)

    print("=== result3 시작: ")
    nlc_unified = NonlinearConstraint(unified_constraint_vector, -np.inf, 0.0)
    result3 = differential_evolution(
        func = objective, # 또는 obj_with_penalty
        bounds = subregion_bounds,
        constraints = nlc_unified,
        strategy='best1bin',
        maxiter=15,
        popsize=20,
        mutation=(0.7, 1.5),
        recombination=0.7,
        tol=1e-3,
        workers=1,  
        callback    = plot_callback,     
        polish=True,
        disp=True
    )

    d_old = d0 
    d0 = result3.x     
    print("Final design: ", d0)

    c_dict, Psi = rebase_all_surrogates(d0, c_dict, Psi)

    objective_value_old = objective(d_old)
    objective_value_new = objective(d0)
    
    d_history.append(d0.copy())
    print("d0 iteration history:")
    for i, d in enumerate(d_history):
        print(f"iter {i}: {d}")
