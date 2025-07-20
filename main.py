import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from whitening import compute_whitening_matrix
from polybasis import generate_monomial_basis, compute_M_matrix
from functions import generate_qmc_normal_samples, fit_all_surrogates, rebase_all_surrogates, y1, y2, y3
from mpss_func import get_subregion_bounds, update_beta
import Parameters
import matplotlib.pyplot as plt

##########################      파라미터 설정    ##########################
N = Parameters.N
S = Parameters.S
m = Parameters.m
m1 = Parameters.m[0]
mean = np.array(Parameters.mean)
cov = np.array(Parameters.cov1)

d_init = np.array([5.0, 5.0])
d_bounds = [(0.0, 10.0), (0.0, 10.0)]

beta = [0.3, 0.3]
penalty_coeff = 1e2
threesigma = 1.35e-3

err1 = Parameters.err1
err2 = Parameters.err2
err_vals = {
    "err3": Parameters.err3, "err4": Parameters.err4, "err5": Parameters.err5,
    "err6": Parameters.err6, "err7": Parameters.err7
}
threshold = 0
mean_0 = mean - d_init

y_functions = [y1, y2, y3]

basis_terms = generate_monomial_basis(N, S, m1)
W = compute_whitening_matrix(N, S, m1, mean_0, cov)

##########################      func    ##########################

def objective(d):
    return -d[0] + d[1]

def unified_constraint_vector(d, c_dict, current_X_samples):
    Z_samples_test = current_X_samples - d
    M_test = compute_M_matrix(Z_samples_test, basis_terms)
    Psi_test = M_test @ W.T

    constraints = []
    for y_func in y_functions:
        c_vector = c_dict[y_func.__name__]
        y_new = Psi_test @ c_vector
        p_failure = np.sum(y_new <= threshold) / len(y_new)
        constraints.append(p_failure - threesigma)

    return np.array(constraints)

def obj_with_penalty(d, c_dict, current_X_samples):
    f = objective(d)
    c = unified_constraint_vector(d, c_dict, current_X_samples)
    violation = np.maximum(0, c)
    return f + penalty_coeff * np.sum(violation**2)

##########################   d 추적   ##########################
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(d_bounds[0])
ax.set_ylim(d_bounds[1])
scat = ax.scatter([], [], s=30)
texts = []
history = []

def plot_callback(xk, convergence):
    history.append(xk.copy())
    H = np.array(history)
    scat.set_offsets(np.c_[H[:,0], H[:,1]])
    for txt in texts:
        txt.remove()
    texts.clear()
    for xi, yi in zip(H[:,0], H[:,1]):
        texts.append(ax.text(xi, yi, f"({xi:.2f}, {yi:.2f})", fontsize=8))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

################     MPSS   q = 1   ################################

X_samples_coef = generate_qmc_normal_samples(mean, cov, 64)
X_samples_eval = generate_qmc_normal_samples(mean, cov, int(1e6))

d0 = d_init
Z0_samples = X_samples_coef - d0

# Fit all surrogates at once
c_dict, Psi = fit_all_surrogates(basis_terms, W, y_functions, Z0_samples)

###########################    MAIN OPTIMIZATION LOOP    ################################
d_old = d0
objective_value_old = objective(d_old)

MAX_ITER = 30
for i in range(MAX_ITER):
    print(f"\n--- Iteration {i+1}/{MAX_ITER} ---")

    # 1. Check feasibility and find a feasible point if necessary
    c_current = unified_constraint_vector(d0, c_dict, X_samples_eval)
    if np.any(c_current > 0):
        print(f"Current design {d0} is infeasible. Violations: {c_current[c_current > 0]}")
        # Simple interpolation to find a feasible point (can be improved)
        d_safe_guess = np.array([5.0, 5.0]) # A known safe point
        d0 = 0.7 * d0 + 0.3 * d_safe_guess
        print(f"Moved to a new trial point: {d0}")
        # Update surrogates for the new point
        c_dict, Psi = rebase_all_surrogates(d0, basis_terms, W, c_dict, Psi, X_samples_coef)
        continue # Restart the loop with the new feasible point

    # 2. Update subregion and perform optimization
    c_old_eval = unified_constraint_vector(d_old, c_dict, X_samples_eval)
    c_new_eval = unified_constraint_vector(d0, c_dict, X_samples_eval)
    beta = update_beta(beta, d0, d_old, c_new_eval, c_old_eval, d_bounds, err_vals)
    subregion_bounds = get_subregion_bounds(d0, beta, d_bounds)
    print(f"Subregion for optimization: {subregion_bounds}")

    # Use a lambda function to pass the current c_dict to the objective
    obj_func_for_opt = lambda d: obj_with_penalty(d, c_dict, X_samples_eval)
    if not np.any(c_new_eval > 0):
        obj_func_for_opt = objective

    nlc = NonlinearConstraint(lambda d: unified_constraint_vector(d, c_dict, X_samples_eval), -np.inf, 0.0)

    result = differential_evolution(
        func=obj_func_for_opt,
        bounds=subregion_bounds,
        constraints=nlc if obj_func_for_opt == objective else None,
        strategy='best1bin', maxiter=15, popsize=20,
        mutation=(0.7, 1.5), recombination=0.7, tol=1e-3,
        workers=1, polish=True, callback=plot_callback, disp=True
    )

    d_old = d0
    d0 = result.x
    objective_value_new = objective(d0)

    # 3. Update surrogates for the next iteration
    c_dict, Psi = rebase_all_surrogates(d0, basis_terms, W, c_dict, Psi, X_samples_coef)

    # 4. Check for convergence
    if np.linalg.norm(d0 - d_old) < err1 and np.abs(objective_value_new - objective_value_old) < err2:
        print("\nConvergence criteria met. Stopping optimization.")
        break

    objective_value_old = objective_value_new
    d_history.append(d0.copy())
    print(f"End of Iteration {i+1}: d = {d0}, objective = {objective_value_new}")

print("\n################# OPTIMIZATION FINISHED #################")
print("최적 설계 변수:", d0)
print("최적 목적 함수 값:", objective(d0))
print("최종 제약 조건 값:", unified_constraint_vector(d0, c_dict, X_samples_eval))

plt.ioff()
plt.show()