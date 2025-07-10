
import numpy as np
from scipy.stats import norm, qmc
from itertools import combinations
from numba import njit, prange 

def generate_monomial_basis(N, S, m):
    basis_terms = [((), ())]
    term_set = {"1"}
    for s in range(1, S + 1):
        for nu in combinations(range(N), s):
            for total_deg in range(1, m + 1):
                def gen_degree_powers(s, total):
                    if s == 1:
                        yield (total,)
                    else:
                        for i in range(total + 1):
                            for tail in gen_degree_powers(s - 1, total - i):
                                yield (i,) + tail
                for power in gen_degree_powers(s, total_deg):
                    term_str = "*".join(sorted([f"x{i+1}^{p}" for i, p in zip(nu, power) if p > 0]))
                    if term_str not in term_set:
                        basis_terms.append((nu, power))
                        term_set.add(term_str)
    return basis_terms


# def compute_M_matrix(X, basis_terms):
#     n = X.shape[0]
#     M = np.zeros((n, len(basis_terms)))
#     for i, (nu, power) in enumerate(basis_terms):
#         if len(nu) == 0:
#             M[:, i] = 1
#         else:
#             M[:, i] = np.prod(X[:, list(nu)]**power, axis=1)
#     return M

@njit(parallel=True, fastmath=True)
def compute_M_matrix_numba(X, basis_terms_nu, basis_terms_power):
    n_samples = X.shape[0]
    n_terms = len(basis_terms_nu)
    M = np.zeros((n_samples, n_terms))
    for i in range(n_terms):
        nu = basis_terms_nu[i]
        power = basis_terms_power[i]
        # 상수항
        if nu.size == 0:
            for j in prange(n_samples):
                M[j, i] = 1.0
            continue
        # 일반항
        for j in prange(n_samples):
            prod = 1.0
            for k in range(nu.size):
                prod *= X[j, nu[k]] ** power[k]
            M[j, i] = prod
    return M

# 외부에서 호출할 때는 wrapper를 사용해 basis_terms를 배열 리스트로 변환
def compute_M_matrix(X, basis_terms):
    """
    numpy 버전 대신 이 함수를 호출하세요.
    basis_terms: List of (nu: list[int], power: list[float])
    """
    # numba가 이해할 수 있게 int64, float64 배열로 변환
    basis_terms_nu = [np.array(bt[0], dtype=np.int64)   for bt in basis_terms]
    basis_terms_power = [np.array(bt[1], dtype=np.float64) for bt in basis_terms]
    return compute_M_matrix_numba(X, basis_terms_nu, basis_terms_power)

def sample_qmc_norm(mean, cov, sampler, batch_size):
    u = sampler.random(batch_size)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    z = norm.ppf(u)
    L = np.linalg.cholesky(cov)
    return mean + z @ L.T


def compute_whitening_matrix(N, S, m, mean, cov, total_samples=5000000, batch_size=1000000):
    # generate basis terms
    basis_terms = generate_monomial_basis(N, S, m)
    d_basis = len(basis_terms)

    # initialize sampler and accumulator
    sampler = qmc.Sobol(d=N, scramble=True)
    G_accum = np.zeros((d_basis, d_basis))

    # process full batches
    n_batches = total_samples // batch_size
    for _ in range(n_batches):
        X_batch = sample_qmc_norm(mean, cov, sampler, batch_size)
        M_batch = compute_M_matrix(X_batch, basis_terms)
        G_accum += M_batch.T @ M_batch

    # process remainder
    remainder = total_samples % batch_size
    if remainder:
        X_batch = sample_qmc_norm(mean, cov, sampler, remainder)
        M_batch = compute_M_matrix(X_batch, basis_terms)
        G_accum += M_batch.T @ M_batch

    # finalize moment matrix
    G = G_accum / total_samples
    # chol and whitening
    L = np.linalg.cholesky(G)
    W = np.linalg.inv(L)
    return W