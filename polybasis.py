import numpy as np
from itertools import combinations

# Monomial basis generation
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

# Compute M matrix
def compute_M_matrix(X, basis_terms):
    n, d = X.shape[0], len(basis_terms)
    M = np.zeros((n, d))
    for i, (nu, power) in enumerate(basis_terms):
        if len(nu) == 0:
            M[:, i] = 1
        else:
            M[:, i] = np.prod(X[:, list(nu)] ** power, axis=1)
    return M

