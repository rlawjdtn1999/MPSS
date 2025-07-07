import numpy as np


def update_beta(beta_old, d_new, d_old, c_new, c_old, d_bounds, err):

    phi = (1 + np.sqrt(5)) / 2   # 황금비 ≈1.618
    beta_new = np.array(beta_old, copy=True)
    M = len(beta_new)
    K = len(c_new)

    # --- Step 6-1 & 6-2: 제약 변화량 기준 (전 제약 l에 대해 검사) ---
    # 6-1: any constraint 변화가 작으면 β *= (2-1/φ)
    flag_inc = any(
        abs(c_new[l] - c_old[l]) <= err["err3"] * abs(c_new[l])
        for l in range(K)
    )
    # 6-2: any constraint 변화가 크면 β /= φ
    flag_dec = any(
        abs(c_new[l] - c_old[l]) >= err["err4"] * abs(c_new[l])
        for l in range(K)
    )

    if flag_inc:
        beta_new = np.minimum(1.0, (2 - 1/phi) * beta_new)
        return beta_new
    elif flag_dec:
        beta_new = beta_new / phi
        return beta_new

    # --- Step 6-3, 6-4, 6-5: 각 축별 세부 업데이트 ---
    for k in range(M):
        lb, ub = d_bounds[k]

        # 6-3: 설계변수 경계 active → β *= (2-1/φ)
        if (d_new[k] - lb) <= err["err5"] or (ub - d_new[k]) <= err["err5"]:
            beta_new[k] = min(1.0, (2 - 1/phi) * beta_new[k])

        # 6-4: 설계변수 변화 작음 → β /= φ
        if abs(d_new[k] - d_old[k]) <= err["err6"]:
            beta_new[k] = beta_new[k] / phi

        # 6-5: 최소 이동 한계 보장
        span = ub - lb
        if beta_new[k] * span <= err["err7"]:
            beta_new[k] = err["err7"] / span

    return beta_new



def get_subregion_bounds(d_center, beta, d_bounds):

    bounds = []
    for i in range(len(d_center)):
        dL, dU = d_bounds[i]
        half_range = (dU - dL) / 2
        lower = max(dL, d_center[i] - beta[i] * half_range)
        upper = min(dU, d_center[i] + beta[i] * half_range)
        bounds.append((lower, upper))
    return bounds
