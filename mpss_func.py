import numpy as np


def update_beta(beta_old, d0, d_old, c_new, c_old, d_bounds, err_vals):
    """
    Parameters:
    - beta_old: 현재 beta 리스트
    - d0: 현재 설계 벡터
    - d_old: 이전 설계 벡터
    - c_new: 현재 constraint 값들 (np.array)
    - c_old: 이전 constraint 값들 (np.array)
    - d_bounds: 설계 변수 하한/상한 [(L, U), ...]
    - err_vals: tolerance 값들 dict (err3 ~ err7)

    Returns:
    - beta_new: 업데이트된 beta 리스트
    """
    phi = 1.618  # 황금비
    beta_new = beta_old.copy()
    M = len(d0)

    for k in range(M):
        # Step 6-1: constraint 변화가 작으면 beta 증가
        if np.linalg.norm(c_new - c_old) <= err_vals["err3"] * np.linalg.norm(c_new):
            beta_new[k] = min(1.0, (2 - 1/phi) * beta_old[k])

        # Step 6-2: constraint 변화가 크면 beta 감소
        elif np.linalg.norm(c_new - c_old) >= err_vals["err4"] * np.linalg.norm(c_new):
            beta_new[k] = beta_old[k] / phi

        # Step 6-3: d가 경계에 가까우면 beta 증가
        if abs(d0[k] - d_bounds[k][0]) <= err_vals["err5"] or abs(d0[k] - d_bounds[k][1]) <= err_vals["err5"]:
            beta_new[k] = min(1.0, (2 - 1/phi) * beta_old[k])

        # Step 6-4: d 변화가 작으면 beta 감소
        if abs(d0[k] - d_old[k]) <= err_vals["err6"]:
            beta_new[k] = beta_old[k] / phi

        # Step 6-5: beta 너무 작아지면 최소값 유지
        min_beta = err_vals["err7"] / (d_bounds[k][1] - d_bounds[k][0])
        if beta_new[k] < min_beta:
            beta_new[k] = min_beta

    return beta_new



def get_subregion_bounds(d_center, beta, d_bounds):
    """
    d_center: 중심 설계값 (예: [5.0, 5.0, 3.0]) 서브 리즌마다 업데이트
    beta: subregion 스케일 (예: [0.3, 0.3, 0.2]) 서브 리즌마다 업데이트
    d_bounds: 전체 설계공간 경계 (예: [(0, 10), (0, 10), (1, 5)]) 이건 고정
    
    return: subregion 경계 [(lower1, upper1), (lower2, upper2), ...]
    """
    bounds = []
    for i in range(len(d_center)):
        dL, dU = d_bounds[i]
        half_range = (dU - dL) / 2
        lower = max(dL, d_center[i] - beta[i] * half_range)
        upper = min(dU, d_center[i] + beta[i] * half_range)
        bounds.append((lower, upper))
    return bounds
