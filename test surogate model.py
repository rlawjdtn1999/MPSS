import numpy as np
import matplotlib.pyplot as plt

# MPSS 프로젝트 모듈 임포트
from whitening       import compute_whitening_matrix
from polybasis       import generate_monomial_basis, compute_M_matrix
from functions       import generate_qmc_normal_samples, fit_surrogate, rebase_surrogate, y1, y2, y3
import Parameters

# 1) 파라미터 불러오기
N      = Parameters.N         # 차원
S      = Parameters.S         # 확률 차원 (보통 1)
m1     = Parameters.m[1]      # GPCE 차수
mean   = np.array(Parameters.mean)
cov    = np.array(Parameters.cov1)

# 2) 기준 설계점(d_init) 및 화이트닝 기준
d_init = np.array([5.0, 5.0])
mean_0 = mean - d_init

# 3) 기저(basis) 및 whitening 행렬
basis = generate_monomial_basis(N, S, m1)
W     = compute_whitening_matrix(N, S, m1, mean_0, cov)

# 4) 초기 QMC 샘플로 surrogate 학습
X0 = generate_qmc_normal_samples(mean, cov, 128)     # 초기 샘플
Z0 = X0 - d_init
c1_init, Psi1_init = fit_surrogate(basis, W, y1, Z0)
c2_init, Psi2_init = fit_surrogate(basis, W, y2, Z0)
c3_init, Psi3_init = fit_surrogate(basis, W, y3, Z0)


# 5) 그리드 및 QMC 샘플 준비
d1_vals = np.linspace(0, 10, 60)
d2_vals = np.linspace(0, 10, 60)
D1, D2   = np.meshgrid(d1_vals, d2_vals)

X_surr   = generate_qmc_normal_samples(mean, cov, int(1e6))  # 대규모 샘플
# c2 
# 2e5   4:08 ~ 4:10(2m)    >> 다가능
# 3e5   4:04 ~ 4:07(3m)    >> 부정확
# 5e5   4:11 ~ 4:16(4m)    >> 부정확    
# MPSS를 사용해야 정확해질듯

# 6) 각 (d1,d2) 점에서 rebasing 후 실패확률 계산
c_d = c2_init
PF1 = np.zeros_like(D1)
for i in range(D1.shape[0]):
    for j in range(D1.shape[1]):
        d = np.array([D1[i, j], D2[i, j]])
        # rebasing
        c_d, _ = rebase_surrogate(d, basis, W, c_d, Psi2_init, X0)
        # surrogate 예측
        Z = X_surr - d
        Psi = compute_M_matrix(Z, basis) @ W.T
        Yh  = Psi @ c_d
        PF1[i, j] = np.mean(Yh < 0)


# 7) 플롯
threshold = 1.35e-3   # Phi(-3)

mask = PF1 <= threshold

plt.figure(figsize=(6,6))
# feasible 영역(mask==True)만 파란 음영
plt.contourf(D1, D2, mask, levels=[-0.5, 0.5, 1.5],
             colors=['lightgray','lightblue'], alpha=0.7)
# c1=0 경계: PF1 == threshold
cont = plt.contour(D1, D2, PF1, levels=[threshold],
                   colors='blue', linestyles='--')
plt.clabel(cont, fmt='$c(d)=0$', colors='blue')

plt.xlabel('$d_1$')
plt.ylabel('$d_2$')
plt.tight_layout()
plt.show()
