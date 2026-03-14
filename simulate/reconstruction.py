import numpy as np

from sklearn.linear_model import OrthogonalMatchingPursuit
from simulate.config import *

def get_fourier_dict(N):
    D = np.zeros((N, N))
    D[:, 0] = 1.0 / np.sqrt(N) # Thành phần DC
    
    for k in range(1, N // 2):
        D[:, 2*k - 1] = np.sqrt(2/N) * np.cos(2 * np.pi * k * np.arange(N) / N)
        D[:, 2*k]     = np.sqrt(2/N) * np.sin(2 * np.pi * k * np.arange(N) / N)
        
    D[:, -1] = 1.0 / np.sqrt(N) * np.cos(np.pi * np.arange(N))
    return D

# Khởi tạo ma trận từ điển dùng chung
D_fourier = get_fourier_dict(Ne)

def reconstruct(A, Po):
    Theta = A @ D_fourier

    # Sử dụng OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
    omp.fit(Theta, Po)
    coef_hat = omp.coef_

    # Khôi phục tín hiệu sóng mang
    S_rec = D_fourier @ coef_hat
    
    return S_rec