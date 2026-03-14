import numpy as np

from simulate.config import *

def add_awgn(signal, snr_db):
    if snr_db is None:
        return signal
    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    noise        = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def generate(snr_db):
    n = np.arange(Ne)
    t = n * delta_T

    # Sóng mang gốc s(t)
    S = Ap * (2 + np.cos(2 * np.pi * Fc1 * t) + np.cos(2 * np.pi * Fc2 * t))
    S_noise = add_awgn(S, snr_db)
    
    # Nd_test có thể chạy từ 0 → Ne-1
    max_Nd = Ne - 1
    total_length = max_Nd + (M - 1)*Nc + Ne

    # Chỉ tạo 1 chuỗi g duy nhất
    g = np.random.randint(0, 2, size=total_length)

    # Tạo ma trận A đúng theo paper
    A = np.zeros((M, Ne))
    
    for m in range(M):
        start = Nd + m*Nc
        A[m,:] = g[start : start + Ne]

    # Tín hiệu nén quan sát được tại PD
    Po = A @ S_noise

    return t, S, A, Po, g