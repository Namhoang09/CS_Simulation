import numpy as np

from simulate.config import *
from simulate.config import FRAC_S, FRAC_D

def quantize(S, A, D):
    # S_INT: int32
    # S ∈ [0, 120] × 4096 → S_INT ∈ [0, 491520]
    S_INT = np.round(S * (1 << FRAC_S)).astype(np.int32) 

    # A_INT: int8 (binary 0/1)
    A_INT = A.astype(np.int8)

    # D_INT: int16
    # D_fourier ∈ [-0.1, 0.1] × 1024 → D_INT ∈ [-103, 103]
    D_INT = np.round(D * (1 << FRAC_D)).astype(np.int16) 

    # Theta_INT: int32
    # Mỗi phần tử = Σ(A_row × D_col), max = Ne × 103 = 20600
    Theta_INT = A_INT.astype(np.int32) @ D_INT.astype(np.int32)

    return S_INT, A_INT, D_INT, Theta_INT

def quantize_po(Po):
    # Po_INT: int32
    # Mỗi phần tử ≤ Ne × S_INT_max = 200 × 491520 = 98M
    Po_INT = np.round(Po * (1 << FRAC_S)).astype(np.int32)

    return Po_INT

def compute_norm_sq(Theta_int):
    return np.sum(Theta_int.astype(np.int64) ** 2, axis=0)

def mp_integer(Theta_int, Po_int, norm_sq, k):
    r = Po_int.astype(np.int64).copy()
    coef_int = np.zeros(Ne, dtype=np.int64)

    for _ in range(k):
        # Tương quan
        corr = Theta_int.T.astype(np.int64) @ r

        # Argmax
        idx = int(np.argmax(np.abs(corr)))

        # Alpha 
        n = int(norm_sq[idx])
        alpha = int(corr[idx]) // n if n != 0 else 0 

        # Tích lũy hệ số 
        coef_int[idx] += alpha

        # Cập nhật residual
        r -= Theta_int[:, idx].astype(np.int64) * alpha

    return coef_int

def reconstruct_int(D_int, coef_int):
    raw = D_int.astype(np.int64) @ coef_int
    return raw.astype(np.float64) / (1 << FRAC_S)