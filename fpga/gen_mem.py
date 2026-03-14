import numpy as np
import struct

# ─── Tham số ────────────────────────────────────────────────────────────────
NE       = 200
AP       = 30
FC1      = 100e6      # Hz
FC2      = 110e6      # Hz
FS       = 2e9        # Hz
DELTA_T  = 1.0 / FS   # = 0.5 ns
D_FRAC   = 14
D_SCALE  = 2**D_FRAC  # = 16384

# ─── 1. Carrier signal S[n] ─────────────────────────────────────────────────
# S[n] = Ap * (2 + cos(2π·fc1·n·ΔT) + cos(2π·fc2·n·ΔT))
# Dải giá trị: [0, 4·Ap] = [0, 120]  →  8-bit unsigned (0x00..0x78)
n = np.arange(NE)
S_float = AP * (2.0 + np.cos(2*np.pi*FC1*n*DELTA_T)
                    + np.cos(2*np.pi*FC2*n*DELTA_T))
S_int = np.round(S_float).astype(np.int32)
print(f"S range: [{S_int.min()}, {S_int.max()}]  (expected [0, 120])")
assert S_int.min() >= 0 and S_int.max() <= 255, "S out of 8-bit range!"

with open("./fpga/data/s_lut.mem", "w") as f:
    for val in S_int:
        f.write(f"{val & 0xFF:02X}\n")
print("✓ s_lut.mem written (200 entries, 8-bit hex)")

# ─── 2. Dictionary matrix D[NE × NE] ────────────────────────────────────────
# Xây dựng theo reconstruction.py: get_fourier_dict(NE)
#   Cột 0          : DC     → 1/√N
#   Cột 2k-1 (k≥1) : cosine → √(2/N)·cos(2π·k·n/N)
#   Cột 2k   (k≥1) : sine   → √(2/N)·sin(2π·k·n/N)
#   Cột NE-1       : Nyquist → 1/√N·cos(π·n)  (chỉ có khi N chẵn)
#
# Encode: D_int = round(D_float × 2^D_FRAC) → 16-bit signed two's complement
# Layout file: column-major → dòng thứ (k*NE + n) = D[n][k]

D = np.zeros((NE, NE), dtype=np.float64)
D[:, 0] = 1.0 / np.sqrt(NE)

k = 1
col = 1
while col < NE - 1:
    D[:, col]   = np.sqrt(2.0/NE) * np.cos(2*np.pi*k*n/NE)   # cosine
    D[:, col+1] = np.sqrt(2.0/NE) * np.sin(2*np.pi*k*n/NE)   # sine
    col += 2
    k   += 1

if NE % 2 == 0:
    D[:, NE-1] = (1.0/np.sqrt(NE)) * np.cos(np.pi * n)        # Nyquist

D_int = np.round(D * D_SCALE).astype(np.int16)
print(f"D range: [{D_int.min()}, {D_int.max()}]  (expected ~[-104, 104])")

# Lưu column-major: addr = k*NE + n
with open("./fpga/data/d_matrix.mem", "w") as f:
    for k_col in range(NE):
        for row in range(NE):
            val = int(D_int[row, k_col]) & 0xFFFF  # two's complement 16-bit
            f.write(f"{val:04X}\n")
print("✓ d_matrix.mem written (40000 entries, 16-bit hex, column-major)")

# ─── 3. Kiểm tra nhanh ──────────────────────────────────────────────────────
# Verify orthonormality: D^T @ D ≈ I
DtD = D.T @ D
err = np.max(np.abs(DtD - np.eye(NE)))
print(f"D orthonormality error (max|DᵀD - I|): {err:.2e}  (should be < 1e-13)")
assert err < 1e-10, "D is not orthonormal!"

print("\n✓ All memory files generated successfully.")
print("  Copy s_lut.mem và d_matrix.mem vào cùng thư mục với ModelSim project.")