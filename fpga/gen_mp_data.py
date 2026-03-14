#!/usr/bin/env python3
"""
gen_mp_test_data.py  –  Sinh theta.mem và po.mem cho Nd_t = ND_REAL = 100
Chạy 1 lần trước khi simulate MP core trong ModelSim.

Output (cùng thư mục với file .sv):
    theta.mem  –  4000 entries × 6 hex digits (Theta[M×NE], 24-bit signed)
    po.mem     –    20 entries × 4 hex digits (Po[M],       16-bit signed)
"""

import numpy as np

# ─── Tham số (khớp config_pkg.sv) ──────────────────────────────────────────
NE      = 200
NC      = 200
M_OBS   = 20
ND_REAL = 100     # Nd thật → đây là input đo được của sensor
FC1     = 100e6
FC2     = 110e6
FS      = 2e9
DT      = 1.0 / FS
AP      = 30
FRAC_D  = 14      # D_INT = round(D × 2^14)
SEED    = 0xACE12345

# ─── 1. Carrier S[n] ────────────────────────────────────────────────────────
n     = np.arange(NE)
S_int = np.round(AP * (2 + np.cos(2*np.pi*FC1*n*DT)
                         + np.cos(2*np.pi*FC2*n*DT))).astype(np.int32)
print(f"S  : range [{S_int.min()}, {S_int.max()}]  (expected [1,120])")

# ─── 2. Dictionary D[NE×NE] Q1.14 ──────────────────────────────────────────
D = np.zeros((NE, NE), dtype=np.float64)
D[:, 0] = 1.0 / np.sqrt(NE)
k, col = 1, 1
while col < NE - 1:
    D[:, col]   = np.sqrt(2.0/NE) * np.cos(2*np.pi*k*n/NE)
    D[:, col+1] = np.sqrt(2.0/NE) * np.sin(2*np.pi*k*n/NE)
    col += 2; k += 1
if NE % 2 == 0:
    D[:, NE-1] = (1.0/np.sqrt(NE)) * np.cos(np.pi * n)
D_int = np.round(D * 2**FRAC_D).astype(np.int16)
print(f"D  : range [{D_int.min()}, {D_int.max()}]  (expected ~[-1638,1638])")

# ─── 3. LFSR g[] (khớp lfsr_g.sv) ──────────────────────────────────────────
G_LEN = (NC - 1) + (M_OBS - 1) * NC + NE  # = 4199
lfsr  = SEED & 0xFFFFFFFF
g     = np.zeros(G_LEN, dtype=np.uint8)
for i in range(G_LEN):
    g[i]    = lfsr & 1
    new_bit = ((lfsr>>31) ^ (lfsr>>21) ^ (lfsr>>1) ^ lfsr) & 1
    lfsr    = ((lfsr >> 1) | (new_bit << 31)) & 0xFFFFFFFF
print(f"g  : {int(g.sum())} ones / {G_LEN}  ({100*g.mean():.1f}%)")

# ─── 4. Po[M] = Σ_{n: g[ND_REAL+m*NC+n]=1} S[n] ────────────────────────────
Po = np.zeros(M_OBS, dtype=np.int32)
for m in range(M_OBS):
    for ni in range(NE):
        if g[ND_REAL + m*NC + ni]:
            Po[m] += S_int[ni]
print(f"Po : range [{Po.min()}, {Po.max()}]  (expected ~[5000,9000])")

# ─── 5. Theta[M×NE] = A(ND_REAL) @ D_int ────────────────────────────────────
# Dùng ND_REAL để test: ModelSim sẽ chạy MP với đúng input thật
Theta = np.zeros((M_OBS, NE), dtype=np.int32)
for m in range(M_OBS):
    for ni in range(NE):
        if g[ND_REAL + m*NC + ni]:
            Theta[m, :] += D_int[ni, :].astype(np.int32)
print(f"Theta: range [{Theta.min()}, {Theta.max()}]  (expected ~[-2000,9000])")

# ─── 6. Verify nhanh (Python MP để biết coefs đúng phải ra sao) ─────────────
print("\n--- Python MP reference (K=7) ---")
r     = Po.astype(np.float64)
coefs = np.zeros(NE, dtype=np.float64)
Theta_f = Theta.astype(np.float64)
for iteration in range(7):
    corr    = Theta_f.T @ r
    idx     = np.argmax(np.abs(corr))
    norm_sq = Theta_f[:, idx] @ Theta_f[:, idx]
    alpha   = corr[idx] / norm_sq
    coefs[idx] += alpha
    r -= alpha * Theta_f[:, idx]
    print(f"  K={iteration+1}: idx={idx:3d}  alpha={alpha:10.2f}  "
          f"|r|={np.linalg.norm(r):.1f}")

D_f   = D_int.astype(np.float64) / 2**FRAC_D
S_rec = D_f @ coefs
mse   = np.sum((S_int.astype(np.float64) - S_rec)**2)
print(f"MSE tại Nd=100 (Python ref): {mse:.1f}")
print(f"Nonzero coefs: {np.sum(np.abs(coefs) > 0.01)}")

# ─── 7. Ghi theta.mem ────────────────────────────────────────────────────────
# Format: M_OBS × NE entries, mỗi entry 6 hex digits (24-bit two's complement)
# Layout dòng: m*NE + k  →  Theta[m][k]
with open("./fpga/data/theta.mem", "w") as f:
    for m in range(M_OBS):
        for k in range(NE):
            val = int(Theta[m, k]) & 0xFFFFFF
            f.write(f"{val:06X}\n")
print(f"\n✓ theta.mem  ({M_OBS*NE} entries, 24-bit hex)")

# ─── 8. Ghi po.mem ───────────────────────────────────────────────────────────
with open("./fpga/data/po.mem", "w") as f:
    for m in range(M_OBS):
        val = int(Po[m]) & 0xFFFF
        f.write(f"{val:04X}\n")
print(f"✓ po.mem     ({M_OBS} entries, 16-bit hex)")
print("\nCopy theta.mem và po.mem vào thư mục project ModelSim rồi chạy simulation.")