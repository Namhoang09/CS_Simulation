import numpy as np
import time
import os

from simulate.config import *
from simulate.measurement import generate
from simulate.reconstruction import get_fourier_dict
from simulate.evaluation import calculate
from simulate.mp import *

# Cấu hình cho phần MP integer
REPEATS = 100    
RESULTS_PATH = "statistic/Results.md"
STATS_PATH   = "statistic/MP_stats.md"

# OMP float
def run_omp():
    times, errs = [], []
    details = []
    for _ in range(REPEATS):
        _, S, _, Po, g = generate(SNR_DB)

        t0 = time.perf_counter()
        distances, rmse_list = calculate(g, S, Po)

        elapsed = (time.perf_counter() - t0) * 1e3
        best_dist = distances[int(np.argmin(rmse_list))]
        err = abs(best_dist - actual_dist)

        times.append(elapsed)
        errs.append(err)
        details.append((elapsed, best_dist, err))

    return round(np.mean(times), 2), round(np.mean(errs), 4), details

# MP integer
def run_mp(k):
    times, errs  = [], []
    details = []
    for _ in range(REPEATS):
        _, S, A, Po, g = generate(SNR_DB)
        D = get_fourier_dict(Ne)
        _, _, D_INT, _ = quantize(S, A, D)
        Po_INT = quantize_po(Po)

        t0 = time.perf_counter()
        rmse_list = []
        hypothetical_Nd = np.arange(Ne)
        distances = c * hypothetical_Nd * delta_T

        for Nd_t in hypothetical_Nd:
            A_t = np.zeros((M, Ne), dtype=np.int8)
            for m in range(M):
                A_t[m, :] = g[Nd_t + m * Nc: Nd_t + m * Nc + Ne]

            Theta_t = A_t.astype(np.int32) @ D_INT.astype(np.int32)
            norm_sq_t  = compute_norm_sq(Theta_t)
            coef_t = mp_integer(Theta_t, Po_INT, norm_sq_t, k)
            S_rec_t = reconstruct_int(D_INT, coef_t)
            rmse_list.append(np.sqrt(np.mean((S - S_rec_t) ** 2)))

        elapsed = (time.perf_counter() - t0) * 1e3
        best_dist = distances[int(np.argmin(rmse_list))]
        err = abs(best_dist - actual_dist)

        times.append(elapsed)
        errs.append(err)
        details.append((elapsed, best_dist, err))

    return round(np.mean(times), 2), round(np.mean(errs), 4), details

mp_time, mp_err, mp_details = run_mp(K_MP)
omp_time, omp_err, omp_details = run_omp()
speedup = round(omp_time / mp_time, 2) if mp_time > 0 else 0

print(f"MP integer : {mp_time} ms  |  Sai số khoảng cách: {mp_err} m")
print(f"OMP float  : {omp_time} ms  |  Sai số khoảng cách: {omp_err} m")

# Đọc / ghi bảng Markdown
def read_table(text, headers):
    rows, n = [], len(headers)
    in_table = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            in_table = False
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        inner = line.strip("|").replace("-","").replace("|","").strip()
        if not inner:                         # dòng separator
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != n:
            continue
        if parts == headers:                  # dòng header
            in_table = True
            continue
        if not in_table:
            continue
        row = []
        for p in parts:
            try:    row.append(int(p))
            except ValueError:
                try: row.append(float(p))
                except ValueError: row.append(p)
        rows.append(row)
    return rows

def render_table(headers, data_rows):
    all_r      = [headers] + [[str(v) for v in r] for r in data_rows]
    widths     = [max(len(r[c]) for r in all_r) for c in range(len(headers))]
    def fmt(row): return "| " + " | ".join(str(v).ljust(widths[i]) for i,v in enumerate(row)) + " |"
    sep = "| " + " | ".join("-"*w for w in widths) + " |"
    return "\n".join([fmt(headers), sep] + [fmt([str(v) for v in r]) for r in data_rows])

def upsert(data_rows, new_row, key_cols):
    key = tuple(new_row[c] for c in key_cols)
    for i, row in enumerate(data_rows):
        try:
            if tuple(row[c] for c in key_cols) == key:
                data_rows[i] = new_row
                return data_rows, True
        except IndexError:
            continue
    data_rows.append(new_row)
    return data_rows, False

#  Bảng: MP vs OMP
H1 = ["M", "K_MP", "SNR", "Dist thực tế", "MP time", "MP sai số", "OMP time", "OMP sai số", "Speedup"]

row1 = [M, K_MP, SNR_DB, f"{actual_dist:.2f}", mp_time, mp_err, omp_time, omp_err, speedup]
old_text = open(RESULTS_PATH, encoding="utf-8").read() if os.path.exists(RESULTS_PATH) else ""

import re as _re
_m = _re.search(r'Repeats=(\d+)', old_text)
old_repeats = int(_m.group(1)) if _m else None
if old_repeats != REPEATS:
    if old_repeats is not None:
        print(f"Repeats thay đổi ({old_repeats} → {REPEATS}) → reset bảng")
    t1_rows = [row1]
else:
    t1_rows, _ = upsert(read_table(old_text, H1), row1, key_cols=[0, 1, 2])

#  Ghi file
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("# Benchmark: MP Integer vs OMP Float\n\n")
    f.write(f"> Tham số: Nc={Nc}, Ne={Ne}, Nd={Nd}, FRAC_D={FRAC_D}, FRAC_S={FRAC_S}, Repeats={REPEATS}\n\n")
    f.write(render_table(H1, t1_rows))
    f.write("\n")

print(f"Đã lưu: {RESULTS_PATH}")

# Bảng chi tiết
H2 = ["Lần chạy", "Thời gian MP", "MP ước lượng", "Sai số MP", "Thời gian OMP", "OMP ước lượng", "Sai số OMP"]

stat_rows = []
for i, ((t_ms_mp, best_mp, err_mp), (t_ms_omp, best_omp, err_omp)) in enumerate(zip(mp_details, omp_details), start=1):
    stat_rows.append([i, f"{t_ms_mp:.4f}", f"{best_mp:.2f}", f"{err_mp:.4f}", f"{t_ms_omp:.4f}", f"{best_omp:.2f}", f"{err_omp:.4f}"])

stat_rows.append(["TB", f"{mp_time:.4f}", "—", f"{mp_err:.4f}", f"{omp_time:.4f}", "—", f"{omp_err:.4f}"])

#  Ghi file
with open(STATS_PATH, "w", encoding="utf-8") as f:
    f.write("# Thống kê chi tiết\n\n")
    f.write(f"> Nc={Nc}, Ne={Ne}, Nd={Nd}, M={M}, K_MP={K_MP}, SNR={SNR_DB}, Repeats={REPEATS}\n\n")
    f.write(render_table(H2, stat_rows))
    f.write("\n")

print(f"Đã lưu: {STATS_PATH}")
