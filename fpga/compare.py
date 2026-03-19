import numpy as np

from simulate.config import *
from simulate.mp import reconstruct_int
from fpga.gen_mp_data import S

# Đọc coef từ FPGA
coef_fpga = np.loadtxt("../CS_FPGA/data/coef_output.txt", dtype=np.int32)
norms = np.loadtxt("../CS_FPGA/data/norms.txt", dtype=np.float64)
D_int = np.loadtxt("../CS_FPGA/data/d_matrix.txt", dtype=np.int64).reshape(Ne, Ne)

# Đưa coef_fpga về cùng thang đo với coef_int của Python
coef_int = np.round(coef_fpga / norms).astype(np.int64)

S_rec = reconstruct_int(D_int, coef_int)

rmse = np.sqrt(np.mean((S - S_rec)**2))

print(f"RMSE: {rmse}")