import numpy as np

from simulate.config import *
from simulate.measurement import generate
from simulate.reconstruction import get_fourier_dict
from simulate.mp import *

# Sinh dữ liệu mẫu
_, S, A, Po, g = generate(SNR_DB)
D = get_fourier_dict(Ne)

# Chuyển đổi sang dạng integer
S_int, A_int, D_int, _ = quantize(S, A, D)
Po_int = quantize_po(Po)

# Tính ma trận Theta_int
Theta = A @ D
# Chuẩn hóa L2-norm của từng cột về 1
norms = np.linalg.norm(Theta, axis=0)
norms[norms == 0] = 1 # Tránh lỗi chia cho 0
Theta_norm = Theta / norms
Theta_int = np.round(Theta_norm * (1 << FRAC_D)).astype(np.int32)

# Lưu dữ liệu vào file
np.savetxt("fpga/data/d_matrix.txt", D_int.flatten(), fmt='%d')
np.savetxt("fpga/data/theta_matrix.txt", Theta_int.flatten(), fmt='%d')
np.savetxt("fpga/data/po_vector.txt", Po_int, fmt='%d')

print(f"XONG! Đã lưu 3 file nạp BRAM vào thư mục 'fpga/data/'")