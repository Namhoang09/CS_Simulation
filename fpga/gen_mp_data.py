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

# Hàm xuất dữ liệu ra file hex
def export_hex(array, filename, width=32):
    mask = (1 << width) - 1
    with open(filename, 'w') as f:
        for val in array.flatten():
            f.write(f"{int(val) & mask:08x}\n")

# Lưu dữ liệu vào file
export_hex(Theta_int.flatten(), "../CS_FPGA/data/theta_matrix.txt")
export_hex(Po_int.flatten(),    "../CS_FPGA/data/po_vector.txt")

print(f"XONG! Đã lưu 2 file nạp BRAM vào thư mục 'CS_FPGA/data/'")