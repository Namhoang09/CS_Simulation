import numpy as np

from simulate.config import *
from simulate.measurement import generate
from simulate.reconstruction import get_fourier_dict
from simulate.mp import *

# Sinh dữ liệu mẫu
_, S, A, Po, _ = generate(SNR_DB)
D = get_fourier_dict(Ne)

# Hàm xuất dữ liệu ra file hex
def export_hex(array, filename, width=32):
    mask = (1 << width) - 1
    with open(filename, 'w') as f:
        for val in array.flatten():
            f.write(f"{int(val) & mask:08x}\n")

# Hàm main
def fpga():
    # Chuyển đổi sang dạng integer
    _, _, D_int, _ = quantize(S, A, D)
    Po_int = quantize_po(Po)

    # Tính ma trận Theta_int
    Theta = A @ D
    # Chuẩn hóa L2-norm của từng cột về 1
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1 # Tránh lỗi chia cho 0
    Theta_norm = Theta / norms
    Theta_int = np.round(Theta_norm * (1 << FRAC_D)).astype(np.int32)

    # Lưu dữ liệu vào file
    export_hex(Theta_int.flatten('F'), "../CS_FPGA/data/theta_matrix.txt")
    export_hex(Po_int,    "../CS_FPGA/data/po_vector.txt")
    np.savetxt("../CS_FPGA/data/norms.txt", norms, fmt='%f')
    np.savetxt("../CS_FPGA/data/d_matrix.txt", D_int.flatten(), fmt='%d')

    print(f"XONG! Đã lưu file nạp BRAM vào thư mục 'CS_FPGA/data/'")

if __name__ == "__main__":
    fpga()
