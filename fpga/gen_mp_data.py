import numpy as np

# =====================================================================
# 1. Cấu hình tham số hệ thống (Phải khớp với file SystemVerilog)
# =====================================================================
M = 20        # Số hàng (số mẫu Photodiode)
NE = 200      # Số cột (Cửa sổ phơi sáng)
K = 7         # Số vòng lặp Matching Pursuit

# Hệ số Scale (Q-format) để chuyển số thực (float) thành số nguyên (integer) cho FPGA
# Ta dùng 15 bit cho phần thập phân.
SCALE_FACTOR = 2**15  

# =====================================================================
# 2. Tạo dữ liệu giả lập (Mock Data)
# =====================================================================
np.random.seed(42) # Cố định seed để kết quả các lần chạy giống nhau

# Tạo ma trận đo lường Theta (M x NE) ngẫu nhiên
Theta_float = np.random.randn(M, NE)

# QUAN TRỌNG: Chuẩn hóa các cột của Theta để độ dài bằng 1
# Điều này giúp loại bỏ phép chia trong FPGA (alpha = corr / 1 = corr)
for j in range(NE):
    norm = np.linalg.norm(Theta_float[:, j])
    if norm > 0:
        Theta_float[:, j] = Theta_float[:, j] / norm

# Tạo tín hiệu thưa gốc (chỉ có 2 thành phần tần số f_c1 và f_c2 có giá trị)
true_coef_float = np.zeros(NE)
true_coef_float[25] = 0.85  # Tần số 1
true_coef_float[150] = 0.60 # Tần số 2

# Tính vector Po (Tín hiệu Photodiode thu được)
# Po = Theta * true_coef
Po_float = np.dot(Theta_float, true_coef_float)

# =====================================================================
# 3. Chạy thuật toán Matching Pursuit (MP) trên Python để lấy "Đáp án"
# =====================================================================
r = Po_float.copy()
expected_coef_float = np.zeros(NE)

for i in range(K):
    # Tính tương quan: corr = Theta^T * r
    corr = np.dot(Theta_float.T, r)
    
    # Tìm index j* có giá trị tuyệt đối lớn nhất
    best_j = np.argmax(np.abs(corr))
    
    # Cập nhật hệ số: alpha = corr[best_j] (do mẫu số norm_sq đã chuẩn hóa = 1)
    alpha = corr[best_j]
    expected_coef_float[best_j] += alpha
    
    # Cập nhật phần dư r
    r = r - Theta_float[:, best_j] * alpha

# =====================================================================
# 4. Scale sang Số Nguyên (Fixed-Point) và xuất ra file .txt
# =====================================================================
# Ép kiểu int32 cho khớp với THETA_W = 32 và PO_W = 32 trong FPGA
Theta_int = np.round(Theta_float * SCALE_FACTOR).astype(np.int32)
Po_int = np.round(Po_float * SCALE_FACTOR).astype(np.int32)

# Riêng mảng Coef có thể bị phóng đại sau nhiều phép tính MAC, 
# ta cần scale bù trừ tương ứng (tùy thuộc vào thiết kế bit của bạn). 
# Ở đây ta giữ nguyên nhân 1 lần SCALE.
expected_coef_int = np.round(expected_coef_float * SCALE_FACTOR).astype(np.int32)

# Lưu ma trận Theta (lưu phẳng thành 1 cột dài M * NE phần tử)
np.savetxt("fpga/data/theta_matrix.txt", Theta_int.flatten(), fmt='%d')

# Lưu vector Po (M phần tử)
np.savetxt("fpga/data/po_vector.txt", Po_int, fmt='%d')

# Lưu đáp án Coef (NE phần tử)
np.savetxt("fpga/data/expected_coef.txt", expected_coef_int, fmt='%d')

print(f"Hoàn tất! Đã lưu {M*NE} phần tử Theta, {M} phần tử Po, và {NE} phần tử Coef vào thư mục 'fpga/data/'.")