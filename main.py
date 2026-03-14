import matplotlib.pyplot as plt
import numpy as np

from simulate.config import *
from simulate.measurement import generate
from simulate.reconstruction import reconstruct
from simulate.evaluation import calculate

def run_simulation():
    # Tạo tín hiệu gốc
    t, S, A, Po, g = generate(SNR_DB)

    # Đánh giá RMSE
    distances, rmse_list = calculate(g, S, Po)
    time_us = t * 1e6  # Trục thời gian (µs)

    # --- Tái tạo sóng mang ---
    M_list = [5, 10, 15, 20]
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    for i, m_val in enumerate(M_list):
        A_m = A[:m_val, :]
        Po_m = Po[:m_val]

        S_rec_m = reconstruct(A_m, Po_m)

        axs[i].plot(time_us, S_rec_m, color='black', linewidth=0.8)
        axs[i].set_ylabel('Intensity (arb. uni.)')
        axs[i].text(0.02, 0.8, f'({chr(97+i)}) M={m_val}', transform=axs[i].transAxes, fontsize=12)

    axs[-1].set_xlabel('Time (µs)')
    fig.suptitle('Waveform reconstructed by CS technique')
    fig.tight_layout()
    fig.savefig('figure/Reconstruct.png', dpi=300)

    # --- Tính RMSE ---
    plt.figure(figsize=(8, 4))
    plt.plot(distances, rmse_list, color='black', linewidth=1)
    plt.title('RMSE calculated to determine the actual position')
    plt.xlabel('Hypothetical position of the photodiode (m)')
    plt.ylabel('RMSE (arb. uni.)')

    min_rmse_idx = np.argmin(rmse_list)
    plt.annotate('Actual position of\nthe photodiode', 
                 xy=(distances[min_rmse_idx], rmse_list[min_rmse_idx]), 
                 xytext=(distances[min_rmse_idx] + 2, max(rmse_list)/6),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.tight_layout()
    plt.savefig('figure/RMSE.png', dpi=300)

    print(f"Quãng đường quang học L thực tế: {actual_dist:.2f} m")
    print(f"Khoảng cách ước lượng tốt nhất: {distances[min_rmse_idx]:.2f} m")

    # --- So sánh Nd ---
    Nd_tests = [Nd-2, Nd-1, Nd]
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    for i, Nd_t in enumerate(Nd_tests):
        # Tạo ma trận A giả định với Nd đang test
        A_hypo = np.zeros((M, Ne))

        for m in range(M):
            start = Nd_t + m * Nc
            A_hypo[m, :] = g[start : start + Ne]

        # Tái tạo lại sóng với đúng giả định Nd đang test
        S_rec_test = reconstruct(A_hypo, Po)
        
        # Vẽ tín hiệu được tái tạo
        axs[i].plot(time_us, S_rec_test, color='black', linewidth=1)
        
        axs[i].set_ylabel('Intensity\n(arb. uni.)')
        
        # Chèn nhãn (a), (b), (c) như trong bài báo
        axs[i].text(0.02, 0.8, f'({chr(97+i)}) Nd_test={Nd_t}', transform=axs[i].transAxes, fontsize=12)

    axs[-1].set_xlabel('Time (µs)')
    fig.suptitle('Waveforms evaluated at different Nd values')
    fig.tight_layout()
    fig.savefig('figure/Waveforms.png', dpi=300)

    plt.show()

if __name__ == "__main__":
    run_simulation()