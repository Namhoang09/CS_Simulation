import matplotlib.pyplot as plt
import numpy as np

from simulate.config import *
from simulate.measurement import generate
from simulate.reconstruction import get_fourier_dict
from simulate.mp import *

SNR_LIST = [None, 20, 40]
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

print(f"Khoảng cách thực tế: {actual_dist:.2f} m")

for i, snr in enumerate(SNR_LIST):
    t, S, A, Po, g = generate(snr)
    D = get_fourier_dict(Ne)
    _, _, D_INT, _ = quantize(S, A, D)   
    Po_INT = quantize_po(Po)

    time_us = t * 1e6

    rmse_list = []
    hypothetical_Nd = np.arange(Ne)
    distances = c * hypothetical_Nd * delta_T

    for Nd_t in hypothetical_Nd:
        A_t = np.zeros((M, Ne), dtype=np.int8)
        for m in range(M):
            A_t[m, :] = g[Nd_t + m * Nc: Nd_t + m * Nc + Ne]

        Theta_t = A_t.astype(np.int32) @ D_INT.astype(np.int32)
        norm_sq_t  = compute_norm_sq(Theta_t)
        coef_t = mp_integer(Theta_t, Po_INT, norm_sq_t, K_MP)
        S_rec_t = reconstruct_int(D_INT, coef_t)
        rmse_list.append(np.sqrt(np.mean((S - S_rec_t) ** 2)))

    min_rmse_idx = np.argmin(rmse_list)

    axs[i].plot(distances, rmse_list, color='black', linewidth=1)
    axs[i].set_ylabel('RMSE (arb. uni.)')

    if snr is None:
        axs[i].text(0.02, 0.05, f'({chr(97+i)}) SNR = {snr}', transform=axs[i].transAxes, fontsize=12)
        print(f"SNR = {snr} | MP ước lượng: {distances[min_rmse_idx]:.2f} m | RMSE: {rmse_list[min_rmse_idx]:.4f}")
    else:
        axs[i].text(0.02, 0.05, f'({chr(97+i)}) SNR = {snr} dB', transform=axs[i].transAxes, fontsize=12)
        print(f"SNR = {snr}dB | MP ước lượng: {distances[min_rmse_idx]:.2f} m | RMSE: {rmse_list[min_rmse_idx]:.4f}")

    axs[i].annotate('Actual position',
                    xy=(distances[min_rmse_idx], rmse_list[min_rmse_idx]),
                    xytext=(distances[min_rmse_idx] + 2, max(rmse_list) * 0.5),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9)

axs[-1].set_xlabel('Hypothetical position of the photodiode (m)')
fig.suptitle('RMSE calculated to determine the actual position with MP')
fig.tight_layout()
fig.savefig('figure/RMSE_SNR_MP.png', dpi=300)

plt.show()
