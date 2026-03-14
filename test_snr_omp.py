import matplotlib.pyplot as plt
import numpy as np

from simulate.config import *
from simulate.measurement import generate
from simulate.evaluation import calculate

SNR_LIST = [None, 20, 40]
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

print(f"Khoảng cách thực tế: {actual_dist:.2f} m")

for i, snr in enumerate(SNR_LIST):
    t, S, _, Po, g = generate(snr)
    distances, rmse_list = calculate(g, S, Po)

    time_us = t * 1e6

    min_rmse_idx = np.argmin(rmse_list)

    axs[i].plot(distances, rmse_list, color='black', linewidth=1)
    axs[i].set_ylabel('RMSE (arb. uni.)')

    if snr is None:
        axs[i].text(0.02, 0.05, f'({chr(97+i)}) SNR = {snr}', transform=axs[i].transAxes, fontsize=12)
        print(f"SNR = {snr} | OMP ước lượng: {distances[min_rmse_idx]:.2f} m | RMSE: {rmse_list[min_rmse_idx]:.4f}")
    else:
        axs[i].text(0.02, 0.05, f'({chr(97+i)}) SNR = {snr} dB', transform=axs[i].transAxes, fontsize=12)
        print(f"SNR = {snr}dB | OMP ước lượng: {distances[min_rmse_idx]:.2f} m | RMSE: {rmse_list[min_rmse_idx]:.4f}")

    axs[i].annotate('Actual position',
                    xy=(distances[min_rmse_idx], rmse_list[min_rmse_idx]),
                    xytext=(distances[min_rmse_idx] + 2, max(rmse_list) * 0.5),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=9)
        
axs[-1].set_xlabel('Hypothetical position of the photodiode (m)')
fig.suptitle('RMSE calculated to determine the actual position with OMP')
fig.tight_layout()
fig.savefig('figure/RMSE_SNR_OMP.png', dpi=300)

plt.show()
