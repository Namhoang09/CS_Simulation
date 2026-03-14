import numpy as np

from simulate.config import *
from simulate.reconstruction import reconstruct

def calculate(g, S, Po):
    rmse_list = []
    hypothetical_Nd = np.arange(Ne)
    
    # Tính tổng quãng đường quang học L = c * Nd * delta_T
    distances = c * hypothetical_Nd * delta_T

    for Nd_t in hypothetical_Nd:
        A_test = np.zeros((M, Ne))

        for m in range(M):
            start = Nd_t + m*Nc
            A_test[m,:] = g[start : start + Ne]
        
        # Chạy CS reconstruct cho mỗi giả định Nd
        S_rec = reconstruct(A_test, Po)
        
        # Tính RMSE
        rmse = np.sqrt(np.mean((S - S_rec)**2))
        rmse_list.append(rmse)
        
    return distances, rmse_list