# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:05:13 2026

@author: Gabriel
"""

import numpy as np
# Import your computing pipeline 
from pathlib import Path
from get_vectorized_superfluid_density import get_vectorized_superfluid_density, get_vectorized_superfluid_density_Zeeman_and_Doppler

n_cores = 15
points = 2 * n_cores

Delta = 0.08  # meV

# MANDATORY: Protected block for your main pipeline
if __name__ == '__main__':
    # Sample Test Inputs
    B_values = np.linspace(0, 0.5*Delta, points) # Tesla
    beta_values = np.linspace(56, 70, 8)  # np.linspace(30, 100, 36)
    q_B_values = np.linspace(0.015, 0.035, 21)
    superfluid_density_xx = np.zeros((len(B_values), len(beta_values), len(q_B_values)))
    superfluid_density_yy = np.zeros((len(B_values), len(beta_values), len(q_B_values)))
    # Run execution safely
    for i, beta in enumerate(beta_values):
        for j, q_B_value in enumerate(q_B_values):
            superfluid_density_xx[:, i, j], superfluid_density_yy[:, i, j] = get_vectorized_superfluid_density_Zeeman_and_Doppler(B_values, beta, q_B_value, n_cores)
        print(i)
    data_folder = Path("Data/")
    data_folder.mkdir(exist_ok=True)
    
    name = f"Zeeman_and_Doppler_sweep_in_Temperature_with_{points}_values_of_beta_in_({np.min(beta_values)}-{np.max(beta_values)})_and_with_q_B_values_in_({np.min(q_B_values)}-{np.max(q_B_values)})"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             B_values=B_values, beta_values=beta_values,
             q_B_values=q_B_values)
    print("Done! \007")