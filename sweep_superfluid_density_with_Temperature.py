# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:49:29 2026

@author: Gabriel
"""

import numpy as np
# Import your computing pipeline 
from pathlib import Path
from get_vectorized_superfluid_density import get_vectorized_superfluid_density

n_cores = 15
points = 1 * n_cores

Delta = 0.08  # meV

# MANDATORY: Protected block for your main pipeline
if __name__ == '__main__':
    # Sample Test Inputs
    B_values = np.linspace(0 * Delta, 3 * Delta, points)
    beta_values = np.linspace(30, 100, 36)
    superfluid_density_xx = np.zeros((len(B_values), len(beta_values)))
    superfluid_density_yy = np.zeros((len(B_values), len(beta_values)))
    # Run execution safely
    for i, beta in enumerate(beta_values):
        superfluid_density_xx[:, i], superfluid_density_yy[:, i] = get_vectorized_superfluid_density(B_values, beta, n_cores)
        print(i)
    data_folder = Path("Data/")
    data_folder.mkdir(exist_ok=True)
    
    name = f"Zeeman_sweep_in_Temperature_with_{points}_values_of_beta_in_({np.min(beta_values)}-{np.max(beta_values)})"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             B_values=B_values, beta_values=beta_values)
    print("Done! \007")
