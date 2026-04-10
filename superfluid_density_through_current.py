#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:07:49 2025

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from get_pockets import plot_interpolated_contours, integrate_pocket, get_pockets_contour, integrate_brute_force, integrate_Romberg, integrate_brute_force_current_x, integrate_brute_force_current_y
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from skopt import gp_minimize
from skopt.space import Space

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/T

Delta = 0.08   #  meVs
mu = 50.6   # 623 Delta #50.6  #  meV
Lambda = 16*Delta # meV*nm    # 8 * Delta  #0.644 meV 

N_phi = 3  #101  # 101  # it should be odd to include zero
h = 1e-5*k_F

phi_x_values = np.array([-h, 0, h])     #np.linspace(-0.002 * k_F, 0.002 * k_F, N_phi)   #np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)
cut_off = 1.1*k_F # 1.1 k_F

theta = np.pi/2 #np.pi/2   # float

N = 300 #300 #100  #514   #300
n_cores = 19
points = 1* n_cores
N_polifit = 2  # 4
C = 0

T = True
beta = 150


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "C": C, "T": T, "beta": beta
              }

def integrate_B(B):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    current_phi = np.zeros_like(phi_x_values)

    for j, phi_x in enumerate(phi_x_values):
        print(j)
        integral, low_integral, high_integral = integrate_brute_force_current_x(N, mu, B_y, Delta, phi_x, gamma, Lambda, k_F, cut_off, B_x, phi_y=0, T=T, beta=beta, h=h)    # phi_y=-phi_x if magnetic field is at 45º
        current_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    total_current = current_phi + 2*np.pi * cut_off**2 * gamma*phi_x_values
    
    superfluid_density_finite_differences_0 = (total_current[2] - total_current[0])/(2*h)
        
    phi_y_values = np.array([-h, 0, h])
    current_phi_0 = np.zeros_like(phi_y_values)

    for j, phi_y in enumerate(phi_y_values):
        print(3+j)
        integral, low_integral, high_integral = integrate_brute_force_current_y(N, mu, B_y, Delta, 0, gamma, Lambda, k_F, cut_off, B_x, phi_y, T=T, beta=beta, h=h)
        current_phi_0[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    total_current_0 = current_phi_0 + 2*np.pi * cut_off**2 * gamma*phi_y_values 

    superfluid_density_yy_0 = (total_current_0[2]-total_current_0[0])/(2*h)

    return superfluid_density_finite_differences_0, superfluid_density_yy_0

if __name__ == "__main__":
    B_values = np.linspace(0.*Delta, 2.5*Delta, points)
    integrate = integrate_B
    B_direction = f"{theta:.2}"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_finite_differences_0, superfluid_density_yy_0 = zip(*pool.map(integrate, B_values))
    superfluid_density_finite_differences_0 = np.array(superfluid_density_finite_differences_0)
    superfluid_density_yy_0 = np.array(superfluid_density_yy_0)
    data_folder = Path("Data/")
    name = f"superfluid_density_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}_C={C}_T={T}_beta={beta}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open,
             superfluid_density_finite_differences_0=superfluid_density_finite_differences_0,
             superfluid_density_yy_0=superfluid_density_yy_0,
             B_values=B_values, **parameters)
    print("\007")

