#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:44:38 2025

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from get_pockets import integrate_brute_force
from diagonalization import get_Energies_in_polars
from pathlib import Path
import multiprocessing
from pathlib import Path


c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0.08  #0.08  #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
Lambda = 8 * Delta # 8 * Delta  #0.644 meV 

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

B = 1.2 * Delta #0.5 * 0.08  # meV
q_B = 0.024/4 * B  # 1/nm
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
N_phi = 3  # it should be odd to include zero
cut_off = 1.1*k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al   # 1.1*k_F_Al 
Delta_Al = 0.2   #0.2

Aluminum_constant = 10000

phi_y = 0

T = False
beta = 150

N = 100
n_cores = 19
points = 1 * n_cores


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B, "q_B": q_B,
              "B_x": B_x, "B_y": B_y, "theta":theta,
              "gamma_Al": gamma_Al, "k_F_Al":k_F_Al, "phi_y":phi_y,
              "T": T, "beta":beta,
              "cut_off_Al": cut_off_Al, "Delta_Al": Delta_Al,
              "Aluminum_constant": Aluminum_constant
              }

def analytic_term(mu, phi_x, gamma, cut_off):
    return np.pi/2 * cut_off**2 * (2*gamma*phi_x**2 - 2*mu + gamma*cut_off**2)

#%% Calculate fundamental energy

def integrate(i):
    fundamental_energy = np.zeros(len(phi_y_values))
    phi_x = phi_x_values[i]
    for j, phi_y in enumerate(phi_y_values):
        print(j)
        integral, low_integral, high_integral = integrate_brute_force(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off, B_x, phi_y, T, beta)
        energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        fundamental_energy[j] = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*((phi_x + q_B)**2 + phi_y**2) - 2*mu + gamma*cut_off**2)
    return fundamental_energy

if __name__ == "__main__":
    phi_x_values = np.linspace(-0.003 * k_F, 0.003 * k_F, points)
    phi_y_values = np.linspace(-0.003 * k_F, 0.003 * k_F, points)
    indeces = np.arange(len(phi_x_values))
    with multiprocessing.Pool(n_cores) as pool:
        fundamental_energy = pool.map(integrate, indeces)
    fundamental_energy = np.stack(fundamental_energy)
    data_folder = Path("Data/")
    name = f"Fundamental_energy_contour_B_over_Delta={np.round(B/Delta,3)}_B_in_{np.round(theta,3)}_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_points={points}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, fundamental_energy=fundamental_energy, phi_x_values=phi_x_values,
             phi_y_values=phi_y_values,
             **parameters)
    print("\007")
    
#%%

data_folder = Path(r"./Data")
file_to_open = data_folder / "Fundamental_energy_contour_B_over_Delta=1.2_B_in_1.571_phi_x_in_(-0.003-0.003)_points=19.npz"

Data = np.load(file_to_open)
fundamental_energy = Data["fundamental_energy"]
phi_x_values = Data["phi_x_values"]
phi_y_values = Data["phi_y_values"]



X = phi_x_values
# Y = phi_y_values
Y = phi_y_values

X, Y = np.meshgrid(X, Y)        # The result of meshgrid is a coordinate grid
Z = fundamental_energy

fig, ax = plt.subplots()

# I need to tranpose X and Y because Z is matrix ordered (like an image)
CS = ax.contour(X.T, Y.T, Z, origin="upper", levels=50)


ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$\phi_y$")
ax.set_aspect('equal')

# plt.tight_layout()
# ax.imshow(Z)


# ax.clabel(CS, fontsize=10)