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
from get_pockets import integrate_brute_force_grand_potential
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0 #0.122 #0.08  #0.08  #  meV
mu = E_F   # 623 Delta #50.6  #  meV
Lambda = 0 # 15 # 8 * Delta  #0.644 meV 

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

B = 0 * Delta #0.5 * 0.08  # meV
q_B = 0 #1e-4*k_F #0.024/8 * B  # 1/nm
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
N_phi = 3  # it should be odd to include zero
cut_off = 10 * k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al   # 1.1*k_F_Al 
Delta_Al = 0.2   #0.2


phi_y = 0

T = False
beta = 150

N = 50
n_cores = 19
points = 1 * n_cores

# radius_values = np.linspace(0.98*k_F, 1.02*k_F, N)
# radius_values = [np.linspace(0.95*k_F, 0.98*k_F, N), np.linspace(0.98*k_F, 1.02*k_F, N), np.linspace(1.02*k_F, 1.05*k_F, N)]
# radius_values = [np.linspace(0.984*k_F, 0.994*k_F, N), np.linspace(0.994*k_F, 1.006*k_F, N), np.linspace(1.006*k_F, 1.016*k_F, N)]
radius_values = [np.linspace(0.955*k_F, 0.98*k_F, N), np.linspace(0.98*k_F, 1.02*k_F, N), np.linspace(1.02*k_F, 1.05*k_F, N)]

    
parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B, "q_B": q_B,
              "B_x": B_x, "B_y": B_y, "theta":theta,
              "gamma_Al": gamma_Al, "k_F_Al":k_F_Al, "phi_y":phi_y,
              "T": T, "beta":beta,
              "cut_off_Al": cut_off_Al, "Delta_Al": Delta_Al,
              "radius_values": radius_values
              }

def integrate_phi_x(phi_x):
    integral, low_integral, high_integral, normal_density = integrate_brute_force_grand_potential(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off,
                                                                  B_x, 0, T, beta, radius_values)
    energy_phi_2DEG = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    # fundamental_energy_2DEG = energy_phi_2DEG + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2+q_B**2) - 2*mu + gamma*cut_off**2)
    fundamental_energy_2DEG = energy_phi_2DEG +  np.pi/2 * cut_off**2 * (2*gamma*(phi_x+q_B)**2 - 2*mu + gamma*cut_off**2)
    # print(np.sum(integral))
    # integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, Delta_Al, phi_x, gamma_Al, 0, k_F_Al, cut_off_Al,
    #                                                             B_x, 0, T, beta)
    # energy_phi_Al = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    # # fundamental_energy_Al = energy_phi_Al + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)
    # fundamental_energy_Al = energy_phi_Al + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)
    
    # integral, low_integral, high_integral = integrate_quad_grand_potential(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off,
    #                                                               B_x, 0, T, beta)
    # energy_phi_2DEG = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    # # fundamental_energy_2DEG = energy_phi_2DEG + np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2+q_B**2) - 2*mu + gamma*cut_off**2)
    # fundamental_energy_2DEG = energy_phi_2DEG +  np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2+q_B**2) - 2*mu + gamma*cut_off**2)
    # # print(np.sum(integral))
    # integral, low_integral, high_integral = integrate_quad_grand_potential(N, mu, B_y, Delta_Al, phi_x, gamma_Al, 0, k_F_Al, cut_off_Al,
    #                                                             B_x, 0, T, beta)
    # energy_phi_Al = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    # # fundamental_energy_Al = energy_phi_Al + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)
    # fundamental_energy_Al = energy_phi_Al + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)

    
    return fundamental_energy_2DEG, normal_density

def integrate_phi_y(phi_y):
    integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, Delta, -0.0005, gamma, Lambda, k_F, cut_off,
                                                                  B_x, phi_y, T, beta, radius_values)
    energy_phi_2DEG = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy_2DEG = energy_phi_2DEG  + np.pi/2 * cut_off**2 * (2*gamma*phi_y**2 - 2*mu + gamma*cut_off**2)

    return fundamental_energy_2DEG


if __name__ == "__main__":
    phi_x_values = np.linspace(-1e-4, 1e-4, points)  #phi_x_values = np.linspace(-0.0015, 0, points)
    integrate = integrate_phi_x   # integrate_phi_x
    with multiprocessing.Pool(n_cores) as pool:
        fundamental_energy_2DEG, normal_density = zip(*pool.map(integrate, phi_x_values))
    # fundamental_energy = np.array(fundamental_energy)
    fundamental_energy_2DEG = np.array(fundamental_energy_2DEG)
    normal_density = np.array(normal_density)
    # fundamental_energy_Al = np.array(fundamental_energy_Al)
    data_folder = Path("Data/")
    name = f"total_fundamental_energy_B={B}_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}_T={T}_beta={beta}.npz"
    # name = f"circle_area_cut_off_Al={cut_off_Al}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, #fundamental_energy=fundamental_energy,
             fundamental_energy_2DEG=fundamental_energy_2DEG,
             normal_density=normal_density,
             # fundamental_energy_Al=fundamental_energy_Al,
             phi_x_values=phi_x_values, **parameters)
    print("\007")
