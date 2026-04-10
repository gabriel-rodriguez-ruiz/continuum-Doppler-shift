#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:28:24 2026

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from get_pockets import integrate_brute_force_grand_potential
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 100 * 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/TT


Delta = 0.08 #0.08   #  meVs
mu = E_F  # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 24*Delta # meV*nm    # 8 * Delta  #0.644 meV 
theta = np.pi/2

B = 3*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
q_B_constant = 0 #0.024/8
phi_x = 0 #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s
Delta_Al = 0

k_values = np.linspace(0.945*k_F, 0.965*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0])

k_values = np.linspace(1.035*k_F, 1.055*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0])

#%%

Lambda = 8*Delta

k_values = np.linspace(0.98*k_F, 1*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0], colors="red")

k_values = np.linspace(1*k_F, 1.02*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0], colors="red")

#%%

Lambda = 48*Delta

k_values = np.linspace(0.88*k_F, 0.94*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0], colors="green")

k_values = np.linspace(1.08*k_F, 1.1*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0], colors="green")



#%%

Lambda = 96*Delta

k_values = np.linspace(0.82*k_F, 0.86*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0], colors="blue")

k_values = np.linspace(1.17*k_F, 1.21*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0], colors="blue")

#%%

Lambda = 192*Delta

k_values = np.linspace(0.69*k_F, 0.72*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0], colors="orange")

k_values = np.linspace(1.4*k_F, 1.43*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0], colors="orange")


#%%

Lambda = 384*Delta

k_values = np.linspace(0.5*k_F, 0.54*k_F, 100)
theta_values = np.linspace(0, 2*np.pi, 100)

Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 2], levels=[0], colors="orange")

k_values = np.linspace(1.9*k_F, 1.94*k_F, 100)
Energies_polar = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

ax.contour(theta_values, k_values/k_F, Energies_polar[:, :, 1], levels=[0], colors="orange")





