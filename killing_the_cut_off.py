#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:27:32 2026

@author: gabriel
"""

import matplotlib.pyplot as plt
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

Delta = 0  #0.08  #  meV
mu = 50.6   # 623 Delta #50.6  #  meV
Lambda = 0 * Delta # 8 * Delta  #0.644 meV 

m_Al = 2*m #1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

B = 0 #0.5 * 0.08  # meV
q_B = 0  #  0.024 * 0.1 * 0.08  # 1/nm
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
N_phi = 3  # it should be odd to include zero
cut_off = 1.1*k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al   # 1.1*k_F_Al 
Delta_Al = 0.1   #0.2

phi_y = 0

T = False
beta = 100

N = 100
n_cores = 19
points = 1 * n_cores
radius_values = np.linspace(0.99, 1.01, N)*k_F
radius_values_Al = np.linspace(0.99, 1.01, N)*k_F_Al


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B, "q_B": q_B,
              "B_x": B_x, "B_y": B_y, "theta":theta,
              "gamma_Al": gamma_Al, "k_F_Al":k_F_Al, "phi_y":phi_y,
              "T": T, "beta":beta,
              "cut_off_Al": cut_off_Al, "Delta_Al": Delta_Al
              }

def q_square_term(k, Delta, gamma, mu):
    E_k = np.sqrt((gamma*k**2 - mu)**2 + Delta**2)
    return 1/2 * gamma * (Delta**2) / E_k**2

fig, ax = plt.subplots()
k_values = np.linspace(0.9*k_F_Al, 1.1*k_F_Al, 1000)

for Delta_Al in [0, 0.1, 0.2]:
    ax.plot(k_values/k_F_Al, [q_square_term(k, Delta_Al, gamma_Al, mu) for k in k_values],
            label=r"$\Delta$=" + f"{Delta_Al}")
ax.legend()
ax.set_xlabel(r"$k/k_F$")
ax.set_ylabel(r"$\frac{1}{2}\gamma\frac{\Delta^2}{E_k^2}$")

fig, ax = plt.subplots()
k_values = np.linspace(0*k_F_Al, 20*1.1*k_F_Al, 1000)

for Delta_Al in [0, 0.1, 0.2]:
    ax.plot(k_values/k_F_Al, gamma_Al*k_values**2 - mu - np.sqrt( (gamma_Al*k_values**2 - mu)**2 + Delta_Al**2),
            label=r"$\Delta$=" + f"{Delta_Al}")
ax.legend()
ax.set_xlabel(r"$k/k_F$")
ax.set_ylabel(r"$\xi_k - E_k$")

#%%
fig, ax = plt.subplots()
k_values_Al = np.linspace(0.99*k_F_Al, 1.01*k_F_Al, 300)
k_values = np.linspace(0.99*k_F, 1.01*k_F, 1000)

# ax.plot(k_values_Al/k_F_Al, [q_square_term(k, Delta_Al, gamma_Al, mu) for k in k_values_Al], "o",
#         label=r"$\Delta$=" + f"{Delta_Al}")


# ax.plot(k_values_Al/k_F_Al, [q_square_term(k, Delta_Al, gamma_Al, mu) for k in k_values_Al],
#         label=r"$\Delta$=" + f"{Delta_Al}")

ax.plot(k_values_Al/k_F_Al, [gamma_Al/2 * (1 - (gamma_Al*(k**2 - k_F_Al**2)/Delta_Al)**2) for k in k_values_Al],
        label=r"$\Delta$=" + f"{Delta_Al}")

ax.plot(k_values/k_F, [gamma/2 * (1 - (gamma*(k**2 - k_F**2)/Delta)**2) for k in k_values],
        label=r"$\Delta$=" + f"{Delta}")

ax.set_xlabel("k/k_F")
#%%

def complete_integrand(k, Delta, gamma, mu, q):
    chi = lambda k: gamma*k**2 - mu
    return chi(k+q) - 1/2 * np.sqrt((chi(k+q) + chi(-k+q))**2 + 4*Delta**2)

fig, ax = plt.subplots()
k_values = np.linspace(0, 20.1*k_F_Al, 1000)

for Delta_Al in [0, 0.1, 0.2]:
    ax.plot(k_values/k_F_Al, [complete_integrand(k, Delta_Al, gamma_Al, mu, q=0.) for k in k_values],
            label=r"$\Delta$=" + f"{Delta_Al}")

ax.legend()
ax.set_xlabel(r"$k/k_F$")
ax.set_ylabel(r"$\xi(k+q) - \frac{1}{2}\sqrt{(\xi(k+q) + \xi(-k+q))^2 + 4\Delta^2}$")