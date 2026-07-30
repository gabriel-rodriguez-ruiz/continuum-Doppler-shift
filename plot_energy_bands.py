#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:24:58 2026

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from diagonalization import get_Energies_in_polars, get_Energies, get_Eigenvectors_in_polars
import scipy.optimize

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 #50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s  #np.sqrt(2*mu/m)*1e-9
mu_B = 5.788e-2 # meV/TT


Delta = 0.08# 0.08 #0.08  #2*0.122 # 0.08 #0.08   #  meVs
mu = E_F  # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 30  #15 #187*Delta/2 # meV*nm    # 8 * Delta  #0.644 meV 
theta = np.pi/2

B = 3*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
theta_values = np.array([0])
q_B_constant = 0#0.024/8
phi_x = 0#q_B_constant*2*Delta  #-1/2*B/gamma/k_F  #1e-3 * k_F #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s
Delta_Al = 0

k_values = np.linspace(0.9*k_F, 1.1*k_F, 300)
# k_values = np.linspace(0.5*k_F, 0.54*k_F, 100)
# k_values = np.linspace(1.9*k_F, 1.94*k_F, 100)

# k_values_Al = np.linspace(0.9*k_F_Al, 1.1*k_F_Al, 100)
# k_values_Al = np.linspace(0, 3*k_F_Al, 100)

# chi_k = gamma_Al*k_values_Al**2 - mu
# chi_k_plus = ( gamma_Al*(k_values_Al+phi_x)**2 - mu + ( gamma_Al*(-k_values_Al+phi_x)**2 - mu ) )/2
# chi_k_minus = ( gamma_Al*(k_values_Al+phi_x)**2 - mu - ( gamma_Al*(-k_values_Al+phi_x)**2 -mu ) ) / 2


E = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
# E = get_sorted_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

# E, V = get_Eigenvectors_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

# def eigh_interval(x):
#     """Emulate behavior of sparse diagonalization by computing levels closest to a point."""
#     e, psi = get_Eigenvectors_in_polars([x], theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
#     ind = np.argsort(abs(e[0][0]))
#     # return e[0][0][ind], psi[0][0][:, ind]
#     return e[0][0], psi[0][0]

    
# e, psi =  eigh_interval(k_values[0])
# sorted_levels = [e]
# for x in k_values[1:]:
#     e2, psi2 = eigh_interval(x)
#     Q = np.abs(psi.T.conj() @ psi2)  # Overlap matrix
#     assignment = scipy.optimize.linear_sum_assignment(-Q)[1]
#     sorted_levels.append(e2[assignment])
#     psi = psi2[:, assignment]

# E = get_Energies_in_polars(k_values_Al, theta_values, mu, B_y, Delta_Al, phi_x, gamma_Al, 0, B_x, phi_y)

fig, ax = plt.subplots()
# ax.plot(k_values_Al/k_F_Al, E[:, 0], marker="o", markersize=3)
# ax.plot(k_values/k_F, E[:, 0], marker="o", markersize=3)
ax.plot(k_values/k_F, E[:, 0], marker="v", markersize=3)

# ax.plot(k_values/k_F, E[:, 1], marker="o", markersize=3)

# ax.plot(k_values/k_F, sorted_levels, marker="o", markersize=3)

# ax.plot(k_values_Al/k_F_Al, chi_k_minus/2 - 1/2*np.sqrt(chi_k_plus**2 + Delta_Al**2))
# ax.plot(k_values/k_F, chi_k_minus/2 - 1/2*np.sqrt(chi_k_plus**2 + Delta_Al**2))

# chi_k_plus = ( gamma*(k_values+phi_x)**2 - mu + ( gamma*(-k_values+phi_x)**2 - mu ) )/2
# chi_k_minus = ( gamma*(k_values+phi_x)**2 - mu - ( gamma*(-k_values+phi_x)**2 -mu ) ) / 2
chi_k_minus = 2*gamma*k_values*np.cos(theta_values)*phi_x
chi_k_plus = gamma *( k_values**2 + phi_x**2 ) - mu

E_plus_plus = B + chi_k_minus + np.sqrt(chi_k_plus**2 + Delta**2)
E_minus_plus = -B + chi_k_minus + np.sqrt(chi_k_plus**2 + Delta**2) 
E_plus_minus = B + chi_k_minus - np.sqrt(chi_k_plus**2 + Delta**2) 
E_minus_minus = -B + chi_k_minus - np.sqrt(chi_k_plus**2 + Delta**2) 

ax.plot(k_values/k_F, E_plus_plus)
ax.plot(k_values/k_F, E_minus_plus)
ax.plot(k_values/k_F, E_plus_minus)
ax.plot(k_values/k_F, E_minus_minus)

ax.set_xlabel(r"$k_x/k_F$")
ax.set_ylabel(r"$E$")
plt.grid()

#%% In cartesian coordinates
k_x_values = np.linspace(-1.5*k_F, 1.5*k_F, 300)
k_y_values = np.zeros_like(k_x_values)
E_1 = get_Energies(k_x_values, k_y_values, mu, B, Delta, phi_x, gamma, Lambda)

fig, ax = plt.subplots()

ax.plot(k_x_values/k_F_Al, E_1[:, 0], marker="o", markersize=3)
plt.grid()

#%%
theta_values = np.array([np.pi/2])
E = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

fig, ax = plt.subplots()
ax.plot(k_values/k_F, E[:, 0], marker="o", markersize=3)
ax.set_xlabel(r"$k_y/k_F$")
ax.set_ylabel(r"$E$")
plt.grid()

#%% Angular dependence

fig, ax = plt.subplots()

theta_values = np.linspace(-np.pi/2, 3*np.pi/2, 100)
E = get_Energies_in_polars([0.965*k_F], theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
ax.plot(theta_values, E[0, :], marker="o", markersize=3)

plt.grid()