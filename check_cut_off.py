#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:36:05 2026

@author: gabriel
"""

import numpy as np
from get_pockets import integrate_brute_force_grand_potential

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 0.5 #50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0.08  #  meV
mu = E_F   # 623 Delta #50.6  #  meV
Lambda = 16 * Delta # 8 * Delta  #0.644 meV 

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

B = 3*Delta #0.5 * 0.08  # meV
q_B = 0 # 0.024 * 0.05 * 0.08  # 1/nm
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
N_phi = 3  # it should be odd to include zero
cut_off = 5*k_F # 1.1 k_F
cut_off_Al = 5*k_F_Al # 1.1 k_F
Delta_Al = 0.2

phi_x = 0
phi_y = 0

T = False
beta = 100

N = 100
n_cores = 19
points = 1 * n_cores
radius_values = radius_values = [np.linspace(0.55*k_F, 0.95*k_F, N), np.linspace(0.95*k_F, 1*k_F, N), np.linspace(1.*k_F, 1.45*k_F, N)]
radius_values_Al = np.linspace(0.99, 1.01, N)*k_F_Al


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B, "q_B": q_B,
              "B_x": B_x, "B_y": B_y, "theta":theta,
              "gamma_Al": gamma_Al, "k_F_Al":k_F_Al, "phi_y":phi_y,
              "T": T, "beta":beta,
              "cut_off_Al": cut_off_Al
              }


integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, Delta_Al, phi_x, gamma_Al, 0, k_F_Al, cut_off_Al,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_Al = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)

integral_without_gap, low_integral_without_gap, high_integral_without_gap = integrate_brute_force_grand_potential(N, mu, B_y, 0, 0, gamma_Al, 0, k_F_Al, cut_off_Al,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_Al_without_gap_and_phi_x = np.sum(integral_without_gap) + np.sum(low_integral_without_gap) + np.sum(high_integral_without_gap)

integral_without_phi, low_integral_without_phi, high_integral_without_phi = integrate_brute_force_grand_potential(N, mu, B_y, 0.2, 0, gamma_Al, 0, k_F_Al, cut_off_Al,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_Al_without_phi_x = np.sum(integral_without_phi) + np.sum(low_integral_without_phi) + np.sum(high_integral_without_phi)

integral_without_gap, low_integral_without_gap, high_integral_without_gap = integrate_brute_force_grand_potential(N, mu, B_y, 0, phi_x, gamma_Al, 0, k_F_Al, cut_off_Al,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_Al_without_gap_but_phi_x = np.sum(integral_without_phi) + np.sum(low_integral_without_phi) + np.sum(high_integral_without_phi) 



fundamental_energy_Al = energy_phi_Al + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)

cut_off_term = np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)   # np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)

difference_Al = energy_phi_Al_without_gap_and_phi_x + cut_off_term - 2*np.pi*k_F_Al**2*(gamma_Al*k_F_Al**2/2 - mu)# it should tend to zero
difference_due_to_gap_and_phi_x = energy_phi_Al - energy_phi_Al_without_gap_and_phi_x
difference_due_to_phi = energy_phi_Al_without_gap_but_phi_x + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2) - energy_phi_Al_without_gap_and_phi_x
relative_effect_of_phi = (np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2) ) / difference_due_to_phi

#%% 2DEG

integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_2DEG = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
fundamental_energy_2DEG = energy_phi_2DEG + np.pi/2 * cut_off**2 * (- 2*mu + gamma*cut_off**2)

integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, 0, 0, gamma, Lambda, k_F, cut_off,
                                                              B_x, 0, T, beta, radius_values)
energy_phi_2DEG_without_gap_and_phi_x = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)

cut_off_term = np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2+q_B**2) - 2*mu + gamma*cut_off**2)   # np.pi/2 * cut_off**2 * (2*gamma*(phi_x**2+q_B**2) - 2*mu + gamma*cut_off**2)

difference_2DEG = energy_phi_2DEG_without_gap_and_phi_x + cut_off_term - 2*np.pi*k_F**2*(gamma*k_F**2/2 - mu) # it should tend to zero
difference_due_to_gap_and_phi_x = energy_phi_2DEG - energy_phi_2DEG_without_gap_and_phi_x