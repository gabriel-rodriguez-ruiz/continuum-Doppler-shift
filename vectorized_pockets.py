# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:20:17 2026

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from get_pockets import integrate_brute_force_grand_potential
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y
import matplotlib.pyplot as plt

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0.08 #0.122 #0.08  #0.08  #  meV
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
cut_off = 5 * k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al   # 1.1*k_F_Al 
Delta_Al = 0.2   #0.2


phi_y = 0

T = False
beta = 50

N = 100
n_cores = 15
points = 3 * n_cores

# radius_values = np.linspace(0.98*k_F, 1.02*k_F, N)
# radius_values = [np.linspace(0.95*k_F, 0.98*k_F, N), np.linspace(0.98*k_F, 1.02*k_F, N), np.linspace(1.02*k_F, 1.05*k_F, N)]
# radius_values = [np.linspace(0.984*k_F, 0.994*k_F, N), np.linspace(0.994*k_F, 1.006*k_F, N), np.linspace(1.006*k_F, 1.016*k_F, N)]
k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
radius_values = [np.linspace(0.99*k_1, 1.01*k_1, N), np.linspace(1.01*k_1,
                                     0.99*k_2, N), np.linspace(0.99*k_2, 1.01*k_2, N)]
    
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

def get_Hamiltonian_in_polars(k, theta, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    """Return the Hamiltonian for a given k."""
    k_x = k * np.cos(theta)
    k_y = k * np.sin(theta)
    # I have putted an anysotropic mass
    chi_k_plus = gamma * ( (k_x + phi_x)**2 + (k_y + phi_y)**2) - mu
    chi_k_minus = gamma * ( (-k_x + phi_x)**2 + (-k_y + phi_y)**2 ) - mu
    return ( chi_k_plus[..., np.newaxis, np.newaxis] * np.kron( ( tau_0 + tau_z )/2, sigma_0)
                   - chi_k_minus[..., np.newaxis, np.newaxis]  * np.kron( ( tau_0 - tau_z )/2, sigma_0)  
                   - B_y * np.kron(tau_0, sigma_y)
                   - B_x * np.kron(tau_0, sigma_x)
                   - Delta * np.kron(tau_x, sigma_0)
                   + Lambda * (k_x + phi_x)[..., np.newaxis, np.newaxis]  * np.kron( ( tau_0 + tau_z )/2, sigma_y )
                   + Lambda * (-k_x + phi_x)[..., np.newaxis, np.newaxis]  * np.kron( ( tau_0 - tau_z )/2, sigma_y )
                   - Lambda * (k_y + phi_y)[..., np.newaxis, np.newaxis]  * np.kron( ( tau_0 + tau_z )/2, sigma_x )
                   - Lambda * (-k_y + phi_y)[..., np.newaxis, np.newaxis]  * np.kron( ( tau_0 - tau_z )/2, sigma_x )
                 )       

def get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    """Return the energies of the Hamiltonian at a given k."""
    H = get_Hamiltonian_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
    return np.linalg.eigvalsh(H)

def get_ground_state_energy(K, Theta, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    Z = get_Energies_in_polars(K, Theta, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
    mask = ( Z <= 0 )
    integrand = K.T * Z.T * mask.T
    q = np.sqrt(phi_x**2 + phi_y**2)
    angular_integral = scipy.integrate.trapezoid(integrand, theta_values,
                                                 axis=2)
    radial_integral = scipy.integrate.trapezoid(angular_integral, k_values,
                                                axis=1)
    return (1/2*np.sum(radial_integral)
            +  np.pi/2 * cut_off**2 * (2*gamma*q**2 - 2*mu + gamma*cut_off**2))

def get_ground_state_energy_vs_q_with_Doppler(K, Theta, mu, B_y, Delta, phi_x_values, gamma, Lambda, B_x, phi_y, q_D):
    E_q = np.zeros_like(phi_x_values)
    for i, q in enumerate(phi_x_values):
        q_x = q * np.cos(varphi) + q_D
        q_y = q * np.sin(varphi)
        E_q[i] = get_ground_state_energy(K, Theta, mu, B_y, Delta, q_x, gamma, Lambda, B_x, q_y)
    return E_q


#%% Pockets
cut_off = 2*k_F
N = 1000
B = 0*Delta
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
varphi = 0
k_values = np.append(np.linspace(0, 0.99*k_F, N), [np.linspace(0.99*k_F, 1.01*k_F, N),
            np.linspace(1.01*k_F, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, 1000)
q_c = Delta/(2*np.sqrt(mu*gamma))
q_values = np.linspace(0, 6*q_c, 100)
K, Theta = np.meshgrid(k_values, theta_values)

phi_x = 1.2*q_c
phi_y = 0
Z = get_Energies_in_polars(K, Theta, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

def get_k_plus(theta):
    k_plus = 1/np.sqrt(gamma) * np.sqrt(mu +
                    np.sqrt((2*gamma*k_F*phi_x*np.cos(theta))**2 + Delta**2))
    return k_plus

def get_k_minus(theta):
    k_minus = 1/np.sqrt(gamma) * np.sqrt(mu -
                    np.sqrt((2*gamma*k_F*phi_x*np.cos(theta))**2 + Delta**2))
    return k_minus


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

ax.contour(Theta, K/k_F, Z[:, :, 1], colors='black', levels=[0],
           linestyles="dashed")
ax.contour(Theta, K/k_F, Z[:, :, 3], colors='black', levels=[0])
ax.set_ylim([0.997, 1.003])
ax.set_yticks([0.998, 1, 1.002], [r"$0.998k_F$" , r"$k_F$", r"$1.002k_F$"])
ax.set_rlabel_position(0)

