# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 08:53:56 2026

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

c = 3e17 # nm/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0.08  #  meV
mu = E_F   #  meV
Lambda = 15

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

q_B_constant = 0.024/8
theta = np.pi/2
N_phi = 3  
cut_off = 2 * k_F 

phi_y = 0
h = 1e-5
T = False
beta = 50

N = 1000
n_cores = 15
points = 4 * n_cores

k_values = np.append(np.linspace(0, 0.99*k_F, N), [np.linspace(0.99*k_F, 1.01*k_F, N),
            np.linspace(1.01*k_F, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, N)
q_c = Delta/(2*np.sqrt(mu*gamma))
q_values = np.sort(np.append(np.linspace(-3*q_c, 3*q_c, 10),[0]))

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off,
              "phi_y":phi_y,
              "T": T, "beta":beta, "k_values": k_values,
              "theta_values": theta_values}

# Pre-compute Kronecker products globally once to save CPU cycles
KRON_PLUS  = np.kron((tau_0 + tau_z) / 2, sigma_0)
KRON_MINUS = np.kron((tau_0 - tau_z) / 2, sigma_0)
KRON_Y     = np.kron(tau_0, sigma_y)
KRON_X     = np.kron(tau_0, sigma_x)
KRON_X_0   = np.kron(tau_x, sigma_0)
KRON_PLUS_Y  = np.kron((tau_0 + tau_z) / 2, sigma_y)
KRON_MINUS_Y = np.kron((tau_0 - tau_z) / 2, sigma_y)
KRON_PLUS_X  = np.kron((tau_0 + tau_z) / 2, sigma_x)
KRON_MINUS_X = np.kron((tau_0 - tau_z) / 2, sigma_x)

# High-performance constants independent of k and theta

def get_Hamiltonian_for_single_theta(k_1d, theta_val, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    """
    Computes Hamiltonian for ALL k values, but for only ONE single theta value.
    Array size drops from 732 MiB to 0.73 MiB!
    """
    k_x = k_1d * np.cos(theta_val)
    k_y = k_1d * np.sin(theta_val)

    chi_k_plus = gamma * ( (k_x + phi_x)**2 + (k_y + phi_y)**2) - mu
    chi_k_minus = gamma * ( (-k_x + phi_x)**2 + (-k_y + phi_y)**2 ) - mu
    CONSTANT_TERMS = - B_y * KRON_Y - B_x * KRON_X - Delta * KRON_X_0

    # Shape: (3000, 4, 4)
    H = ( chi_k_plus[..., np.newaxis, np.newaxis] * KRON_PLUS
        - chi_k_minus[..., np.newaxis, np.newaxis] * KRON_MINUS
        + CONSTANT_TERMS )
                   
    if Lambda != 0:
        H += ( Lambda * (k_x + phi_x)[..., np.newaxis, np.newaxis] * KRON_PLUS_Y
             + Lambda * (-k_x + phi_x)[..., np.newaxis, np.newaxis] * KRON_MINUS_Y
             - Lambda * (k_y + phi_y)[..., np.newaxis, np.newaxis] * KRON_PLUS_X
             - Lambda * (-k_y + phi_y)[..., np.newaxis, np.newaxis] * KRON_MINUS_X )
    return H

def get_ground_state_energy(k_1d, theta_1d, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y):
    """
    Computes the total angular and radial integral step-by-step 
    without storing the massive multi-dimensional grid.
    """
    angular_integrand_points = []
    
    # Loop over theta explicitly. Since N=1000, this Python loop is light
    # and keeps RAM usage bound to a maximum of 1 MiB per process.
    for t_val in theta_1d:
        H_slice = get_Hamiltonian_for_single_theta(k_1d, t_val, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)
        Z_slice = np.linalg.eigvalsh(H_slice) # Shape: (3000, 4)
        
        # Isolate negative energies
        mask = (Z_slice <= 0)
        
        # Sum over energy bands (axis 1) and scale by the cylindrical Jacobian component (k)
        # Shape results in a 1D array of 3000 elements for this specific theta
        k_scaled_energy = np.sum(Z_slice * mask, axis=1) * k_1d
        
        # Integrate across the radial k dimension immediately for this theta profile
        radial_integral_for_theta = scipy.integrate.trapezoid(k_scaled_energy, k_1d)
        angular_integrand_points.append(radial_integral_for_theta)
        
    # Convert points back to an array and integrate across the angular theta dimension
    angular_integrand_points = np.array(angular_integrand_points)
    total_integral = scipy.integrate.trapezoid(angular_integrand_points, theta_1d)
    
    q = np.sqrt(phi_x**2 + phi_y**2)
    return (1/2 * total_integral
            + np.pi/2 * cut_off**2 * (2*gamma*q**2 - 2*mu + gamma*cut_off**2))

def get_superfluid_density(k_1d, theta_1d, mu, B_y, Delta, q, gamma, Lambda, B_x, phi_y, q_D_x, h):
    E_q_0 = get_ground_state_energy(k_1d, theta_1d, mu, B_y, Delta, q_D_x, gamma, Lambda, B_x, 0)
    varphi = 0
    E_q_x = np.zeros(2)
    for i, dq in enumerate([-h, h]):
        q_x = dq * np.cos(varphi) + q_D_x
        q_y = dq * np.sin(varphi)
        E_q_x[i] = get_ground_state_energy(k_1d, theta_1d, mu, B_y, Delta, q_x, gamma, Lambda, B_x, q_y)
    n_xx = ( E_q_x[1] -2*E_q_0 + E_q_x[0] )/h**2
    varphi = np.pi/2
    E_q_y = np.zeros(2)
    for i, dq in enumerate([-h, h]):
        q_x = dq * np.cos(varphi) + q_D_x
        q_y = dq * np.sin(varphi)
        E_q_y[i] = get_ground_state_energy(k_1d, theta_1d, mu, B_y, Delta, q_x, gamma, Lambda, B_x, q_y)
    n_yy = ( E_q_y[1] -2*E_q_0 + E_q_y[0] )/h**2
    return n_xx, n_yy    

def integrate(B):
    B_x =  B * np.cos(theta)
    B_y =  B * np.sin(theta)
    q_D_x = q_B_constant * B
    return get_superfluid_density(k_values, theta_values, mu, B_y, Delta, 0, gamma, Lambda, B_x, 0, q_D_x, h)

if __name__ == "__main__":
    B_values = np.linspace(0*Delta, 3*Delta, points)
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_xx, superfluid_density_yy = zip(*pool.map(integrate, B_values))
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    data_folder = Path("Data/")
    data_folder.mkdir(exist_ok=True)
    
    name = f"superfluid_density_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}_T={T}_beta={beta}_q_B_constant={q_B_constant}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             B_values=B_values, **parameters)
    print("Done! \007")
