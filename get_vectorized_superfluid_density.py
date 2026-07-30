# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:24:06 2026

@author: Gabriel
"""

import numpy as np
import multiprocessing
import scipy
from functools import partial

# Keep ONLY pure physical definitions at the top level
c = 3e17 
m_e = 5.1e8 / c**2 
m = 0.0403 * m_e 
hbar = 6.58e-13 
gamma = hbar**2 / (2*m) 
E_F = 50.6 
k_F = np.sqrt(E_F / gamma) 
Delta = 0.08  
mu = E_F   
Lambda = 15 
cut_off = 2 * k_F 
h = 1e-5
theta = np.pi/2
mu_B = 5.79e-2  # meV/T

# Pre-compute structures outside functions for performance (these never change)
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y
KRON_PLUS  = np.kron((tau_0 + tau_z) / 2, sigma_0)
KRON_MINUS = np.kron((tau_0 - tau_z) / 2, sigma_0)
KRON_Y     = np.kron(tau_0, sigma_y)
KRON_X     = np.kron(tau_0, sigma_x)
KRON_X_0   = np.kron(tau_x, sigma_0)
KRON_PLUS_Y  = np.kron((tau_0 + tau_z) / 2, sigma_y)
KRON_MINUS_Y = np.kron((tau_0 - tau_z) / 2, sigma_y)
KRON_PLUS_X  = np.kron((tau_0 + tau_z) / 2, sigma_x)
KRON_MINUS_X = np.kron((tau_0 - tau_z) / 2, sigma_x)

def get_Hamiltonian_tensor(k_1d, theta_val, mu, B_y_vec, Delta, phi_x_tensor, gamma, Lambda, B_x_vec, phi_y_tensor):
    k_x = k_1d * np.cos(theta_val)
    k_y = k_1d * np.sin(theta_val)

    px = phi_x_tensor[:, :, np.newaxis]
    py = phi_y_tensor[:, :, np.newaxis]
    kx = k_x[np.newaxis, np.newaxis, :]
    ky = k_y[np.newaxis, np.newaxis, :]

    chi_k_plus = gamma * ((kx + px)**2 + (ky + py)**2) - mu
    chi_k_minus = gamma * ((-kx + px)**2 + (-ky + py)**2) - mu

    bx = B_x_vec[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    by = B_y_vec[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    constant_terms = - by * KRON_Y - bx * KRON_X - Delta * KRON_X_0

    H = (chi_k_plus[..., np.newaxis, np.newaxis] * KRON_PLUS
         - chi_k_minus[..., np.newaxis, np.newaxis] * KRON_MINUS
         + constant_terms)
                   
    if Lambda != 0:
        H += (Lambda * (kx + px)[..., np.newaxis, np.newaxis] * KRON_PLUS_Y
              + Lambda * (-kx + px)[..., np.newaxis, np.newaxis] * KRON_MINUS_Y
              - Lambda * (ky + py)[..., np.newaxis, np.newaxis] * KRON_PLUS_X
              - Lambda * (-ky + py)[..., np.newaxis, np.newaxis] * KRON_MINUS_X)
    return H

# --- THE WORKER TASK ---
# Rule: It accepts the variable item first, and the configuration dict second
def process_B_chunk(B_chunk, params):
    """Processes a sub-batch of B values without accessing global script scope.
        B_chunk is the Zeeman energy"""
    N_B_chunk = len(B_chunk)
    
    # Read variables directly from the frozen params package
    Zeeman = params["Zeeman"]
    q_B_constant = params["q_B_constant"]
    T = params["T"]
    beta = params["beta"]
    k_values = params["k_values"]
    theta_values = params["theta_values"]

    B_x_vec = B_chunk * np.cos(theta) * Zeeman
    B_y_vec = B_chunk * np.sin(theta) * Zeeman
    q_D_x_vec = q_B_constant * B_chunk * np.sin(theta)
    q_D_y_vec = q_B_constant * B_chunk * (-np.cos(theta))

    phi_x_tensor = np.zeros((N_B_chunk, 9))
    phi_y_tensor = np.zeros((N_B_chunk, 9))

    for b_idx in range(N_B_chunk):
        q_D_x = q_D_x_vec[b_idx]
        q_D_y = q_D_y_vec[b_idx]
        phi_x_tensor[b_idx, :] = np.array([q_D_x, -h + q_D_x, h + q_D_x, q_D_x, q_D_x, q_D_x + h, q_D_x - h, q_D_x + h, q_D_x - h])
        phi_y_tensor[b_idx, :] = np.array([q_D_y, q_D_y, q_D_y, -h+q_D_y, h+q_D_y, q_D_y + h, q_D_y - h, q_D_y - h, q_D_y + h])

    q_squared_tensor = phi_x_tensor**2 + phi_y_tensor**2
    surface_constants = np.pi / 2 * cut_off**2 * (2 * gamma * q_squared_tensor - 2 * mu + gamma * cut_off**2)

    angular_integrand_accumulator = []

    for t_val in theta_values:
        H_tensor = get_Hamiltonian_tensor(k_values, t_val, mu, B_y_vec, Delta, phi_x_tensor, gamma, Lambda, B_x_vec, phi_y_tensor)
        Z_tensor = np.linalg.eigvalsh(H_tensor)
        
        if T:
            stable_term = np.minimum(0.0, Z_tensor)
            log_term = - (1.0 / beta) * np.log1p(np.exp(-beta * np.abs(Z_tensor)))
            energy_density = stable_term + log_term
        else:
            energy_density = Z_tensor * (Z_tensor <= 0)
            
        radial_profiles = np.sum(energy_density, axis=3) * k_values[np.newaxis, np.newaxis, :]
        radial_integrals = scipy.integrate.trapezoid(radial_profiles, k_values, axis=2)
        angular_integrand_accumulator.append(radial_integrals)

    angular_integrand_accumulator = np.moveaxis(np.array(angular_integrand_accumulator), 0, -1)
    total_integrals = scipy.integrate.trapezoid(angular_integrand_accumulator, theta_values, axis=2)
    
    E = 0.5 * total_integrals + surface_constants

    n_xx_chunk = (E[:, 2] - 2 * E[:, 0] + E[:, 1]) / h**2
    n_yy_chunk = (E[:, 4] - 2 * E[:, 0] + E[:, 3]) / h**2
    n_xy_chunk = (E[:, 5] - E[:, 7] - E[:, 8] + E[:, 6]) / (4*h**2)

    return list(zip(n_xx_chunk, n_yy_chunk, n_xy_chunk))

# --- THE WRAPPER FUNCTION ---
def get_vectorized_superfluid_density(B_values, beta, n_cores=15):
    # Move environment array generation inside execution so it's calculated on demand
    N = 500  #1000
    k_1 = (-Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
    k_2 = (Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
    k_vals = np.append(np.linspace(0, 0.99*k_1, N), [
        np.linspace(0.99*k_1, 1.01*k_1, N),
        np.linspace(1.01*k_1, 0.99*k_2, N), 
        np.linspace(0.99*k_2, 1.01*k_2, N),
        np.linspace(1.01*k_2, cut_off, N)
    ])
    theta_vals = np.linspace(0, 2*np.pi, N)

    # Bundle all settings into a dict to pass down to children safely
    execution_params = {
        "Zeeman": True,
        "T": True,
        "beta": beta,
        "q_B_constant": 0,
        "k_values": k_vals,
        "theta_values": theta_vals
    }

    # Slice tasks into arrays for processing chunks
    CHUNKS_OF_B = 1 
    B_batched_chunks = [B_values[i:i + CHUNKS_OF_B] for i in range(0, len(B_values), CHUNKS_OF_B)]
    
    with multiprocessing.Pool(n_cores) as pool:
        # Freeze execution_params so pool.map continues to accept 1 iterable item
        frozen_worker = partial(process_B_chunk, params=execution_params)
        chunk_results = pool.map(frozen_worker, B_batched_chunks)
        
    flat_results = [item for sublist in chunk_results for item in sublist]
    superfluid_density_xx, superfluid_density_yy, _ = zip(*flat_results)
    
    return np.array(superfluid_density_xx), np.array(superfluid_density_yy)

def get_vectorized_superfluid_density_Zeeman_and_Doppler(B_values, beta, q_B_constant, n_cores=15):
    # Move environment array generation inside execution so it's calculated on demand
    N = 100  # 500 #1000
    k_1 = (-Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
    k_2 = (Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
    k_vals = np.append(np.linspace(0, 0.99*k_1, N), [
        np.linspace(0.99*k_1, 1.01*k_1, N),
        np.linspace(1.01*k_1, 0.99*k_2, N), 
        np.linspace(0.99*k_2, 1.01*k_2, N),
        np.linspace(1.01*k_2, cut_off, N)
    ])
    theta_vals = np.linspace(0, 2*np.pi, N)
    
    # Bundle all settings into a dict to pass down to children safely
    execution_params = {
        "Zeeman": True,
        "T": True,
        "beta": beta,
        "q_B_constant": q_B_constant,
        "k_values": k_vals,
        "theta_values": theta_vals
    }

    # Slice tasks into arrays for processing chunks
    CHUNKS_OF_B = 1 
    B_batched_chunks = [B_values[i:i + CHUNKS_OF_B] for i in range(0, len(B_values), CHUNKS_OF_B)]
    
    with multiprocessing.Pool(n_cores) as pool:
        # Freeze execution_params so pool.map continues to accept 1 iterable item
        frozen_worker = partial(process_B_chunk, params=execution_params)
        chunk_results = pool.map(frozen_worker, B_batched_chunks)
        
    flat_results = [item for sublist in chunk_results for item in sublist]
    superfluid_density_xx, superfluid_density_yy, _ = zip(*flat_results)
    
    return np.array(superfluid_density_xx), np.array(superfluid_density_yy)


