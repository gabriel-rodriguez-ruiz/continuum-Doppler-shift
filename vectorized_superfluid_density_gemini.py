# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:36:27 2026

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from pauli_matrices import tau_0, tau_z, sigma_0, tau_x, sigma_z, sigma_x, sigma_y

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
Lambda = 15 #15  # meV nm

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

q_B_constant = 0.023  #0.0114155 #0.0235921 #0.00342466 #5/(2*np.sqrt(gamma*mu))#0.024/8*5
theta = np.pi/2
N_phi = 3  
cut_off = 2 * k_F 

phi_y = 0
h = 1e-5

# --- TEMPERATURE PARAMETERS ---
T = True      # Set to True to active finite temperature thermodynamics
beta = 66     # Inverse Temperature (1 / k_B T)
Zeeman = True
N = 500 #500  #1000
n_cores = 15
points = 4 * n_cores

# k_values = np.append(np.linspace(0, 0.98*k_F, N), [np.linspace(0.98*k_F, 1.02*k_F, N),
#             np.linspace(1.02*k_F, cut_off, N)])
k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
k_values = np.append(np.linspace(0, 0.99*k_1, N), [np.linspace(0.99*k_1, 1.01*k_1, N),
            np.linspace(1.01*k_1, 0.99*k_2, N), np.linspace(0.99*k_2, 1.01*k_2, N),
            np.linspace(1.01*k_2, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, N)
q_c = Delta/(2*np.sqrt(mu*gamma))

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off,
              "phi_y":phi_y,
              "T": T, "beta":beta, "k_values": k_values,
              "theta_values": theta_values,
              "q_B_constant": q_B_constant}

# Pre-compute Kronecker structures
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
    """Constructs the Hamiltonian tensor for a chunk of B values and 5 stencils."""
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

    H = ( chi_k_plus[..., np.newaxis, np.newaxis] * KRON_PLUS
        - chi_k_minus[..., np.newaxis, np.newaxis] * KRON_MINUS
        + constant_terms )
                   
    if Lambda != 0:
        H += ( Lambda * (kx + px)[..., np.newaxis, np.newaxis] * KRON_PLUS_Y
             + Lambda * (-kx + px)[..., np.newaxis, np.newaxis] * KRON_MINUS_Y
             - Lambda * (ky + py)[..., np.newaxis, np.newaxis] * KRON_PLUS_X
             - Lambda * (-ky + py)[..., np.newaxis, np.newaxis] * KRON_MINUS_X )
    return H

def process_B_chunk(B_chunk):
    """Vectorized calculation over a subset chunk of B values supporting T=True/False."""
    N_B_chunk = len(B_chunk)
    
    B_x_vec = B_chunk * np.cos(theta) * Zeeman  ################
    B_y_vec = B_chunk * np.sin(theta) * Zeeman    ################
    q_D_x_vec = q_B_constant * B_chunk * np.sin(theta)
    q_D_y_vec = q_B_constant * B_chunk * (-np.cos(theta))

    phi_x_tensor = np.zeros((N_B_chunk, 9))
    phi_y_tensor = np.zeros((N_B_chunk, 9))

    for b_idx in range(N_B_chunk):
        q_D_x = q_D_x_vec[b_idx]
        q_D_y = q_D_y_vec[b_idx]
        phi_x_tensor[b_idx, :] = np.array([q_D_x, -h + q_D_x, h + q_D_x, q_D_x,     q_D_x,  q_D_x + h,  q_D_x - h, q_D_x + h, q_D_x - h])
        phi_y_tensor[b_idx, :] = np.array([q_D_y, q_D_y,      q_D_y,     -h+q_D_y,  h+q_D_y, q_D_y + h, q_D_y - h, q_D_y - h, q_D_y + h])

    q_squared_tensor = phi_x_tensor**2 + phi_y_tensor**2
    surface_constants = np.pi / 2 * cut_off**2 * (2 * gamma * q_squared_tensor - 2 * mu + gamma * cut_off**2)

    angular_integrand_accumulator = []

    for t_val in theta_values:
        H_tensor = get_Hamiltonian_tensor(k_values, t_val, mu, B_y_vec, Delta, phi_x_tensor, gamma, Lambda, B_x_vec, phi_y_tensor)
        Z_tensor = np.linalg.eigvalsh(H_tensor) # Shape: (N_B_chunk, 5, 3000, 4)
        
        # --- THERMODYNAMIC POTENTIAL EVALUATION ---
        if T:
            # Numerically stable evaluation of: -1/beta * ln(1 + exp(-beta * Z))
            # Separating the negative components preserves precision and avoids overflow warnings
            stable_term = np.minimum(0.0, Z_tensor)
            log_term = - (1.0 / beta) * np.log1p(np.exp(-beta * np.abs(Z_tensor)))
            energy_density = stable_term + log_term
        else:
            # T=False logic: Pure ground state energy density (sum of negative eigenvalues)
            mask = (Z_tensor <= 0)
            energy_density = Z_tensor * mask
            
        # Sum across the 4 Hamiltonian bands -> Result shape: (N_B_chunk, 5, 3000)
        radial_profiles = np.sum(energy_density, axis=3) * k_values[np.newaxis, np.newaxis, :]
        
        # Integrate along the radial dimension (axis 2) -> Result shape: (N_B_chunk, 5)
        radial_integrals = scipy.integrate.trapezoid(radial_profiles, k_values, axis=2)
        angular_integrand_accumulator.append(radial_integrals)

    # Convert to shape: (N_B_chunk, 5, N_theta)
    angular_integrand_accumulator = np.moveaxis(np.array(angular_integrand_accumulator), 0, -1)
    
    # Integrate along the angular dimension (axis 2) -> Result shape: (N_B_chunk, 5)
    total_integrals = scipy.integrate.trapezoid(angular_integrand_accumulator, theta_values, axis=2)
    
    # Total Free Energy / Grand Potential profile
    E = 0.5 * total_integrals + surface_constants

    # Compute numerical stencil solutions
    n_xx_chunk = (E[:, 2] - 2 * E[:, 0] + E[:, 1]) / h**2
    n_yy_chunk = (E[:, 4] - 2 * E[:, 0] + E[:, 3]) / h**2
    n_xy_chunk = (E[:, 5] - E[:, 7] - E[:, 8] + E[:, 6]) / (4*h**2)

    return list(zip(n_xx_chunk, n_yy_chunk, n_xy_chunk))

if __name__ == "__main__":
    B_values = np.linspace(0 * Delta, 3 * Delta, points)
    
    # Process 4 magnetic steps per core simultaneously
    CHUNKS_OF_B = 4 
    B_batched_chunks = [B_values[i:i + CHUNKS_OF_B] for i in range(0, len(B_values), CHUNKS_OF_B)]
    
    with multiprocessing.Pool(n_cores) as pool:
        chunk_results = pool.map(process_B_chunk, B_batched_chunks)
        
    flat_results = [item for sublist in chunk_results for item in sublist]
    superfluid_density_xx, superfluid_density_yy, superfluid_density_xy = zip(*flat_results)
    
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    superfluid_density_xy = np.array(superfluid_density_xy)
    data_folder = Path("Data/")
    data_folder.mkdir(exist_ok=True)
    
    name = f"superfluid_density_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}_T={T}_beta={beta}_q_B_constant={q_B_constant}_Zeeman={Zeeman}_theta={theta}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy, superfluid_density_xy=superfluid_density_xy,
             B_values=B_values, **parameters)
    print("Done! \007")

