# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:30:23 2026

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
Lambda = 15  # meV nm

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

q_B_constant = 13/(6*np.sqrt(gamma*mu))  
theta = np.pi/2
N_phi = 3   
cut_off = 2 * k_F  

# Note: h is no longer needed for stencils but kept in parameters dictionary if needed for reference
h = 1e-5

# --- TEMPERATURE PARAMETERS ---
T = True       # Set to True to active finite temperature thermodynamics
beta = 100     # Inverse Temperature (1 / k_B T)

N = 1000  
n_cores = 15
points = 4 * n_cores

k_1 = (-Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2 + 4*gamma*mu)) / (2*gamma)
k_values = np.append(np.linspace(0, 0.99*k_1, N), [np.linspace(0.99*k_1, 1.01*k_1, N),
            np.linspace(1.01*k_1, 0.99*k_2, N), np.linspace(0.99*k_2, 1.01*k_2, N),
            np.linspace(1.01*k_2, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, N)
q_c = Delta/(2*np.sqrt(mu*gamma))

parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off,
              "T": T, "beta":beta, "k_values": k_values,
              "theta_values": theta_values,
              "q_B_constant": q_B_constant}

# Pre-compute Kronecker structures for the Hamiltonian
KRON_PLUS  = np.kron((tau_0 + tau_z) / 2, sigma_0)
KRON_MINUS = np.kron((tau_0 - tau_z) / 2, sigma_0)
KRON_Y     = np.kron(tau_0, sigma_y)
KRON_X     = np.kron(tau_0, sigma_x)
KRON_X_0   = np.kron(tau_x, sigma_0)
KRON_PLUS_Y  = np.kron((tau_0 + tau_z) / 2, sigma_y)
KRON_MINUS_Y = np.kron((tau_0 - tau_z) / 2, sigma_y)
KRON_PLUS_X  = np.kron((tau_0 + tau_z) / 2, sigma_x)
KRON_MINUS_X = np.kron((tau_0 - tau_z) / 2, sigma_x)

# Kubo-specific structural operators
KRON_00 = np.kron(tau_0, sigma_0)
KRON_z0 = np.kron(tau_z, sigma_0)
KRON_0y = np.kron(tau_0, sigma_y)
KRON_0x = np.kron(tau_0, sigma_x)

def get_Hamiltonian_tensor(k_1d, theta_val, mu, B_y_vec, Delta, q_Dx_vec, gamma, Lambda, B_x_vec):
    """Constructs the Hamiltonian tensor at the optimal helical momentum q_Dx (no stencils)."""
    k_x = k_1d * np.cos(theta_val)
    k_y = k_1d * np.sin(theta_val)

    px = q_Dx_vec[:, np.newaxis]
    kx = k_x[np.newaxis, :]
    ky = k_y[np.newaxis, :]

    chi_k_plus = gamma * ((kx + px)**2 + ky**2) - mu
    chi_k_minus = gamma * ((-kx + px)**2 + ky**2) - mu

    bx = B_x_vec[:, np.newaxis, np.newaxis, np.newaxis]
    by = B_y_vec[:, np.newaxis, np.newaxis, np.newaxis]
    constant_terms = - by * KRON_Y - bx * KRON_X - Delta * KRON_X_0

    H = ( chi_k_plus[..., np.newaxis, np.newaxis] * KRON_PLUS
        - chi_k_minus[..., np.newaxis, np.newaxis] * KRON_MINUS
        + constant_terms )
                   
    if Lambda != 0:
        H += ( Lambda * (kx + px)[..., np.newaxis, np.newaxis] * KRON_PLUS_Y
             + Lambda * (-kx + px)[..., np.newaxis, np.newaxis] * KRON_MINUS_Y
             - Lambda * ky[..., np.newaxis, np.newaxis] * KRON_PLUS_X
             - Lambda * (-ky)[..., np.newaxis, np.newaxis] * KRON_MINUS_X )
    return H

def process_B_chunk(B_chunk):
    """Evaluates superfluid density via the analytical Kubo formula framework."""
    N_B_chunk = len(B_chunk)
    
    B_x_vec = B_chunk * np.cos(theta)   
    B_y_vec = B_chunk * np.sin(theta)    
    q_D_x_vec = q_B_constant * B_chunk

    # Analytical second derivative of the boundary surface constants term
    deriv_surface = 2 * np.pi * gamma * cut_off**2

    angular_integrand_x = []
    angular_integrand_y = []

    for t_val in theta_values:
        # 1. Obtain local eigenvalues and eigenvectors 
        H_tensor = get_Hamiltonian_tensor(k_values, t_val, mu, B_y_vec, Delta, q_D_x_vec, gamma, Lambda, B_x_vec)
        E_tensor, V_tensor = np.linalg.eigh(H_tensor) # shapes: (N_B, N_k, 4) and (N_B, N_k, 4, 4)
        
        k_x = k_values * np.cos(t_val)
        k_y = k_values * np.sin(t_val)
        
        # 2. Build Analytical Current and Stiffness Operators
        J_x_tensor = (2 * gamma * k_x)[np.newaxis, :, np.newaxis, np.newaxis] * KRON_00 + \
                     (2 * gamma * q_D_x_vec)[:, np.newaxis, np.newaxis, np.newaxis] * KRON_z0 + \
                     Lambda * KRON_0y
                     
        J_y_tensor = (2 * gamma * k_y)[np.newaxis, :, np.newaxis, np.newaxis] * KRON_00 - \
                     Lambda * KRON_0x
                     
        Tau_tensor = 2 * gamma * KRON_z0
        
        # 3. Rotate Operators to Eigenbasis
        V_adj = np.conjugate(V_tensor).transpose(0, 1, 3, 2)
        J_x_tilde = V_adj @ J_x_tensor @ V_tensor
        J_y_tilde = V_adj @ J_y_tensor @ V_tensor
        Tau_tilde = V_adj @ Tau_tensor @ V_tensor
        
        # 4. Thermodynamic Weighting Factors
        if T:
            f = 1.0 / (1.0 + np.exp(np.clip(beta * E_tensor, -500, 500)))
            df = beta * f * (f - 1.0)
        else:
            f = np.where(E_tensor <= 0, 1.0, 0.0)
            df = np.zeros_like(E_tensor)
            
        E_n = E_tensor[:, :, :, np.newaxis]
        E_m = E_tensor[:, :, np.newaxis, :]
        f_n = f[:, :, :, np.newaxis]
        f_m = f[:, :, np.newaxis, :]
        df_n = df[:, :, :, np.newaxis]
        
        diff_E = E_n - E_m
        abs_diff_E = np.abs(diff_E)
        
        # Lehmann representation kernel matrix
        M = np.where(abs_diff_E > 1e-7, (f_n - f_m) / diff_E, df_n)
        
        # 5. Evaluate Diamagnetic and Paramagnetic Traces
        dia_term = np.sum(f * np.real(np.diagonal(Tau_tilde, axis1=2, axis2=3)), axis=2)
        para_term_x = np.sum(M * np.abs(J_x_tilde)**2, axis=(2, 3))
        para_term_y = np.sum(M * np.abs(J_y_tilde)**2, axis=(2, 3))
        
        integrand_x = (dia_term + para_term_x) * k_values[np.newaxis, :]
        integrand_y = (dia_term + para_term_y) * k_values[np.newaxis, :]
        
        # Integrate across radial 1D profiles
        radial_integrals_x = scipy.integrate.trapezoid(integrand_x, k_values, axis=1)
        radial_integrals_y = scipy.integrate.trapezoid(integrand_y, k_values, axis=1)
        
        angular_integrand_x.append(radial_integrals_x)
        angular_integrand_y.append(radial_integrals_y)

    angular_integrand_x = np.array(angular_integrand_x)
    angular_integrand_y = np.array(angular_integrand_y)
    
    # Integrate across angular coordinates
    total_integral_x = scipy.integrate.trapezoid(angular_integrand_x, theta_values, axis=0)
    total_integral_y = scipy.integrate.trapezoid(angular_integrand_y, theta_values, axis=0)
    
    # Factor 0.5 accounts for the BdG particle-hole redundancy scaling
    n_xx_chunk = 0.5 * total_integral_x + deriv_surface
    n_yy_chunk = 0.5 * total_integral_y + deriv_surface

    return list(zip(n_xx_chunk, n_yy_chunk))

if __name__ == "__main__":
    B_values = np.linspace(0 * Delta, 1 * Delta, points)
    
    CHUNKS_OF_B = 4 
    B_batched_chunks = [B_values[i:i + CHUNKS_OF_B] for i in range(0, len(B_values), CHUNKS_OF_B)]
    
    with multiprocessing.Pool(n_cores) as pool:
        chunk_results = pool.map(process_B_chunk, B_batched_chunks)
        
    flat_results = [item for sublist in chunk_results for item in sublist]
    superfluid_density_xx, superfluid_density_yy = zip(*flat_results)
    
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    data_folder = Path("Data/")
    data_folder.mkdir(exist_ok=True)
    
    name = f"superfluid_density_Kubo_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N={N}_T={T}_beta={beta}_q_B_constant={q_B_constant}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open, superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             B_values=B_values, **parameters)
    print("Done! \007")