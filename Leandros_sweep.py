# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:28:18 2026

@author: Gabriel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- 0) Load Experimental Data for Comparison ---
filename='field_dep_4_9_GHz.dat'

print("Loading experimental data...")
data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")

try:
    df = pd.read_csv(data_folder / filename, sep='\t', encoding='UTF-8', on_bad_lines='skip')
except TypeError:
    df = pd.read_csv(data_folder / filename, sep='\t', encoding='UTF-8', error_bad_lines=False)
    
data_90 = df[['fields 90°', 'Delta n_s 90°']].dropna()
B_exp = data_90['fields 90°'].values * 1000  # mT
ns_exp = data_90['Delta n_s 90°'].values

# --- 1) Real Material Parameters (InAs 2DEG) ---
mu = 50.6
m_eff_ratio = 0.04
alpha_R = 15.0
Delta0 = 0.08

hbar_sq_over_2me = 38.1
t_factor = hbar_sq_over_2me / m_eff_ratio

kc = np.sqrt(mu / t_factor)
k_so = alpha_R / (2 * t_factor)

k_min = kc - k_so - 0.002
k_max = kc + k_so + 0.002

# Fast scanning resolution (bump back up for final paper figures if needed)
N_k, N_theta = 200, 200

k_arr = np.linspace(k_min, k_max, N_k)
theta_arr = np.linspace(0, 2*np.pi, N_theta)
K, THETA = np.meshgrid(k_arr, theta_arr)

KX = K * np.cos(THETA)
KY = K * np.sin(THETA)

dk = k_arr[1] - k_arr[0]
dtheta = theta_arr[1] - theta_arr[0]
d2k = K * dk * dtheta

B_axis = np.linspace(0, 0.20, 30)  # 30 points for fast scanning
dq = 1e-5

def build_eigvals_BdG(qx=0.0, qBx=0.0, Vy=0.0):
    shape = KX.shape
    kpx, kpy = KX + qx + qBx, KY
    kmx, kmy = -KX + qx + qBx, -KY

    xi_p = t_factor * (kpx**2 + kpy**2) - mu
    xi_m = t_factor * (kmx**2 + kmy**2) - mu

    soc_p = alpha_R * (kpy + 1j * kpx)
    soc_m = alpha_R * (kmy + 1j * kmx)
    zeeman = -1j * Vy

    H = np.zeros(shape + (4, 4), dtype=complex)
    H[..., 0, 0] = xi_p
    H[..., 1, 1] = xi_p
    H[..., 0, 1] = soc_p + zeeman
    H[..., 1, 0] = np.conj(soc_p + zeeman)
    H[..., 2, 2] = -xi_m
    H[..., 3, 3] = -xi_m
    H[..., 2, 3] = -soc_m + zeeman
    H[..., 3, 2] = np.conj(soc_m + zeeman)
    H[..., 0, 2] = -Delta0
    H[..., 2, 0] = -Delta0
    H[..., 1, 3] = -Delta0
    H[..., 3, 1] = -Delta0

    return np.linalg.eigvalsh(H), xi_p

def calc_ground_state_energy_temp(qx, qBx, Vy, T_mK):
    eigvals, xi_p = build_eigvals_BdG(qx, qBx, Vy)
    kbT = max(T_mK * 0.00008617, 0.001)
    occupancies = 0.5 * (1 - np.tanh(eigvals / (2*kbT)))
    neg_E_sum = np.sum(eigvals * occupancies, axis=-1)
    integrand = 0.5 * neg_E_sum + xi_p
    return np.sum(integrand * d2k) / (2 * np.pi)**2

def Stiffness(Bc1, Bc2, T_mK):
    vF_factor = Delta0 * (Bc1 + Bc2) / (Bc1 * Bc2) / 20
    g_factor  = Delta0 * (Bc2 - Bc1) / (Bc1 * Bc2) / 0.0289 / 2

    stiffness_arr = []
    spectral_gap_arr = []
    v_F = 2 * t_factor * kc

    for B in B_axis:
        Vy = 0.0289 * g_factor * B
        V_D = vF_factor * 10 * B
        qBx = V_D / v_F

        eigvals, _ = build_eigvals_BdG(qx=0.0, qBx=qBx, Vy=Vy)
        gap = np.min(np.abs(eigvals))
        spectral_gap_arr.append(gap)

        E_plus  = calc_ground_state_energy_temp(qx=dq,  qBx=qBx, Vy=Vy, T_mK=T_mK)
        E_zero  = calc_ground_state_energy_temp(qx=0.0, qBx=qBx, Vy=Vy, T_mK=T_mK)
        E_minus = calc_ground_state_energy_temp(qx=-dq, qBx=qBx, Vy=Vy, T_mK=T_mK)

        rho_s = (E_plus - 2*E_zero + E_minus) / (dq**2)
        stiffness_arr.append(rho_s)

    stiff = np.array(stiffness_arr)
    delta_stiff = (stiff - stiff[0]) / stiff[0]
    return spectral_gap_arr, delta_stiff

# --- 2) Sweep Lists ---
Bc1_list = np.linspace(0.04, 0.07, 7)   # 7 steps
Bc2_list = np.linspace(0.075, 0.105, 7) # 7 steps
T_mK_list = np.linspace(100, 500, 9)    # 9 steps -> Total = 225 runs

# Shifted upward to make your previous maximums the new lower-middle bounds
Bc1_list = np.linspace(0.060, 0.090, 7)   # 60 to 90 mT (7 steps: 60, 65, 70, 75, 80, 85, 90)
Bc2_list = np.linspace(0.100, 0.140, 9)   # 100 to 140 mT (9 steps: 100, 105, 110...140)
# Narrowed around the known minimum to increase resolution
T_mK_list = np.linspace(250, 450, 9)      # 250 to 450 mK (9 steps: 250, 275, 300...450)

best_score = float('inf')
best_params = None
best_curves = None

total_runs = len(Bc1_list) * len(Bc2_list) * len(T_mK_list)
current_run = 0

# Storage containers for all runs
history_Bc1, history_Bc2, history_T, history_error = [], [], [], []
history_curves = []

print(f"Starting parameter sweep across {total_runs} combinations...")

for Bc1 in Bc1_list:
    for Bc2 in Bc2_list:
        for T_mK in T_mK_list:
            current_run += 1
            print(f"Run {current_run}/{total_runs} -> Bc1={Bc1*1000:.1f}mT, Bc2={Bc2*1000:.1f}mT, T={T_mK:.0f}mK")

            gap, stiff = Stiffness(Bc1, Bc2, T_mK)

            # Interpolate to match experimental points for error calculation
            sim_B_mT = B_axis * 1000
            interp_stiff = np.interp(B_exp, sim_B_mT, stiff)

            if np.max(np.abs(ns_exp)) > 0 and np.min(interp_stiff) != 0:
                scaling_guess = np.min(ns_exp) / np.min(interp_stiff)
            else:
                scaling_guess = 1.0

            error = np.mean((ns_exp - (interp_stiff * scaling_guess))**2)

            # Log history
            history_Bc1.append(Bc1)
            history_Bc2.append(Bc2)
            history_T.append(T_mK)
            history_error.append(error)
            history_curves.append(stiff)

            if error < best_score:
                best_score = error
                best_params = (Bc1, Bc2, T_mK, scaling_guess)
                best_curves = (gap, stiff * scaling_guess)

# Save all results to disk so you can inspect neighboring solutions later!
np.savez("sweep_results_2.npz",
         Bc1=history_Bc1,
         Bc2=history_Bc2,
         T=history_T,
         error=history_error,
         curves=history_curves,
         B_axis=B_axis * 1000)

print("\n--- SWEEP COMPLETE ---")
print(f"Results saved to 'sweep_results.npz'.")
print(f"Best Fit Found: Bc1={best_params[0]*1000:.1f}mT, Bc2={best_params[1]*1000:.1f}mT, T={best_params[2]:.1f}mK (MSE: {best_score:.6f})")