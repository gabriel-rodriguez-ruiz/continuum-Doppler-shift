# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:40:45 2026

@author: Gabriel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

filename='field_dep_4_9_GHz.dat'

# --- 1) Load Experimental Data ---
print("Loading experimental data...")
data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")

try:
    df = pd.read_csv(data_folder / filename, sep='\t', encoding='UTF-8', on_bad_lines='skip')
except TypeError:
    df = pd.read_csv(data_folder / filename, sep='\t', encoding='UTF-8', error_bad_lines=False)
    
data_90 = df[['fields 90°', 'Delta n_s 90°']].dropna()
B_exp = data_90['fields 90°'].values * 1000  # mT
ns_exp = data_90['Delta n_s 90°'].values

# --- 2) Load the Saved Unscaled Sweep Results ---
print("Loading sweep_results_3.npz...")
data = np.load(Path(r"Files") /"sweep_results_3.npz", allow_pickle=True)
Bc1s = data['Bc1']
Bc2s = data['Bc1']
Ts = data['T']
curves = data['curves']
B_axis_mT = data['B_axis']

# --- 3) Global Minimization Loop (0-150 mT range) ---
# Notice we extended the fit range to 150 mT based on your recent plots
fit_mask = (B_exp >= 0) & (B_exp <= 150)
B_fit = B_exp[fit_mask]
ns_fit = ns_exp[fit_mask]

results_3 = []

print("Scanning the entire parameter landscape... This may take a moment.")
for idx in range(len(curves)):
    raw_curve = curves[idx]

    # Interpolate simulation onto experimental B_fit points
    sim_interp_base = np.interp(B_fit, B_axis_mT, raw_curve)

    # Define objective function (Mean Squared Error) for this specific curve
    def objective(params):
        scale = params[0]
        sim_scaled = sim_interp_base * scale
        return np.mean((ns_fit - sim_scaled)**2)

    # Initial guess using the linear least-squares estimate
    denom = np.sum(sim_interp_base**2)
    init_scale = np.sum(ns_fit * sim_interp_base) / denom if denom > 1e-12 else 1.0

    # Run 1D minimization for the scale
    res = minimize(objective, [init_scale], method='Nelder-Mead')
    opt_scale = res.x[0]
    final_error = res.fun

    # Store all parameters and their resulting error
    results_3.append({
        'idx': idx,
        'T': Ts[idx],
        'Bc1': Bc1s[idx] * 1000,
        'Bc2': Bc2s[idx] * 1000,
        'scale': opt_scale,
        'error': final_error,
        'scaled_curve': raw_curve * opt_scale
    })
    
# --- 4) Sort Results and Display the Global Best ---
# Sort the list of dictionaries by error (lowest first)
results_3_sorted = sorted(results_3, key=lambda x: x['error'])

print("\n--- TOP 5 GLOBAL BEST FITS ---")
for i in range(5):
    r = results_3_sorted[i]
    print(f"{i+1}. T = {r['T']:.1f} mK | Bc1 = {r['Bc1']:.1f} mT | Bc2 = {r['Bc2']:.1f} mT | Scale = {r['scale']:.4f} | MSE = {r['error']:.6e}")

# Extract the absolute best fit
best = results_3_sorted[0]

# --- 5) Plot the Global Best Fit ---
fig, ax = plt.subplots(figsize=(9, 7))

# Plot the best fit in thick red
ax.plot(B_axis_mT, best['scaled_curve'], '-', color='red', lw=3.0,
        label=f"Global Best Fit\n($T$={best['T']:.1f}mK, $B_{{c1}}$={best['Bc1']:.1f}mT, $B_{{c2}}$={best['Bc2']:.1f}mT)")

# Plot Experiment on top
ax.plot(B_exp, ns_exp, marker='s', linestyle='', color='black', markersize=6, label="Experiment $\Delta n_s$", zorder=10)

# Formatting
ax.set_xlabel("Magnetic Field B (mT)", fontsize=14)
ax.set_ylabel(r"$\Delta n_s$", fontsize=14)
#ax.set_title("Absolute Best Fit Across All Temperatures (Range: 0–150 mT)", fontsize=15)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(0, 200)
ax.legend(loc='lower left', fontsize=11)

plt.tight_layout()
plt.show()

results_3_sorted_49=results_3_sorted