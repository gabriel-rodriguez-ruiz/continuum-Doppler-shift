#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:02:37 2026

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=0.64_points=19_N_phi=3_N=100_T=True_beta=100_m=2.283666666666667e-27.npz"

Data = np.load(file_to_open)

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_xx[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=0.64_points=19_N_phi=3_N=100_T=True_beta=100.npz"

Data = np.load(file_to_open)

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_xx[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=0.64_points=19_N_phi=3_N=300_T=True_beta=100.npz"

Data = np.load(file_to_open)

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_xx[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=0.64_points=19_N_phi=3_N=100_T=True_beta=100_m=2.2836666666666666e-26.npz"

Data = np.load(file_to_open)

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_xx[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")