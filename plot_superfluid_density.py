#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 14:44:11 2025

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.2_lambda=0_points=19_N_phi=3_N=100_T=False_beta=100_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

fig, ax = plt.subplots()
ax.plot(B_values/Delta, superfluid_density_xx, "v", label="perpendicular",
             color="green")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

ax.plot(B_values/Delta, superfluid_density_yy, "o", label="parallel",
            color="red")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_xx[0], "o", label="parallel")

ax.legend()
ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

#%% q_eq

fig, ax = plt.subplots()
ax.plot(B_values/Delta, q_eq, "o")

ax.set_xlabel(r"$B/\Delta$")
ax.set_ylabel(r"$q*$")