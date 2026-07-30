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

# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.003_Zeeman=False_theta=0.7853981633974483.npz"
file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.003_Zeeman=True_theta=0.7853981633974483.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0_Zeeman=True_theta=0.7853981633974483.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=False_beta=50_q_B_constant=0.003_Zeeman=True_theta=0.7853981633974483.npz"

# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.006_Zeeman=True_theta=0.7853981633974483.npz"
#file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.0015_Zeeman=True_theta=0.7853981633974483.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.0035_Zeeman=True_theta=0.7853981633974483.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
# q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
gamma = Data["gamma"]
mu = Data["mu"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
q_B_constant = Data["q_B_constant"]
superfluid_density_xy = Data["superfluid_density_xy"]

# q_B_constant = 0.002403126326708455
fig, ax = plt.subplots()
ax.plot(B_values, superfluid_density_xx, "v-", label="field at 45°")
# ax.plot(B_values, superfluid_density_yy, "*-", label="field at 45°")

# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v", label="perpendicular")

# ax.plot(B_values, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values, 1/2*(superfluid_density_xx+superfluid_density_yy), "o-", label="diagonal")

# ax.plot(B_values, 1/2*(superfluid_density_xx+superfluid_density_yy+2*superfluid_density_xy), "o-", label="45°")
# ax.plot(B_values/B_c, (superfluid_density_xx-superfluid_density_xy), "o-", label="resta")
# ax.plot(B_values/B_c, (superfluid_density_xx+superfluid_density_xy), "o-", label="suma")

# ax.plot(B_values/B_c, 1/np.sqrt(2) * (superfluid_density_xx+superfluid_density_xy), "o-", label="45°")

# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o", label="parallel")

ax.set_xlabel(r"$\frac{1}{2}\mu_BgB$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.003_Zeeman=False_theta=1.5707963267948966.npz"
file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.003_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=False_beta=50_q_B_constant=0.003_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=0_points=60_N=500_T=False_beta=50_q_B_constant=0.003_Zeeman=True_theta=1.5707963267948966.npz"

# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.006_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.0015_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.0035_Zeeman=True_theta=1.5707963267948966.npz"

Data = np.load(file_to_open)
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
q_B_constant = Data["q_B_constant"]
B_values = Data["B_values"]
superfluid_density_xy = Data["superfluid_density_xy"]

ax.plot(B_values, 1/2*(superfluid_density_xx+superfluid_density_yy+2*superfluid_density_xy), "o-", label="field in y")
# ax.plot(B_values, superfluid_density_xx)
# ax.plot(B_values, superfluid_density_yy)
# ax.plot(B_values, superfluid_density_xy)


ax.legend()
