# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:31:02 2026

@author: Gabriel
"""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=120_N=1000_T=False_beta=50_q_B_constant=0_Zeeman=True_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=120_N=1000_T=False_beta=50_q_B_constant=0.00365_Zeeman=False_theta=1.5707963267948966.npz"
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=120_N=1000_T=False_beta=50_q_B_constant=0.0114155_Zeeman=True_theta=1.5707963267948966.npz"

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

fig, ax = plt.subplots()

B_values = B_values/Delta
# B_values = B_values/Delta*2*np.sqrt(gamma*mu)*0.00365
mu_B = 5.79e-2  # meV/T
g = 11.5
# B_values = B_values/(0.5*mu_B*g)

ax.plot(B_values, (superfluid_density_xx/superfluid_density_xx[0]+40)/41, "-")
ax.plot(B_values, (superfluid_density_yy/superfluid_density_yy[0]+40)/41, "-")

ax.set_xlabel(r"$V_Z/\Delta_0$")
# ax.set_xlabel(r"$V_D/\Delta_0$")
# ax.set_xlabel(r"$B(T)$")

ax.set_ylabel(r"$D_s(B)/D_s(0)$")
