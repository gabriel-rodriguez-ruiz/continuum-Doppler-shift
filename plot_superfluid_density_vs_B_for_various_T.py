#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:30:24 2026

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=200_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
k_B = 8.617e-2   #meV/K
alpha = 1
Delta_S = 0.2
T_ref = 0.058
T = 1/(k_B*beta)
mu = Data["mu"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

Aluminum_contribution = Delta_S * np.tanh(Delta_S/(2*k_B*T))
Aluminum_contribution_0 = Delta_S * np.tanh(Delta_S/(2*k_B*T_ref))
gas_reference = superfluid_density_xx[0]
gamma = Delta_S/132251

# superfluid_density_xy = Data["superfluid_density_xy"]

fig, ax = plt.subplots()


# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")


ax.set_xlabel(r"$V/\Delta$", fontsize=24)
ax.set_ylabel(r"$\frac{D_s(T_{ref},B)-D_s(T_{ref}, B=0)}{D_s(T_{ref}, B=0)}$", fontsize=24)
# ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")
#%%

file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.07892914385211444_lambda=15_points=57_N_phi=3_N=100_T=True_beta=25_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]


superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

# superfluid_density_xy = Data["superfluid_density_xy"]




# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")



#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=50_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
T = 1/(k_B*beta)

# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-4*np.pi*mu))/(Aluminum_contribution_0+gamma*4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=75_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=100_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.07999999999777795_lambda=15_points=57_N_phi=3_N=100_T=True_beta=125_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")


# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-4*np.pi*mu))/(Aluminum_contribution_0+gamma*4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")


#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=19_N_phi=3_N=100_T=True_beta=15_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-4*np.pi*mu))/(Aluminum_contribution_0+gamma*4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")

#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=19_N_phi=3_N=100_T=True_beta=35_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")


#%%
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.07999999999998504_lambda=15_points=57_N_phi=3_N=100_T=True_beta=150_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
q_eq = Data["q_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
T = 1/(k_B*beta)

superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# ax.plot(B_values/Delta, superfluid_density_xx, "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-superfluid_density_xx[0])/superfluid_density_xx[0], "v-", label="perpendicular")
# ax.plot(B_values/Delta, (superfluid_density_xx-4*np.pi*mu)/(4*np.pi*mu), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_xx-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "v-", label=r"perpendicular $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}", fillstyle="none")

# ax.plot(B_values/Delta, superfluid_density_yy, "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-superfluid_density_yy[0])/superfluid_density_yy[0], "o-", label="parallel")
# ax.plot(B_values/Delta, (superfluid_density_yy-4*np.pi*mu)/(4*np.pi*mu), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")
ax.plot(B_values/Delta, alpha/2*(Aluminum_contribution-Aluminum_contribution_0+gamma*(superfluid_density_yy-gas_reference))/(Aluminum_contribution_0+gamma*gas_reference), "o--", label=r"parallel $\frac{k_BT}{\Delta_0}$" + f"={np.round((k_B*T/0.08), 3)}")

ax.legend(loc="upper right")


