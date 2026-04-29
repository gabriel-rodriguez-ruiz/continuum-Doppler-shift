#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:34:59 2026

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#%% BCS temperature dependence

fig_BCS, ax_BCS = plt.subplots()
k_B = 8.617e-2  # meV/K
T = np.linspace(0.0001, 0.8)
T_c = 1.61
Delta_0 = 1.76*k_B*T_c  #meV
alpha = 1
Al_contribution = np.tanh(Delta_0*1.74*np.sqrt(1-T/T_c)/(2*k_B*T)) - 1
ax_BCS.plot(T, alpha/2 * Al_contribution)
# ax_BCS.plot(T, alpha/2 * ( (1 - (T/T_c)**4 )**(1/2) * np.tanh(Delta_0*(1 - (T/T_c)**4 )**(1/2)/(2*k_B*T)) - 1) )
# ax_BCS.plot(T, alpha/2 * (np.tanh(1.74*np.sqrt(T_c/T - 1)) - 1 ) )
ax_BCS.set_xlabel("T[K]")
ax_BCS.set_ylabel(r"$\frac{\alpha}{2}(\Delta(T)-\Delta(0))/\Delta(0)$")

#%%

plt.rcParams.update({
  "text.usetex": True,
})

data_folder = Path(r"./Data")

# file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=14.96_points=19_N_phi=3_N=100_C=0_T=True_B=0.072.npz"
file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.14_Delta_0=0.141043056.npz"
# file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.141_Delta_0=0.141043056.npz"
# file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.140338_Delta_0=0.141043056.npz"
# file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.139774_Delta_0=0.141043056.npz"
file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.230207_Delta_0=0.1627808746666667.npz"
# file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(12.346-1160.497)_phi_x_in_(-0.0-0.0)_lambda=0_points=19_N_phi=3_N=100_C=0_T=True_B=0.0_Delta_0=0.244171312.npz"



Data = np.load(file_to_open)
beta_values = Data["beta_values"]
Delta_0 = Data["Delta_0"]
Lambda = Data["Lambda"]
k_F = Data["k_F"]
k_B = 8.617e-2  # meV/K
T_values = 1/(beta_values*k_B)
B = Data["B"]


superfluid_density_finite_differences_perpendicular_0 = Data["superfluid_density_finite_differences_0"]

fig, ax = plt.subplots()

# ax.plot(T_values, ((superfluid_density_finite_differences_perpendicular_0-superfluid_density_finite_differences_perpendicular_0[0])/superfluid_density_finite_differences_perpendicular_0[0]), "v-", label=r"perpendicular B=0.072",
#             color="C2")

ax.plot(T_values, (superfluid_density_finite_differences_perpendicular_0-600)/600, "v-", label=r"perpendicular B=0.072",
            color="C2")

ax.set_xlabel(r"$T[K]$")
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
# ax.plot(T_values, ((superfluid_density_yy_0-superfluid_density_yy_0[0])/superfluid_density_yy_0[0]), "o-", label=r"parallel B=0.072",
#             color="C1")
ax.plot(T_values, (superfluid_density_yy_0-600)/600, "o-", label=r"parallel B=0.072",
            color="C1")

    #%%

file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=14.96_points=19_N_phi=3_N=100_C=0_T=True_B=0.088.npz"
file_to_open = data_folder / "superfluid_density_B_in_1.6_beta_in_(14.506-1160.497)_phi_x_in_(-0.0-0.0)_lambda=15_points=19_N_phi=3_N=100_C=0_T=True_B=0.0_Delta_0=0.1627808746666667.npz"

Data = np.load(file_to_open)
beta_values = Data["beta_values"]
# Delta = Data["Delta"]
Lambda = Data["Lambda"]
k_F = Data["k_F"]
k_B = 8.617e-2  # meV/K
T_values = 1/(beta_values*k_B)
B = Data["B"]
n = 4*np.pi*50.6

superfluid_density_finite_differences_perpendicular_0 = Data["superfluid_density_finite_differences_0"]

fig, ax = plt.subplots()
# ax.plot(T_values, (superfluid_density_finite_differences_perpendicular_0 -n)/n, "v-", label=r"perpendicular" + f" B={B}",
#             color="C3")
ax.plot(T_values, superfluid_density_finite_differences_perpendicular_0, "v-", label=r"perpendicular" + f" B={B}",
            color="C3")

ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\lambda/\Delta=$" + f"{Lambda/Delta}")

superfluid_density_yy_0 = Data["superfluid_density_yy_0"]
# ax.plot(T_values, (superfluid_density_yy_0-n)/n, "o-", label=r"parallel" + f" B={B}",
            # color="C4")
ax.plot(T_values, superfluid_density_yy_0, "o-", label=r"parallel" + f" B={B}",
            color="C4")


ax.legend()

#%%

fig_BCS, ax_BCS = plt.subplots()
k_B = 8.617e-2  # meV/K
T = np.linspace(0, 0.8)
T_c = 1.86
Delta_0 = 1.76*k_B*T_c  #meV
alpha = 1
Al_contribution = np.tanh(Delta_0*1.74*np.sqrt(1-T/T_c)/(2*k_B*T)) - 1

ax_BCS.plot(T, alpha/2 * Al_contribution)

ax_BCS.set_xlabel("T[K]")
ax_BCS.set_ylabel(r"$\frac{\alpha}{2}(\Delta(T)-\Delta(0))/\Delta(0)$")

k_B = 8.617e-2  # meV/K
T_c = 1.86
Al_stiffness = 600000   #300000
scale_factor = 5
# ax_BCS.plot(T_values, alpha/2 * ( Al_contribution + ((superfluid_density_yy_0-superfluid_density_yy_0[0])/(superfluid_density_yy_0[0]+Al_stiffness)) - 1), "o")
# ax_BCS.plot(T_values, alpha/2 * ( Al_contribution + ((superfluid_density_finite_differences_perpendicular_0-superfluid_density_finite_differences_perpendicular_0[0])/np.abs(superfluid_density_finite_differences_perpendicular_0[0] + Al_stiffness)) - 1), "v")


D_Al_T =  Al_stiffness * Delta_0*1.74* np.tanh(Delta_0*1.74*np.sqrt(1-T_values/T_c)/(2*k_B*T_values))
D_Al_0 = Al_stiffness * Delta_0*1.74
    
ax_BCS.plot(T_values, alpha/2 * ( ( D_Al_T - D_Al_0 + superfluid_density_yy_0 - superfluid_density_yy_0[0] ) / ( D_Al_0 + superfluid_density_yy_0[0] ) ) - 0.001 , "o")
ax_BCS.plot(T_values, alpha/2 * ( ( D_Al_T - D_Al_0 + scale_factor*(superfluid_density_finite_differences_perpendicular_0 - superfluid_density_finite_differences_perpendicular_0[0]) ) / ( D_Al_0 + scale_factor*np.abs(superfluid_density_finite_differences_perpendicular_0[0] ) ) ) - 0.005 , "v")

ax_BCS.set_xlabel("T[K]")
ax_BCS.set_ylabel(r"$\frac{(D_{Al}(T)-D_{Al}(0)+D_{2DEG}(T)-D_{2DEG}(0))}{D_{Al}(0) + D_{2DEG}(0) }$")

plt.tight_layout()

#%%
import pandas as pd

data_folder = Path("Files/corrected data gabriel 04_26/temp depend")

df_final = pd.read_excel(data_folder / 'T_dep_5_7_GHz.xlsx')
fig, ax = plt.subplots()

ax.errorbar(df_final["T 0mT"]/1000, df_final["Delta fr 0mT"], yerr=df_final['Delta fr err 0mT'], fmt='sk')
ax.errorbar(df_final["T 30mT 0°"]/1000, df_final["Delta fr 30mT 0°"], yerr=df_final['Delta fr err 30mT 0°'], fmt='sr')
ax.errorbar(df_final["T 30mT 90°"]/1000, df_final["Delta fr 30mT 90°"], yerr=df_final['Delta fr err 30mT 90°'], fmt='sr', markerfacecolor="None")
ax.errorbar(df_final["T 60mT 0°"]/1000, df_final["Delta fr 60mT 0°"], yerr=df_final['Delta fr err 60mT 0°'], fmt='sb')
ax.errorbar(df_final["T 60mT 90°"]/1000, df_final["Delta fr 60mT 90°"], yerr=df_final['Delta fr err 60mT 90°'], fmt='sb', markerfacecolor="None")

ax.set_xlabel(r"$T [K]$")
ax.set_ylabel(r"$\Delta f$")

k_B = 8.617e-2  # meV/K
T = np.linspace(0, 0.94)
T_c = 1.61  #1.86
Delta_0 = 1.76*k_B*T_c  #meV
alpha = 1
Al_contribution = np.tanh(Delta_0*1.74*np.sqrt(1-T/T_c)/(2*k_B*T)) - 1

ax.plot(T, alpha/2 * Al_contribution)

geometric_factor = np.float64(6.432694164110614e-06)
anisotropic_factor = 4.5
# ax_BCS.plot(T_values, alpha/2 * ( Al_contribution + ((superfluid_density_yy_0-superfluid_density_yy_0[0])/(superfluid_density_yy_0[0]+Al_stiffness)) - 1), "o")
# ax_BCS.plot(T_values, alpha/2 * ( Al_contribution + ((superfluid_density_finite_differences_perpendicular_0-superfluid_density_finite_differences_perpendicular_0[0])/np.abs(superfluid_density_finite_differences_perpendicular_0[0] + Al_stiffness)) - 1), "v")



Al_contribution = np.tanh(Delta_0*1.74*np.sqrt(1-T_values/T_c)/(2*k_B*T_values)) - 1

ax.plot(T_values, alpha/2 * ( (Al_contribution + geometric_factor*(superfluid_density_yy_0 - superfluid_density_yy_0[0]) ) / ( 1 + geometric_factor*superfluid_density_yy_0[0] ) ) - 0.00055 , "o-")
ax.plot(T_values, alpha/2 * ( ( Al_contribution + anisotropic_factor*geometric_factor*(superfluid_density_finite_differences_perpendicular_0 - superfluid_density_finite_differences_perpendicular_0[0]) ) / ( 1 + anisotropic_factor*geometric_factor*np.abs(superfluid_density_finite_differences_perpendicular_0[0] ) ) ) - 0.0042 , "v-")

# ax.plot(T, alpha/2 * (np.tanh(1.74*np.sqrt(T_c/T - 1)) - 1 ) )


Delta_D_parallel = ( ( D_Al_T - D_Al_0 + superfluid_density_yy_0 - superfluid_density_yy_0[0] ) / ( D_Al_0 + superfluid_density_yy_0[0] ) )
Delta_D_perpendicular =  ( ( D_Al_T - D_Al_0 + scale_factor*(superfluid_density_finite_differences_perpendicular_0 - superfluid_density_finite_differences_perpendicular_0[0]) ) / ( D_Al_0 + scale_factor*np.abs(superfluid_density_finite_differences_perpendicular_0[0] ) ) )


ax.set_xlabel("T[K]")
ax.set_ylabel(r"$\frac{(D_{Al}(T)-D_{Al}(0)+D_{2DEG}(T)-D_{2DEG}(0))}{D_{Al}(0) + D_{2DEG}(0) }$")

plt.tight_layout()
