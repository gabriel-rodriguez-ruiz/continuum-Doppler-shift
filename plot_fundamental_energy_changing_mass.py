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

file_to_open = data_folder / "total_fundamental_energy_B=0_phi_x_in_(-0.001-0.001)_Delta=0.08_lambda=0.0_points=19_N_phi=3_N=100_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
fundamental_energy_Al = Data["fundamental_energy_Al"]

phi_x_values = Data["phi_x_values"]
k_F = Data["k_F"]
Delta = Data["Delta"]
B = Data["B"]
q_B = Data["q_B"]
N = Data["N"]
mu = Data["mu"]
B_y = Data["B_y"]
gamma = Data["gamma"]
Lambda = Data["Lambda"]
cut_off = Data["cut_off"]
phi_y = Data["phi_y"]
B_x = Data["B_x"]
T = Data["T"]
beta = Data["beta"]
gamma_Al = Data["gamma_Al"]
k_F_Al = Data["k_F_Al"]
cut_off_Al = Data["cut_off_Al"]

fig_1, ax_1 = plt.subplots()

ax_1.plot(phi_x_values, fundamental_energy, label=f"{cut_off}")

ax_1.set_xlabel(r"$q_x$")
ax_1.set_ylabel(r"$E_0$")
ax_1.set_title(r"2DEG + Aluminum $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_1.legend()

fig_2, ax_2 = plt.subplots()

ax_2.plot(phi_x_values, fundamental_energy_2DEG-np.min(fundamental_energy_2DEG), label="0.0403")
# plt.axvline(-q_B, linestyle="dashed", color="red")
ax_2.set_title(r"2DEG $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")

fig_3, ax_3 = plt.subplots()

ax_3.plot(phi_x_values, fundamental_energy_Al-np.min(fundamental_energy_Al), label=f"{cut_off}")
ax_3.plot(phi_x_values, fundamental_energy_2DEG-np.min(fundamental_energy_2DEG), label=f"{cut_off}")

ax_3.set_title(r"Aluminum" + r"$; B/\Delta=$" + f"{B/Delta}")

ax_2.legend()


ax_3.legend()

#%%
file_to_open = data_folder / "total_fundamental_energy_B=0_phi_x_in_(-0.001-0.001)_Delta=0.08_lambda=0.0_points=19_N_phi=3_N=100_m=4.567333333333333e-28.npz"

Data = np.load(file_to_open)
fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
fundamental_energy_Al = Data["fundamental_energy_Al"]

phi_x_values = Data["phi_x_values"]
k_F = Data["k_F"]
Delta = Data["Delta"]
B = Data["B"]
q_B = Data["q_B"]
N = Data["N"]
mu = Data["mu"]
B_y = Data["B_y"]
gamma = Data["gamma"]
Lambda = Data["Lambda"]
cut_off = Data["cut_off"]
phi_y = Data["phi_y"]
B_x = Data["B_x"]
T = Data["T"]
beta = Data["beta"]
gamma_Al = Data["gamma_Al"]
k_F_Al = Data["k_F_Al"]
cut_off_Al = Data["cut_off_Al"]


ax_1.plot(phi_x_values, fundamental_energy, label=f"{cut_off}")

ax_1.set_xlabel(r"$q_x$")
ax_1.set_ylabel(r"$E_0$")
ax_1.set_title(r"2DEG + Aluminum $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_1.legend()


ax_2.plot(phi_x_values, fundamental_energy_2DEG-np.min(fundamental_energy_2DEG), label="0.0806")
# plt.axvline(-q_B, linestyle="dashed", color="red")
ax_2.set_title(r"2DEG $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_3.plot(phi_x_values, fundamental_energy_Al, label=f"{cut_off}")
ax_3.set_title(r"Aluminum" + r"$; B/\Delta=$" + f"{B/Delta}")

ax_2.legend()


ax_3.legend()

#%%
file_to_open = data_folder / "total_fundamental_energy_B=0_phi_x_in_(-0.001-0.001)_Delta=0.08_lambda=0.0_points=19_N_phi=3_N=100_m=1.1418333333333333e-28.npz"

Data = np.load(file_to_open)
fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
fundamental_energy_Al = Data["fundamental_energy_Al"]

phi_x_values = Data["phi_x_values"]
k_F = Data["k_F"]
Delta = Data["Delta"]
B = Data["B"]
q_B = Data["q_B"]
N = Data["N"]
mu = Data["mu"]
B_y = Data["B_y"]
gamma = Data["gamma"]
Lambda = Data["Lambda"]
cut_off = Data["cut_off"]
phi_y = Data["phi_y"]
B_x = Data["B_x"]
T = Data["T"]
beta = Data["beta"]
gamma_Al = Data["gamma_Al"]
k_F_Al = Data["k_F_Al"]
cut_off_Al = Data["cut_off_Al"]


ax_1.plot(phi_x_values, fundamental_energy, label=f"{cut_off}")

ax_1.set_xlabel(r"$q_x$")
ax_1.set_ylabel(r"$E_0$")
ax_1.set_title(r"2DEG + Aluminum $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_1.legend()


ax_2.plot(phi_x_values, fundamental_energy_2DEG - np.min(fundamental_energy_2DEG), label="0.02")
# plt.axvline(-q_B, linestyle="dashed", color="red")
ax_2.set_title(r"2DEG $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_3.plot(phi_x_values, fundamental_energy_Al, label=f"{cut_off}")
ax_3.set_title(r"Aluminum" + r"$; B/\Delta=$" + f"{B/Delta}")

ax_2.legend()


ax_3.legend()

#%%
file_to_open = data_folder / "total_fundamental_energy_B=0_phi_x_in_(-0.002-0.002)_Delta=0.05_lambda=0.0_points=19_N_phi=3_N=300_cut_off_Al=14.979071677815341.npz"

Data = np.load(file_to_open)
fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
fundamental_energy_Al = Data["fundamental_energy_Al"]

phi_x_values = Data["phi_x_values"]
k_F = Data["k_F"]
Delta = Data["Delta"]
B = Data["B"]
q_B = Data["q_B"]
N = Data["N"]
mu = Data["mu"]
B_y = Data["B_y"]
gamma = Data["gamma"]
Lambda = Data["Lambda"]
cut_off = Data["cut_off"]
phi_y = Data["phi_y"]
B_x = Data["B_x"]
T = Data["T"]
beta = Data["beta"]
gamma_Al = Data["gamma_Al"]
k_F_Al = Data["k_F_Al"]
cut_off_Al = Data["cut_off_Al"]


ax_1.plot(phi_x_values, fundamental_energy, label=f"{cut_off}")

ax_1.set_xlabel(r"$q_x$")
ax_1.set_ylabel(r"$E_0$")
ax_1.set_title(r"2DEG + Aluminum $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_1.legend()


ax_2.plot(phi_x_values, fundamental_energy_2DEG, label=f"{cut_off}")
# plt.axvline(-q_B, linestyle="dashed", color="red")
ax_2.set_title(r"2DEG $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")


ax_3.plot(phi_x_values, fundamental_energy_Al, label=f"{cut_off}")
ax_3.set_title(r"Aluminum" + r"$; B/\Delta=$" + f"{B/Delta}")

ax_2.legend()


ax_3.legend()