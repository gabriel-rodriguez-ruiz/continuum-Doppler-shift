#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:33:31 2026

@author: gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

data_folder = Path(r"./Data")

# file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=100_m=2.2836666666666667e-28.npz"
file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=19_N_phi=3_N=100_T=True_beta=75_m=2.2836666666666667e-28.npz"

Data = np.load(file_to_open)
# superfluid_density = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
# phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
# C = Data["C"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]

fig, ax = plt.subplots()
ax.scatter(B_values/Delta, superfluid_density_xx, label=r"$D_{s}(B_{\perp}, \textbf{q}_B=0)$",
             s=40, marker="v")


ax.legend(prop={'size': 4})
ax.set_ylabel(r"$D_s$")
ax.set_title(r"$\alpha_R k_F/\Delta=$" + f"{np.round(Lambda*k_F/Delta, 2)}")



# superfluid_density_yy = Data["superfluid_density_yy"]

ax.scatter(B_values/Delta, superfluid_density_yy, label=r"$D_{s}(B_{\parallel}, \mathbf{q}_B=0)$",
           s=40, marker="o")

ax.legend(fontsize=7, loc="upper right", ncols=2)

#%% 5.7 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_5_7_GHz_0deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['n_s 0°'].dropna()
field_0 = sheetX["fields 0°"].dropna()
n_s_0_error = sheetX["n_s 0° err"].dropna()

n_s_45 = sheetX['n_s 45°'].dropna()
field_45 = sheetX["fields 45°"].dropna()
n_s_45_error = sheetX["n_s 45° err"].dropna()

n_s_90 = sheetX['n_s 90°'].dropna()
field_90 = sheetX["fields 90°"].dropna()
n_s_90_error = sheetX["n_s 90° err"].dropna()

n_s_135 = sheetX['n_s 135°'].dropna()
field_135 = sheetX["fields 135°"].dropna()
n_s_135_error = sheetX["n_s 135° err"].dropna()

fig, ax = plt.subplots()
ax.errorbar(field_0[:41], n_s_0[:41], yerr=n_s_0_error[:41], label=r"$n_s(0°)$", color="violet", fmt="s")
# ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90[:41], n_s_90[:41], yerr=n_s_90_error[:41], label=r"$n_s(90°)$", color="orange", fmt="s")
# ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="*")


from scipy.optimize import curve_fit

def interpolation_for_theory(x):
    return [
            np.interp(x, B_values/Delta, superfluid_density_xx),        #I have change x to x/2
            np.interp(x, B_values/Delta, superfluid_density_yy),
            ]

def model_parallel(x, a, c):
    return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1]) / (a + interpolation_for_theory(0)[1]) #+ c* x**2

def model_perpendicular(x, c):
    return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0]) / interpolation_for_theory(0)[0] #+ c* x**2

def model_diagonal(x, c):
    return (interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2]) / interpolation_for_theory(0)[2] #+ c* x**2

B_c = field_0[15]   #field_0[9]   #data["field 0°"][14]# 0.07  T critical field
mu_B = 5.79e-2 # meV/T
g = 0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_0
B_perpendicular = field_90
B_diagonal = field_45

x_model_parallel  = field_0[:41]/B_c
x_model_perpendicular  = field_90[:41]/B_c  #field_90[7:41]/B_c
x_model_diagonal  = field_45[:41]/B_c


initial_parameters_parallel = [ 20000, 2.10435188e+06]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_0[:41],
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [ 3.73583079e+06]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_90[:41],
                                                   p0=initial_parameters_perpendicular
                                                   )

scale_factor_perpendicular = 4

popt_perpendicular[0] = 1*popt_perpendicular[0]   # 2.5

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

ax.plot(B_parallel[:41], model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
ax.plot(B_values/Delta*B_c, (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+popt_parallel[0]), "-bo",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)

# ax.plot(B_perpendicular[7:19], model_perpendicular(x_model_perpendicular, *popt_perpendicular) + c*B_perpendicular[7:19]**2, "--ko",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
# ax.plot(B_perpendicular[0:19], c*B_perpendicular[0:19]**2, "--ko",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
ax.plot(B_values/Delta*B_c, scale_factor_perpendicular * (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+popt_parallel[0]),
        "--ko",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")

plt.axvline(x=B_c, color='r', linestyle='--', linewidth=2)

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$\Delta n_s$")
ax.legend()
plt.tight_layout()
plt.show()