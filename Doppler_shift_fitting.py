# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:47:05 2026

@author: Gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_with_Doppler_shift_B_in_1.6_(0.0-3.0)_phi_x_in_(-0.0-0.0)_Delta=0.08_lambda=15_points=57_N_phi=3_N=100_T=True_beta=50_m=2.2836666666666667e-28.npz"

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



#%% 5.7 GHz resonator

data_folder = Path(r"Files/data gabriel")

file_path = data_folder / 'field_dep_5_7_GHz_0deg.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

# n_s_0 = sheetX['n_s 0°'].dropna()
# field_0 = sheetX["fields 0°"].dropna()
# n_s_0_error = sheetX["n_s 0° err"].dropna()

# n_s_45 = sheetX['n_s 45°'].dropna()
# field_45 = sheetX["fields 45°"].dropna()
# n_s_45_error = sheetX["n_s 45° err"].dropna()

# n_s_90 = sheetX['n_s 90°'].dropna()
# field_90 = sheetX["fields 90°"].dropna()
# n_s_90_error = sheetX["n_s 90° err"].dropna()

n_s_135 = sheetX['n_s 135°'].dropna()
field_135 = sheetX["fields 135°"].dropna()
n_s_135_error = sheetX["n_s 135° err"].dropna()

# fig, ax = plt.subplots()
# ax.errorbar(field_0[:41], n_s_0[:41], yerr=n_s_0_error[:41], label=r"$n_s(0°)$", color="violet", fmt="s")
# ax.errorbar(field_90[:41], n_s_90[:41], yerr=n_s_90_error[:41], label=r"$n_s(90°)$", color="orange", fmt="s")
# ax.errorbar(field_45, n_s_45, yerr=n_s_45_error, label=r"$n_s(45°)$", color="green", fmt="*")
# ax.errorbar(field_135, n_s_135, yerr=n_s_135_error, label=r"$n_s(135°)$", color="darkviolet", fmt="*")


data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")
file_path = data_folder / 'field_dep_5_7_GHz.xlsx'
# file_path = data_folder / 'field_dep_4_9_GHz.xlsx'
xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['Delta_n_s_0°'].dropna()
field_0 = sheetX["fields_0°"].dropna()
n_s_0_error = sheetX["Delta_n_s_err_0°"].dropna()

n_s_45 = sheetX['Delta_n_s_45°'].dropna()
field_45 = sheetX["fields_45°"].dropna()
n_s_45_error = sheetX["Delta_n_s_err_45°"].dropna()

n_s_90 = sheetX['Delta_n_s_90°'].dropna()
field_90 = sheetX["fields_90°"].dropna()
n_s_90_error = sheetX["Delta_n_s_err_90°"].dropna()

n_s_135 = sheetX['Delta_n_s_135°'].dropna()
field_135 = sheetX["fields_135°"].dropna()
n_s_135_error = sheetX["Delta_n_s_err_135°"].dropna()

fig, ax = plt.subplots()
# ax.errorbar(field_0[:41], n_s_0[:41], yerr=n_s_0_error[:41], label=r"$n_s(0°)$", color="violet", fmt="s")
ax.errorbar(field_0, n_s_0, yerr=n_s_0_error, label=r"$n_s(0°)$", color="violet", fmt="s")
ax.errorbar(field_45[:41], n_s_45[:41], yerr=n_s_45_error[:41], label=r"$n_s(45°)$", color="green", fmt="*")
# ax.errorbar(field_90[:41], n_s_90[:41], yerr=n_s_90_error[:41], label=r"$n_s(90°)$", color="orange", fmt="s")
ax.errorbar(field_90, n_s_90, yerr=n_s_90_error, label=r"$n_s(90°)$", color="orange", fmt="s")

ax.errorbar(field_135[:41], n_s_135[:41], yerr=n_s_135_error[:41], label=r"$n_s(135°)$", color="red", fmt="*")


#%%

data_folder = Path(r"./Data")

# Fit without Rashba
# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=0_points=60_N=1000_T=True_beta=50_q_B_constant=0.003_Zeeman=False.npz"
# Fit with Rashba
file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=50_q_B_constant=0.003_Zeeman=False_theta=1.5707963267948966.npz"

Data = np.load(file_to_open)

q_B_constant = Data["q_B_constant"]

# superfluid_density = Data["superfluid_density"]
B_values = Data["B_values"]
Delta = Data["Delta"]
Lambda = Data["Lambda"]
# phi_eq_B = Data["phi_eq"]
k_F = Data["k_F"]
beta = Data["beta"]
gamma = Data["gamma"]
mu = Data["mu"]
# C = Data["C"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
# q_B_constant = Data["q_B_constant"]


from scipy.optimize import curve_fit

def interpolation_for_theory(x):
    B_c = Delta/(2*np.sqrt(gamma*mu)*q_B_constant)
    return [
            np.interp(x, B_values/ B_c, superfluid_density_xx),        #I have change x to x/2
            np.interp(x, B_values/ B_c, superfluid_density_yy),
            ]

def model_parallel(x, a):
    return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1])/(interpolation_for_theory(0)[1] + a)# + c* x**2

def model_perpendicular(x, a):
    return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])/(interpolation_for_theory(0)[0] + a) # + c* x**2


def model_diagonal(x, c):
    return (interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2]) / interpolation_for_theory(0)[2] #+ c* x**2


#%% Interpolation
# x = B_values/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant))
# fig2, ax2 = plt.subplots()
# ax2.plot(x, interpolation_for_theory(x)[0])
# ax2.plot(x, interpolation_for_theory(x)[1])
# ax2.plot(x, superfluid_density_xx, "o")
# ax2.plot(x, superfluid_density_yy, "o")

#%%

B_c_1 = 1/(2*np.sqrt(gamma*mu)*q_B_constant + 1)
B_c_2 = 1/(2*np.sqrt(gamma*mu)*q_B_constant - 1)
B_c = 0.074  #Delta/(2*np.sqrt(gamma*mu)*q_B_constant)  # T
mu_B = 5.79e-2 # meV/T
g = 2 * 0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_0
B_perpendicular = field_90
B_diagonal = field_45

x_model_parallel  = field_0[:41] /B_c
x_model_perpendicular  = field_90[:41] /B_c
x_model_diagonal  = field_45[:41] /B_c


initial_parameters_parallel = [  1.21976767e+05]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_0[:41],
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [  1.21976767e+05 ]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_90[:41],
                                                   p0=initial_parameters_perpendicular
                                                   )

scale_factor_perpendicular = 4.2

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

# ax.plot(B_parallel[:41], model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
# ax.plot(B_values/Delta*B_c, (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+popt_parallel[0]), "-bo",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
# ax.plot(B_values/Delta*B_c, (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+popt_parallel[0]) + popt_parallel[1]*(B_values/Delta*B_c)**2, "-bo",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)
ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+popt_parallel[0]), "b",  label=r"fit of $n_s(\gamma=0, \theta=0°)$", zorder=3)


# ax.plot(B_perpendicular[:41], model_perpendicular(x_model_perpendicular, *popt_perpendicular), "--k",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
# ax.plot(B_perpendicular[0:19], c*B_perpendicular[0:19]**2, "--ko",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
# ax.plot(B_values/Delta*B_c, scale_factor_perpendicular * (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+popt_parallel[0]),
#         "--ko",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)),  ( (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0])),
        "--k",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
# ax.plot(B_values/Delta*B_c,  ( (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0])
#                               + scale_factor_perpendicular*popt_parallel[1]*(B_values/Delta*B_c)**2),
#         "--r",  label=r"fit of $n_s(\gamma=0, \theta=90°)$")
# plt.axvline(x=0.05, color='k', linestyle='--', linewidth=2)
# plt.axvline(x=0.1  , color='k', linestyle='--', linewidth=2)

# q_B_constant = 0.003#3/(2*np.sqrt(gamma*mu))
# B_c_1 = 1/(2*np.sqrt(gamma*mu)*q_B_constant + 1)
# B_c_2 = 1/(2*np.sqrt(gamma*mu)*q_B_constant - 1)

plt.axvline(x=B_c, color='r', linestyle='--', linewidth=2)
# plt.axvline(x=B_c_2*B_c, color='b', linestyle='--', linewidth=2)

# q_B_constant = 5/(2*np.sqrt(gamma*mu))
# B_c_1 = 1/(2*np.sqrt(gamma*mu)*q_B_constant + 1)
# B_c_2 = 1/(2*np.sqrt(gamma*mu)*q_B_constant - 1)

# plt.axvline(x=B_c_1*B_c, color='r', linestyle='--', linewidth=2)
# plt.axvline(x=B_c_2*B_c, color='r', linestyle='--', linewidth=2)

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$\Delta n_s$")
ax.set_xlim([0, 0.2])
# ax.legend()
plt.tight_layout()
plt.show()

#%% 45 degrees
# B_c = B_c*np.sqrt(2)

ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)),  1/2*( (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0]) + (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+ popt_parallel[0])),
        "--g",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")


# ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant))/np.sqrt(2),  1/np.sqrt(2)*( (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0]) + (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+ popt_parallel[0])),
#         "--ro",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")

# ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant))/np.sqrt(2),  1/np.sqrt(2)*( (superfluid_density_xx-superfluid_density_xx[0]+superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0]+superfluid_density_yy[0]+ popt_parallel[0])),
#         "--ro",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")

# ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant))*np.sqrt(2), 1/2*((superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0])-(superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+ popt_parallel[0])),
#         "--go",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")

# file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=0_points=60_N=500_T=True_beta=50_q_B_constant=0.002403126326708455_Zeeman=False_theta=0.7853981633974483.npz"

# Data = np.load(file_to_open)
# superfluid_density_xx = Data["superfluid_density_xx"]
# superfluid_density_yy = Data["superfluid_density_yy"]
# superfluid_density_xy = Data["superfluid_density_xy"]

# B_c = B_c*np.sqrt(2)



# ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)),   (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ np.sqrt(2)/scale_factor_perpendicular*popt_parallel[0]),
#         "--ro",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")


# ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant))*np.sqrt(2),   1/2*( (superfluid_density_xx-superfluid_density_xx[0])/(superfluid_density_xx[0]+ 1/scale_factor_perpendicular*popt_parallel[0]) + (superfluid_density_yy-superfluid_density_yy[0])/(superfluid_density_yy[0]+ popt_parallel[0])),
#         "--ro",  label=r"fit of $n_s(\gamma=0, \theta=45°)$")

