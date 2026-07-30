# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:38:20 2026

@author: Gabriel
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy


data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")
# file_path = data_folder / 'field_dep_5_7_GHz.xlsx'
file_path = data_folder / 'field_dep_4_9_GHz.xlsx'
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


#%% Doppler

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=58_q_B_constant=0.0023_Zeeman=False_theta=1.5707963267948966.npz"

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
    return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1])/(interpolation_for_theory(0)[1] + a) #+ c* x**2

def model_perpendicular(x, a):
    return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])/(interpolation_for_theory(0)[0] + a) #+ c* x**2


def model_diagonal(x, c):
    return (interpolation_for_theory(x)[2] - interpolation_for_theory(0)[2]) / interpolation_for_theory(0)[2] #+ c* x**2


B_c_1 = 1/(2*np.sqrt(gamma*mu)*q_B_constant + 1)
B_c_2 = 1/(2*np.sqrt(gamma*mu)*q_B_constant - 1)
B_c = 0.079  #Delta/(2*np.sqrt(gamma*mu)*q_B_constant)  # T
mu_B = 5.79e-2 # meV/T
g = 2 * 0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_45
B_perpendicular = field_135
B_diagonal = field_45

x_model_parallel  = field_45[:41] /B_c
x_model_perpendicular  = field_135[:41] /B_c
x_model_diagonal  = field_45[:41] /B_c


initial_parameters_parallel = [  7.02780157e+04]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_45[:41],
                                          p0=initial_parameters_parallel)


initial_parameters_perpendicular = [  1.21976767e+05]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_135[:41],
                                                   p0=initial_parameters_perpendicular
                                                   )

scale_factor_perpendicular = 2

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), model_parallel(B_values/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), *popt_parallel), "--k")

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$\Delta n_s$")
ax.set_xlim([0, 0.2])
ax.legend()
plt.tight_layout()
plt.show()

ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), model_perpendicular(B_values/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), 1/scale_factor_perpendicular*popt_parallel),
        "--k")


ax.plot(B_values*B_c/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), 1/2*(model_parallel(B_values/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), *popt_parallel)+model_perpendicular(B_values/(Delta/(2*np.sqrt(gamma*mu)*q_B_constant)), 1/scale_factor_perpendicular*popt_parallel)),
        "--k")

#%% Zeeman

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=58_q_B_constant=0_Zeeman=True_theta=1.5707963267948966.npz"

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

def interpolation_for_theory(x):
    return [
            np.interp(x, B_values/Delta, superfluid_density_xx),        #I have change x to x/2
            np.interp(x, B_values/Delta, superfluid_density_yy),
            ]

def model_parallel(x, a):
    return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1])/(interpolation_for_theory(0)[1] + a) #+ c* x**2

def model_perpendicular(x, a):
    return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])/(interpolation_for_theory(0)[0] + a) #+ c* x**2


B_c = 0.079   #field_0[9]   #data["field 0°"][14]# 0.07  T critical field
mu_B = 5.79e-2 # meV/T
g = 2 * 0.08 / (mu_B*B_c)  #Delta/(mu_B*B_c )   #1 / 1.7
g_xx = g
g_yy = g

B_parallel = field_45
B_perpendicular = field_135
B_diagonal = field_0

x_model_parallel  = field_45[:41]/B_c
x_model_perpendicular  = field_135[:41]/B_c  #field_90[7:41]/B_c
x_model_diagonal  = field_0[:41]/B_c


initial_parameters_parallel = [ 10000]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_45[:41],
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [ 3.73583079e+06]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_135[:41],
                                                   p0=initial_parameters_perpendicular
                                                   )

scale_factor_perpendicular = 2

popt_perpendicular[0] = 1*popt_perpendicular[0]   # 2.5

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

ax.plot(B_values/Delta*B_c, model_parallel(B_values/Delta, *popt_parallel), "-.k")

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$\Delta n_s$")
ax.legend()
plt.tight_layout()
plt.show()

ax.plot(B_values/Delta*B_c, model_perpendicular(B_values/Delta, 1/scale_factor_perpendicular*popt_parallel),
        "-.k")

ax.plot(B_values/Delta*B_c, 1/2*(model_parallel(B_values/Delta, *popt_parallel)+model_perpendicular(B_values/Delta, 1/scale_factor_perpendicular*popt_parallel)),
        "-.k")

#%%

data_folder = Path(r"./Data")

file_to_open = data_folder / "superfluid_density_Delta=0.08_lambda=15_points=60_N=500_T=True_beta=66_q_B_constant=0.015_Zeeman=True_theta=1.5707963267948966.npz"
    
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
    return [
            np.interp(x, B_values, superfluid_density_xx),        #I have change x to x/2
            np.interp(x, B_values, superfluid_density_yy),
            ]

mu_B = 5.79e-2  # meV/T
g = 5.35  #3.47725  #11.4923#17.24
B_c_1 = Delta/(2*np.sqrt(gamma*mu)*q_B_constant*(0.5*mu_B*g) + 0.5*mu_B*g)
B_c_2 = Delta/(2*np.sqrt(gamma*mu)*q_B_constant*(0.5*mu_B*g) - 0.5*mu_B*g)
B_c = Delta/(2*np.sqrt(gamma*mu)*q_B_constant)  # T

B_parallel = field_45
B_perpendicular = field_135
B_diagonal = field_45

x_model_parallel  = field_45[:41] * 0.5*mu_B*g
x_model_perpendicular  = field_135[:41] *0.5*mu_B*g
x_model_diagonal  = field_45[:41] *0.5*mu_B*g


initial_parameters_parallel = [  1.21976767e+05]
popt_parallel, pcov_parallel = curve_fit(model_parallel, x_model_parallel, n_s_45[:41],
                                          p0=initial_parameters_parallel)

initial_parameters_perpendicular = [  1.21976767e+05]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, x_model_perpendicular, n_s_135[:41],
                                                   p0=initial_parameters_perpendicular
                                                   )

scale_factor_perpendicular = 2

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))

ax.plot(B_values/(0.5*mu_B*g), model_parallel(B_values, *popt_parallel), "-k")

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$\Delta n_s$")
ax.set_xlim([0, 0.2])
# ax.legend()
plt.tight_layout()
plt.show()

ax.plot(B_values/(0.5*mu_B*g),  model_perpendicular(B_values, 1/scale_factor_perpendicular*popt_parallel),
        "-k")

ax.plot(B_values/(0.5*mu_B*g), 1/2*(model_perpendicular(B_values, 1/scale_factor_perpendicular*popt_parallel)+model_parallel(B_values, *popt_parallel)),
        "-k")

