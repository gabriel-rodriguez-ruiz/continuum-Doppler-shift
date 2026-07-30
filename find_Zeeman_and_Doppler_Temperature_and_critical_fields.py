# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:04:20 2026

@author: Gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit

data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")
file_path = data_folder / 'field_dep_5_7_GHz.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

n_s_0 = sheetX['Delta_n_s_0°'].dropna()
field_0 = sheetX["fields_0°"].dropna()
n_s_0_error = sheetX["Delta_n_s_err_0°"].dropna()

# n_s_45 = sheetX['Delta_n_s_45°'].dropna()
# field_45 = sheetX["fields_45°"].dropna()
# n_s_45_error = sheetX["Delta_n_s_err_45°"].dropna()

n_s_90 = sheetX['Delta_n_s_90°'].dropna()
field_90 = sheetX["fields_90°"].dropna()
n_s_90_error = sheetX["Delta_n_s_err_90°"].dropna()

# n_s_135 = sheetX['Delta_n_s_135°'].dropna()
# field_135 = sheetX["fields_135°"].dropna()
# n_s_135_error = sheetX["Delta_n_s_err_135°"].dropna()

data_folder = Path(r"Files/corrected data gabriel 04_26/amp dependence")
file_path = data_folder / 'field_dep_4_9_GHz.xlsx'

xls = pd.ExcelFile(file_path) # use r before absolute file path 
sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

# n_s_0 = sheetX['Delta_n_s_0°'].dropna()
# field_0 = sheetX["fields_0°"].dropna()
# n_s_0_error = sheetX["Delta_n_s_err_0°"].dropna()

n_s_45 = sheetX['Delta_n_s_45°'].dropna()
field_45 = sheetX["fields_45°"].dropna()
n_s_45_error = sheetX["Delta_n_s_err_45°"].dropna()

# n_s_90 = sheetX['Delta_n_s_90°'].dropna()
# field_90 = sheetX["fields_90°"].dropna()
# n_s_90_error = sheetX["Delta_n_s_err_90°"].dropna()

n_s_135 = sheetX['Delta_n_s_135°'].dropna()
field_135 = sheetX["fields_135°"].dropna()
n_s_135_error = sheetX["Delta_n_s_err_135°"].dropna()

#%%
data_folder = Path(r"./Data")

file_to_open = data_folder / "Zeeman_and_Doppler_sweep_in_Temperature_with_30_values_of_beta_in_(56.0-70.0)_and_with_q_B_values_in_(0.015-0.035).npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
beta_values = Data["beta_values"]
q_B_values = Data["q_B_values"]

Delta = 0.08
mu_B = 5.79e-2  # meV/T
gamma = 947.95796234
mu = 50.6
g_values = np.linspace(4, 6, 100)
parallel_stiffness_5_7 = np.zeros((len(B_values), len(beta_values), len(q_B_values), len(g_values)))
parallel_stiffness_4_9 = np.zeros((len(B_values), len(beta_values), len(q_B_values), len(g_values)))

Al_stiffness_5_7 = np.zeros((len(q_B_values), len(g_values)))
Al_stiffness_4_9 = np.zeros((len(q_B_values), len(g_values)))
perpendicular_stiffness_5_7 = np.zeros((len(B_values), len(beta_values), len(q_B_values), len(g_values)))
perpendicular_stiffness_4_9 = np.zeros((len(B_values), len(beta_values), len(q_B_values), len(g_values)))

best_score = float('inf')

fig, ax = plt.subplots()
ax.errorbar(field_0[:41], n_s_0[:41], yerr=n_s_0_error[:41], label=r"$n_s(0°)$", color="violet", fmt="s")
ax.errorbar(field_45[:41], n_s_45[:41], yerr=n_s_45_error[:41], label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90[:41], n_s_90[:41], yerr=n_s_90_error[:41], label=r"$n_s(90°)$", color="orange", fmt="s")
ax.errorbar(field_135[:41], n_s_135[:41], yerr=n_s_135_error[:41], label=r"$n_s(135°)$", color="darkviolet", fmt="*")

for i, beta in enumerate(beta_values):
    for j, q_B_value in enumerate(q_B_values):
        for k, g in enumerate(g_values):
            def interpolation_for_theory(x):
                return [
                        np.interp(x, B_values, superfluid_density_xx[:, i, j]),        #I have change x to x/2
                        np.interp(x, B_values, superfluid_density_yy[:, i, j]),
                        ]
            
            def model_parallel(x, a):
                return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1])/(interpolation_for_theory(0)[1] + a)# + c* x**2
            
            def model_perpendicular(x, a):
                return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])/(interpolation_for_theory(0)[0] + a) # + c* x**2

            initial_parameters_parallel = [ 112615]
            popt_parallel_5_7, pcov_parallel = curve_fit(model_parallel, field_0[:41]*0.5*mu_B*g, n_s_0[:41],
                                                      p0=initial_parameters_parallel)
            Al_stiffness_5_7[j, k] = popt_parallel_5_7[0]
            parallel_stiffness_5_7[:, i, j, k] = model_parallel(B_values, *popt_parallel_5_7)
            
            popt_parallel_4_9, pcov_parallel = curve_fit(model_parallel, field_45[:41]*0.5*mu_B*g, n_s_45[:41],
                                                      p0=initial_parameters_parallel)
            Al_stiffness_4_9[j, k] = popt_parallel_4_9[0]
            parallel_stiffness_4_9[:, i, j, k] = model_parallel(B_values, *popt_parallel_4_9)
    
            scale_factor_5_7 = 4
            scale_factor_4_9 = 2
            perpendicular_stiffness_5_7[:, i, j, k] = model_perpendicular(B_values, 1/scale_factor_5_7*popt_parallel_5_7[0])
            perpendicular_stiffness_4_9[:, i, j, k] = model_perpendicular(B_values, 1/scale_factor_4_9*popt_parallel_4_9[0])
    
            error_5_7 = np.mean((n_s_90[:41] - model_perpendicular(field_90[:41]*0.5*mu_B*g, 1/scale_factor_5_7*popt_parallel_5_7[0]))**2)
            error_4_9 = np.mean((n_s_135[:41] - model_perpendicular(field_135[:41]*0.5*mu_B*g, 1/scale_factor_4_9*popt_parallel_4_9[0]))**2)
            error = error_5_7 + error_4_9
            if error < best_score:
                best_score = error
                best_params = (beta, q_B_value, g)
                best_index = (i, j, k)

for i, beta in enumerate(beta_values):
    for j, q_B_value in enumerate(q_B_values):
        for k, g in enumerate(g_values):
            ax.plot(B_values/(0.5*mu_B*g), perpendicular_stiffness_5_7[:, i, j, k])
            ax.plot(B_values/(0.5*mu_B*g), parallel_stiffness_5_7[:, i, j, k])
            ax.plot(B_values/(0.5*mu_B*g), perpendicular_stiffness_4_9[:, i, j, k])
            ax.plot(B_values/(0.5*mu_B*g), parallel_stiffness_4_9[:, i, j, k])
ax.plot(B_values/(0.5*mu_B*g_values[best_index[2]]), perpendicular_stiffness_5_7[:, best_index[0], best_index[1], best_index[2]], "k", zorder=3)
ax.plot(B_values/(0.5*mu_B*g_values[best_index[2]]), parallel_stiffness_5_7[:, best_index[0], best_index[1], best_index[2]], "k", zorder=3)
ax.plot(B_values/(0.5*mu_B*g_values[best_index[2]]), perpendicular_stiffness_4_9[:, best_index[0], best_index[1], best_index[2]], "k", zorder=3)
ax.plot(B_values/(0.5*mu_B*g_values[best_index[2]]), parallel_stiffness_4_9[:, best_index[0], best_index[1], best_index[2]], "k", zorder=3)

B_c_1 = Delta/(2*np.sqrt(gamma*mu)*q_B_values[best_index[1]]*(0.5*mu_B*g_values[best_index[2]]) + 0.5*mu_B*g_values[best_index[2]])
B_c_2 = Delta/(2*np.sqrt(gamma*mu)*q_B_values[best_index[1]]*(0.5*mu_B*g_values[best_index[2]]) - 0.5*mu_B*g_values[best_index[2]])
plt.axvline(x=B_c_1, color='r', linestyle='--', linewidth=2)
plt.axvline(x=B_c_2, color='r', linestyle='--', linewidth=2)

ax.set_xlabel(r"$B(T)$")
ax.set_ylabel(r"$\Delta n_s$")
