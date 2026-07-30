# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:43:31 2026

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

file_to_open = data_folder / "Zeeman_sweep_in_Temperature_with_15_values_of_beta_in_(30.0-100.0).npz"

Data = np.load(file_to_open)
B_values = Data["B_values"]
superfluid_density_xx = Data["superfluid_density_xx"]
superfluid_density_yy = Data["superfluid_density_yy"]
beta_values = Data["beta_values"]

Delta = 0.08
B_c_values = np.linspace(0.02, 0.1, 100)
parallel_stiffness_5_7 = np.zeros((len(B_values), len(beta_values), len(B_c_values)))
parallel_stiffness_4_9 = np.zeros((len(B_values), len(beta_values), len(B_c_values)))

Al_stiffness_5_7 = np.zeros(len(B_c_values))
Al_stiffness_4_9 = np.zeros(len(B_c_values))
perpendicular_stiffness_5_7 = np.zeros((len(B_values), len(beta_values), len(B_c_values)))
perpendicular_stiffness_4_9 = np.zeros((len(B_values), len(beta_values), len(B_c_values)))

best_score = float('inf')

fig, ax = plt.subplots()
ax.errorbar(field_0[:41], n_s_0[:41], yerr=n_s_0_error[:41], label=r"$n_s(0°)$", color="violet", fmt="s")
ax.errorbar(field_45[:41], n_s_45[:41], yerr=n_s_45_error[:41], label=r"$n_s(45°)$", color="green", fmt="*")
ax.errorbar(field_90[:41], n_s_90[:41], yerr=n_s_90_error[:41], label=r"$n_s(90°)$", color="orange", fmt="s")
ax.errorbar(field_135[:41], n_s_135[:41], yerr=n_s_135_error[:41], label=r"$n_s(135°)$", color="darkviolet", fmt="*")

for i, beta in enumerate(beta_values):
    for j, B_c in enumerate(B_c_values):
        def interpolation_for_theory(x):
            return [
                    np.interp(x, B_values/Delta, superfluid_density_xx[:, i]),        #I have change x to x/2
                    np.interp(x, B_values/Delta, superfluid_density_yy[:, i]),
                    ]
        def model_parallel(x, a):
            return (interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1])/(interpolation_for_theory(0)[1] + a) #+ c* x**2
        def model_perpendicular(x, a):
            return (interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])/(interpolation_for_theory(0)[0] + a) #+ c* x**2
    
        initial_parameters_parallel = [ 70278]
        popt_parallel_5_7, pcov_parallel = curve_fit(model_parallel, field_0[:41]/B_c, n_s_0[:41],
                                                  p0=initial_parameters_parallel)
        Al_stiffness_5_7[j] = popt_parallel_5_7[0]
        parallel_stiffness_5_7[:, i, j] = model_parallel(B_values/Delta, *popt_parallel_5_7)
        
        popt_parallel_4_9, pcov_parallel = curve_fit(model_parallel, field_45[:41]/B_c, n_s_45[:41],
                                                  p0=initial_parameters_parallel)
        Al_stiffness_4_9[j] = popt_parallel_4_9[0]
        parallel_stiffness_4_9[:, i, j] = model_parallel(B_values/Delta, *popt_parallel_4_9)

        scale_factor_5_7 = 4
        scale_factor_4_9 = 2
        perpendicular_stiffness_5_7[:, i, j] = model_perpendicular(B_values/Delta, 1/scale_factor_5_7*popt_parallel_5_7[0])
        perpendicular_stiffness_4_9[:, i, j] = model_perpendicular(B_values/Delta, 1/scale_factor_4_9*popt_parallel_4_9[0])

        error_5_7 = np.mean((n_s_90[:41] - model_perpendicular(field_90[:41]/B_c, 1/scale_factor_5_7*popt_parallel_5_7[0]))**2)
        error_4_9 = np.mean((n_s_135[:41] - model_perpendicular(field_135[:41]/B_c, 1/scale_factor_4_9*popt_parallel_4_9[0]))**2)
        error = error_5_7 + error_4_9
        if error < best_score:
            best_score = error
            best_params = (beta, B_c)
            best_index = (i, j)

for i, beta in enumerate(beta_values):
    for j, B_c in enumerate(B_c_values):
        ax.plot(B_values/Delta*B_c, perpendicular_stiffness_5_7[:, i, j])
        ax.plot(B_values/Delta*B_c, parallel_stiffness_5_7[:, i, j])
        ax.plot(B_values/Delta*B_c, perpendicular_stiffness_4_9[:, i, j])
        ax.plot(B_values/Delta*B_c, parallel_stiffness_4_9[:, i, j])
ax.plot(B_values/Delta*B_c_values[best_index[1]], perpendicular_stiffness_5_7[:, best_index[0], best_index[1]], "k", zorder=3)
ax.plot(B_values/Delta*B_c_values[best_index[1]], parallel_stiffness_5_7[:, best_index[0], best_index[1]], "k", zorder=3)
ax.plot(B_values/Delta*B_c_values[best_index[1]], perpendicular_stiffness_4_9[:, best_index[0], best_index[1]], "k", zorder=3)
ax.plot(B_values/Delta*B_c_values[best_index[1]], parallel_stiffness_4_9[:, best_index[0], best_index[1]], "k", zorder=3)

B_c = B_c_values[[best_index[1]]]
mu_B = 5.79e-2  # meV/T
g = 2 * 0.08 / (mu_B*B_c)

ax.set_xlabel(r"$B(T)$")
ax.set_ylabel(r"$\Delta n_s$")
