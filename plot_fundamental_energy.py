#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:57:18 2026

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_folder = Path(r"./Data")

file_to_open = data_folder / "total_fundamental_energy_B=0.096_phi_x_in_(-0.001-0.001)_Delta=0.08_lambda=0_points=60_N_phi=3_N=1000_T=False_beta=50_q_D_x=4.565940020746065e-05.npz"

Data = np.load(file_to_open)
# fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = np.real(Data["fundamental_energy_2DEG"])
# fundamental_energy_Al = Data["fundamental_energy_Al"]
# normal_density = Data["normal_density"]

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
k_values = Data["k_values"]


Aluminum_constant = 0

fig, ax = plt.subplots()

# ax.plot(phi_x_values, fundamental_energy)

# ax.set_xlabel(r"$q_x$")
# ax.set_ylabel(r"$E_0$")
# ax.set_title(r"2DEG + Aluminum $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")
# plt.axvline(-q_B, linestyle="dashed", color="red")
# ax.scatter(phi_x_values[np.where(np.min(fundamental_energy)==fundamental_energy)], np.min(fundamental_energy))
# ax.text(phi_x_values[np.where(np.min(fundamental_energy)==fundamental_energy)],
#         np.min(fundamental_energy), f"{phi_x_values[np.where(np.min(fundamental_energy)==fundamental_energy)]}")

# ax.legend()

# fig, ax = plt.subplots()
# ax.plot(phi_x_values, fundamental_energy_2DEG)
# ax.plot(phi_x_values, fundamental_energy_Al)
ax.plot(phi_x_values,  fundamental_energy_2DEG)
# plt.axvline((1+1.2)*Delta/(2*np.sqrt(gamma*mu))-0.25*q_c, linestyle="dashed", color="red")
# plt.axvline((1.2-1)*Delta/(2*np.sqrt(gamma*mu))-0.25*q_c, linestyle="dashed", color="red")
# plt.axvline((1)*Delta/(2*np.sqrt(gamma*mu))-1.5*q_c, linestyle="dashed", color="red")

# # plt.axvline(-q_B, linestyle="dashed", color="red")
# ax.set_title(r"2DEG $q_B=$" + f"{q_B}" + r"$; B/\Delta=$" + f"{B/Delta}")



# fig, ax = plt.subplots()

# ax.plot(phi_x_values, (fundamental_energy_Al + fundamental_energy_2DEG)-np.min(fundamental_energy_Al + fundamental_energy_2DEG))
# ax.plot(phi_x_values, fundamental_energy_Al + fundamental_energy_2DEG )

ax.set_title(r"$B/\Delta=$" + f"{B/Delta}")
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$E_0$")
plt.show()


#%%

file_to_open = data_folder / "total_fundamental_energy_B=0.096_phi_x_in_(-0.003--0.001)_Delta=0.08_lambda=0.64_points=19_N_phi=3_N=100_T=True_beta=150.npz"

Data = np.load(file_to_open)
# fundamental_energy = Data["fundamental_energy"]
fundamental_energy_2DEG = Data["fundamental_energy_2DEG"]
# fundamental_energy_Al = Data["fundamental_energy_Al"]

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

# ax.plot(phi_x_values, (fundamental_energy_Al + fundamental_energy_2DEG)-np.min(fundamental_energy_Al + fundamental_energy_2DEG))
# ax.plot(phi_x_values, fundamental_energy_Al + fundamental_energy_2DEG )
ax.plot(phi_x_values, fundamental_energy_2DEG )


#%% Bayesian minima finding

from skopt import gp_minimize
from get_pockets import integrate_brute_force_grand_potential

# Define the search space
search_space = [(-0.0007, 0.0007)]  # Search from -3 to 3

def function(phi_x):
    integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off,
                                                                  B_x, 0, T, beta, radius_values)
    energy_phi_2DEG = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
    fundamental_energy_2DEG = energy_phi_2DEG +  np.pi/2 * cut_off**2 * (2*gamma*(phi_x+q_B)**2 - 2*mu + gamma*cut_off**2)
    return fundamental_energy_2DEG

# Bayesian Optimization setup
def objective_function(x):
    """Objective function to minimize"""
    return function(x[0])

# Run Bayesian Optimization with only 25 function evaluations!
result = gp_minimize(
    func=objective_function,
    dimensions=search_space,
    n_calls=10,                    # Only 10 expensive evaluations!
    n_initial_points=3,           # Start with 3 random points
    random_state=None,
    acq_func='LCB',  #Lower Confidence Bound (more exploratory) #"EI"  Expected Improvement
    noise=0.0,
    initial_point_generator="lhs",
    x0=[[0]]
)

# Plot all evaluation points
for i, (x_val, y_val) in enumerate(zip([xi[0] for xi in result.x_iters], result.func_vals)):
    color = 'red' if i < 3 else 'green'  # Initial points in red, BO points in green
    marker = 'o' if i < 3 else 's'
    alpha = 0.7 if i < 3 else 1.0
    ax.scatter(x_val, y_val, c=color, marker=marker, alpha=alpha, s=50)

# Mark the best point found
ax.scatter(result.x[0], result.fun, c='gold', marker='*', s=200, 
           label=f'Best found: x={result.x[0]/k_F:.5f}', edgecolors='black')

ax.text(result.x[0], result.fun, f"{result.x[0]}")
