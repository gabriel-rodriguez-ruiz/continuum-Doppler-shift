#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:07:49 2025

@author: gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
from get_pockets import integrate_brute_force_grand_potential, integrate_brute_force_current_x, integrate_brute_force_current_y
from skopt import gp_minimize


c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/T

mu = E_F   # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 15  #15 # meV*nm    # 8 * Delta  #0.644 meV 

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

Aluminum_constant = 0 #20000  #3737
q_B_constant = 0 #0.024/8

N_phi = 3  #101  # 101  # it should be odd to include zero
h = 1e-5*k_F

phi_x_values = np.linspace(-h, h, N_phi)      # np.array([-h, 0, h])     #np.linspace(-0.002 * k_F, 0.002 * k_F, N_phi)   #np.linspace(-0.003 * k_F, 0.003 * k_F, N_phi)
cut_off = 2 * 1.1*k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al # 1.1 k_F

theta = np.pi/2   # float

N = 100 #300 #100  #514   #300
n_cores = 19
points = 3 * n_cores

T = True
beta = 40
Delta_0 = 0.08
Delta_S = 0.2
Delta = 1/(2.5) * Delta_S * np.tanh(Delta_S*beta/2)

# radius_values = [np.linspace(0.95*k_F, 0.97*k_F, N), np.linspace(0.97*k_F, 1.02*k_F, N), np.linspace(1.02*k_F, 1.05*k_F, N)]
k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
radius_values = [np.linspace(0.99*k_1, 1.01*k_1, N), np.linspace(1.01*k_1,
                                     0.99*k_2, N), np.linspace(0.99*k_2, 1.01*k_2, N)]


parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta, "phi_x_values": phi_x_values,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "T": T, "beta": beta,
              "gamma_Al": gamma_Al, "k_F_Al": k_F_Al,
              "Aluminum_constant": Aluminum_constant
              }



def integrate_B(B):
    B_x =  B * np.cos(theta)
    B_y =  B * np.sin(theta)
    q_B = q_B_constant * B   # * np.cos(np.pi/2 - theta)
    # q_B_y = 0.024 * B * np.sin(np.pi/2 - theta)
    search_space = [(-0.0003, 0.0003)]
    
    #normal density
    # integral, low_integral, high_integral, normal_density = integrate_brute_force_grand_potential(N, mu, B_y, Delta, 0, gamma, Lambda, k_F, cut_off,
    #                                                                   B_x, 0, T, beta, radius_values)    
    
    def function(phi_x):
        integral, low_integral, high_integral, normal_density = integrate_brute_force_grand_potential(N, mu, B_y, Delta, phi_x + q_B, gamma, Lambda, k_F, cut_off,
                                                                      B_x, 0, T, beta, radius_values)
        energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        fundamental_energy_2DEG = energy_phi + np.pi/2 * cut_off**2 * (2*gamma*(phi_x + q_B)**2 - 2*mu + gamma*cut_off**2)
        
        # integral, low_integral, high_integral = integrate_brute_force_grand_potential(N, mu, B_y, 0.2, phi_x, gamma_Al, 0, k_F_Al, cut_off_Al,
        #                                                               B_x, 0, T, beta)
        # energy_phi = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        # fundamental_energy_Al = energy_phi + np.pi/2 * cut_off_Al**2 * (2*gamma_Al*phi_x**2 - 2*mu + gamma_Al*cut_off_Al**2)
        return fundamental_energy_2DEG + Aluminum_constant * phi_x**2    

    def get_minima(search_space):
        def objective_function(x):
            """Objective function to minimize"""
            return function(x[0])
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
        return result.x[0]
    
    q_eq = 0 #get_minima(search_space)
    
    current_phi = np.zeros_like(phi_x_values)
    current_phi_Al = np.zeros_like(phi_x_values)
    
    for j, phi_x in enumerate(phi_x_values):
        print(j)
        integral, low_integral, high_integral = integrate_brute_force_current_x(N, mu, B_y, Delta, phi_x + q_B + q_eq, gamma, Lambda, k_F, cut_off, B_x, phi_y=0, T=T, beta=beta, h=h, radius_values=radius_values)    # phi_y=-phi_x if magnetic field is at 45º
        current_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        # integral, low_integral, high_integral = integrate_brute_force_current_x(N, mu, B_y, 0.2, phi_x + q_eq, gamma_Al, 0, k_F_Al, cut_off_Al, B_x, phi_y=0, T=T, beta=beta, h=h)    # phi_y=-phi_x if magnetic field is at 45º
        # current_phi_Al[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        current_phi_Al[j] = 2 * Aluminum_constant * ( phi_x + q_eq )
    total_current = current_phi + 2*np.pi * cut_off**2 * gamma*(phi_x_values + q_B + q_eq) + current_phi_Al #+ 2*np.pi * cut_off_Al**2 * gamma_Al*phi_x_values
    
    superfluid_density_xx = (total_current[2] - total_current[0])/(2*h)
            
    phi_y_values = np.array([-h, 0, h])
    current_phi = np.zeros_like(phi_y_values)
    current_phi_Al = np.zeros_like(phi_y_values)
    current_phi_x = np.zeros_like(phi_y_values)

    for j, phi_y in enumerate(phi_y_values):
        print(N_phi+j)
        integral, low_integral, high_integral = integrate_brute_force_current_y(N, mu, B_y, Delta, q_eq, gamma, Lambda, k_F, cut_off, B_x, phi_y, T=T, beta=beta, h=h, radius_values=radius_values)
        current_phi[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        
        # integral, low_integral, high_integral = integrate_brute_force_current_x(N, mu, B_y, Delta, q_eq, gamma, Lambda, k_F, cut_off, B_x, phi_y, T=T, beta=beta, h=h, radius_values=radius_values)
        # current_phi_x[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        # integral, low_integral, high_integral = integrate_brute_force_current_y(N, mu, B_y, 0.2, q_eq, gamma_Al, 0, k_F_Al, cut_off_Al, B_x, phi_y, T=T, beta=beta, h=h)
        # current_phi_Al[j] = np.sum(integral) + np.sum(low_integral) + np.sum(high_integral)
        current_phi_Al[j] = 2 * Aluminum_constant * phi_y
    total_current = current_phi + 2*np.pi * cut_off**2 * gamma*phi_y_values + current_phi_Al              #+ 2*np.pi * cut_off_Al**2 * gamma_Al*phi_y_values
    # total_current_xy = current_phi_x + current_phi_Al              #+ 2*np.pi * cut_off_Al**2 * gamma_Al*phi_y_values

    superfluid_density_yy = (total_current[2]-total_current[0])/(2*h)
    # superfluid_density_xy = (total_current_xy[2]-total_current_xy[0])/(2*h)
    return superfluid_density_xx, superfluid_density_yy, q_eq

if __name__ == "__main__":
    B_values = np.linspace(0.*Delta, 3*Delta, points)
    integrate = integrate_B
    B_direction = f"{theta:.2}"
    # integrate = integrate_B_y
    with multiprocessing.Pool(n_cores) as pool:
        superfluid_density_xx, superfluid_density_yy, q_eq = zip(*pool.map(integrate, B_values))
    superfluid_density_xx = np.array(superfluid_density_xx)
    superfluid_density_yy = np.array(superfluid_density_yy)
    # superfluid_density_xy = np.array(superfluid_density_xy)
    q_eq = np.array(q_eq)
    data_folder = Path("Data/")
    name = f"superfluid_density_with_Doppler_shift_B_in_{B_direction}_({np.round(np.min(B_values/Delta),3)}-{np.round(np.max(B_values/Delta),3)})_phi_x_in_({np.round(np.min(phi_x_values/k_F), 3)}-{np.round(np.max(phi_x_values/k_F),3)})_Delta={Delta}_lambda={np.round(Lambda, 2)}_points={points}_N_phi={N_phi}_N={N}_T={T}_beta={beta}_m={m}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open,
             superfluid_density_xx=superfluid_density_xx,
             superfluid_density_yy=superfluid_density_yy,
             q_eq=q_eq,
             #superfluid_density_xy=superfluid_density_xy,
             B_values=B_values, **parameters)
    print("\007")

