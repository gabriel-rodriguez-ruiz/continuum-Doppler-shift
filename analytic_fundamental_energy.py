# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:09:01 2026

@author: Gabriel
"""

import numpy as np
import multiprocessing
from pathlib import Path
import scipy
from get_pockets import integrate_brute_force_grand_potential
from diagonalization import get_Energies_in_polars
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.79e-2  # meV/T

Delta = 0.08 #0.122 #0.08  #0.08  #  meV
mu = E_F   # 623 Delta #50.6  #  meV
Lambda = 15 # 15 # 8 * Delta  #0.644 meV 

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s

B = 3 * Delta #0.5 * 0.08  # meV
q_B = 0 #1e-4*k_F #0.024/8 * B  # 1/nm
theta = np.pi/2
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
N_phi = 3  # it should be odd to include zero
cut_off = 5 * k_F # 1.1 k_F
cut_off_Al = 1.1*k_F_Al   # 1.1*k_F_Al 
Delta_Al = 0.2   #0.2


phi_y = 0

T = True
beta = 50

N = 100
n_cores = 15
points = 3 * n_cores

# radius_values = np.linspace(0.98*k_F, 1.02*k_F, N)
# radius_values = [np.linspace(0.95*k_F, 0.98*k_F, N), np.linspace(0.98*k_F, 1.02*k_F, N), np.linspace(1.02*k_F, 1.05*k_F, N)]
# radius_values = [np.linspace(0.984*k_F, 0.994*k_F, N), np.linspace(0.994*k_F, 1.006*k_F, N), np.linspace(1.006*k_F, 1.016*k_F, N)]
k_1 = (-Lambda + np.sqrt(Lambda**2 
                             + 4*gamma*mu)) / (2*gamma)
k_2 = (Lambda + np.sqrt(Lambda**2
                             + 4*gamma*mu)) / (2*gamma)
radius_values = [np.linspace(0.99*k_1, 1.01*k_1, N), np.linspace(1.01*k_1,
                                     0.99*k_2, N), np.linspace(0.99*k_2, 1.01*k_2, N)]
    
parameters = {"gamma": gamma, "points": points, "k_F": k_F,
              "mu": mu, "Delta": Delta,
              "N_phi": N_phi, "Lambda": Lambda, "N": N,
              "cut_off": cut_off, "B": B, "q_B": q_B,
              "B_x": B_x, "B_y": B_y, "theta":theta,
              "gamma_Al": gamma_Al, "k_F_Al":k_F_Al, "phi_y":phi_y,
              "T": T, "beta":beta,
              "cut_off_Al": cut_off_Al, "Delta_Al": Delta_Al,
              "radius_values": radius_values
              }

def get_analytic_Energy(k, theta, q_x, q_y, mu, Delta, gamma, B):
    chi_k_minus = 2*gamma*k*(np.cos(theta)*q_x + np.sin(theta)*q_y)
    q = np.sqrt(q_x**2 + q_y**2)
    chi_k_plus = gamma *( k**2 + q**2 ) - mu

    E_plus_plus = B + chi_k_minus + np.sqrt(chi_k_plus**2 + Delta**2)
    E_minus_plus = -B + chi_k_minus + np.sqrt(chi_k_plus**2 + Delta**2) 
    E_plus_minus = B + chi_k_minus - np.sqrt(chi_k_plus**2 + Delta**2) 
    E_minus_minus = -B + chi_k_minus - np.sqrt(chi_k_plus**2 + Delta**2) 
    return np.array([E_plus_plus,
                     E_minus_plus,
                     E_plus_minus,
                     E_minus_minus])
    
#%%
import matplotlib.pyplot as plt
import scipy

def get_ground_state_energy(K, Theta, q_x, q_y, mu, Delta, gamma, B):
    Z = get_analytic_Energy(K, Theta, q_x, q_y, mu, Delta, gamma, B)
    mask = ( Z <= 0 )
    integrand = Z * mask * K
    q = np.sqrt(q_x**2 + q_y**2)
    angular_integral = scipy.integrate.trapezoid(integrand, theta_values,
                                                 axis=1)
    radial_integral = scipy.integrate.trapezoid(angular_integral, k_values,
                                                axis=1)
    return (1/2*np.sum(radial_integral)
            +  np.pi/2 * cut_off**2 * (2*gamma*q**2 - 2*mu + gamma*cut_off**2))

def check_cut_off(K, Theta, mu, gamma):
    # It should return 0
    Delta = 0
    q = 0
    B = 0
    return get_ground_state_energy(K, Theta, q, mu, Delta, gamma, B) - 2*np.pi*k_F**2*(gamma*k_F**2/2 - mu)

def get_ground_state_energy_vs_q(K, Theta, q_values, varphi, mu, Delta, gamma, B):
    E_q = np.zeros_like(q_values)
    for i, q in enumerate(q_values):
        q_x = q * np.cos(varphi)
        q_y = q * np.sin(varphi)
        E_q[i] = get_ground_state_energy(K, Theta, q_x, q_y, mu, Delta, gamma, B)
    return E_q

def get_ground_state_energy_vs_q_with_Doppler(K, Theta, q_values, varphi, mu, Delta, gamma, B, q_D):
    E_q = np.zeros_like(q_values)
    for i, q in enumerate(q_values):
        q_x = q * np.cos(varphi) + q_D
        q_y = q * np.sin(varphi)
        E_q[i] = get_ground_state_energy(K, Theta, q_x, q_y, mu, Delta, gamma, B)
    return E_q

#%% Pockets
cut_off = 2*k_F
N = 1000
# k_values = np.linspace(0, cut_off, 3000)
Delta = 0.08
B = 0*Delta
varphi = 0
k_values = np.append(np.linspace(0, 0.99*k_F, N), [np.linspace(0.99*k_F, 1.01*k_F, N),
            np.linspace(1.01*k_F, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, 1000)
q_c = Delta/(2*np.sqrt(mu*gamma))
q_values = np.linspace(0, 6*q_c, 100)
K, Theta = np.meshgrid(k_values, theta_values)

q_x = 1.2*q_c
q_y = 0
Z = get_analytic_Energy(K, Theta, q_x, q_y, mu, Delta, gamma, B)

def get_k_plus(theta):
    k_plus = 1/np.sqrt(gamma) * np.sqrt(mu +
                    np.sqrt((2*gamma*k_F*q_x*np.cos(theta))**2 + Delta**2))
    return k_plus

def get_k_minus(theta):
    k_minus = 1/np.sqrt(gamma) * np.sqrt(mu -
                    np.sqrt((2*gamma*k_F*q_x*np.cos(theta))**2 + Delta**2))
    return k_minus


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

ax.contour(Theta, K/k_F, Z[1, :], colors='black', levels=[0],
           linestyles="dashed")
ax.contour(Theta, K/k_F, Z[3, :], colors='black', levels=[0])
ax.set_ylim([0.997, 1.003])
ax.set_yticks([0.998, 1, 1.002], [r"$0.998k_F$" , r"$k_F$", r"$1.002k_F$"])
ax.set_rlabel_position(0)
# phi_1 = np.arccos(Delta/(2*gamma*k_F*q_x))
# phi_2 = np.arccos(-Delta/(2*gamma*k_F*q_x))
# phi_1_values = np.linspace(-phi_1, phi_1)
# phi_2_values = np.linspace(phi_2, 2*np.pi-phi_2)
# ax.plot(phi_1_values, get_k_plus(phi_1_values)/k_F, color="red")
# ax.plot(phi_2_values, get_k_plus(phi_2_values)/k_F, color="red", linestyle="dashed")
# ax.plot(phi_1_values, get_k_minus(phi_1_values)/k_F, color="red")
# ax.plot(phi_2_values, get_k_minus(phi_2_values)/k_F, color="red", linestyle="dashed")
# ax.plot(phi_1*np.ones(2), [get_k_minus(phi_1)/k_F,  get_k_plus(phi_1)/k_F],
#         color="red")
# ax.plot(-phi_1*np.ones(2), [get_k_minus(-phi_1)/k_F,  get_k_plus(-phi_1)/k_F],
#         color="red")
# ax.plot(phi_2*np.ones(2), [get_k_minus(phi_2)/k_F,  get_k_plus(phi_2)/k_F],
#         color="red", linestyle="dashed")
# ax.plot((2*np.pi-phi_2)*np.ones(2), [get_k_minus(-phi_2)/k_F,  get_k_plus(-phi_2)/k_F],
#         color="red", linestyle="dashed")

#%% Ground state energy



cut_off = 2*k_F
N = 1000
# k_values = np.linspace(0, cut_off, 3000)
k_values = np.append(np.linspace(0, 0.99*k_F, N), [np.linspace(0.99*k_F, 1.01*k_F, N),
            np.linspace(1.01*k_F, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, N)
q_c = Delta/(2*np.sqrt(mu*gamma))
q_values = np.sort(np.append(np.linspace(-3*q_c, 3*q_c), [0]))
K, Theta = np.meshgrid(k_values, theta_values)

Delta = 0.08
B = 0.*Delta
varphi = 0
# E_q = get_ground_state_energy_vs_q(K, Theta, q_values, varphi, mu, Delta, gamma, B)
q_D_values = [0, 0.5*q_c, 1.2*q_c]

fig, ax = plt.subplots()
# ax.plot(q_values, E_q, label=r"$q_D=0$")
# ax.plot(q_values - 0.5*q_c, E_q, label=r"$q_D=0.5q_c$")
# ax.plot(q_values - 1.2*q_c, E_q, label=r"$q_D=1.2q_c$")
E_q = np.zeros((3, len(q_values)))

for i, q_D in enumerate(q_D_values):
    s = [r"$0$", r"$0.5q_c$", r"$1.2q_c$"]
    E_q[i, :] = get_ground_state_energy_vs_q_with_Doppler(K, Theta, q_values, varphi, mu, Delta, gamma, B, q_D)
    ax.plot(q_values, E_q[i, :], label=r"$q_D=$" + f"{s[i]}")

ax.scatter(q_c, get_ground_state_energy(K, Theta, q_c, 0, mu, Delta, gamma, B),
           color="C0")
ax.scatter(-q_c, get_ground_state_energy(K, Theta, -q_c, 0, mu, Delta, gamma, B),
           color="C0")
ax.scatter(q_c- 0.5*q_c, get_ground_state_energy(K, Theta, q_c, 0, mu, Delta, gamma, B),
           color="C1")
ax.scatter(-q_c- 0.5*q_c, get_ground_state_energy(K, Theta, -q_c, 0, mu, Delta, gamma, B),
           color="C1")
ax.scatter(q_c-1.2*q_c, get_ground_state_energy(K, Theta, q_c, 0, mu, Delta, gamma, B),
           color="C2")
ax.scatter(-q_c-1.2*q_c, get_ground_state_energy(K, Theta, -q_c, 0, mu, Delta, gamma, B),
           color="C2")
ax.set_xlim([-2.5*q_c, 2.5*q_c])
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$E_0(q_x)$")

plt.axvline(x=0, color='black', linestyle='--')

ax.legend()

fig, ax = plt.subplots()
ax.plot(q_values, np.gradient(np.gradient(E_q[0, :], np.diff(q_values)[0]), np.diff(q_values)[0]))



#%% Ground state energy in the parallel direction



cut_off = 2*k_F
N = 1000
# k_values = np.linspace(0, cut_off, 3000)
k_values = np.append(np.linspace(0, 0.99*k_F, N), [np.linspace(0.99*k_F, 1.01*k_F, N),
            np.linspace(1.01*k_F, cut_off, N)])
theta_values = np.linspace(0, 2*np.pi, 1000)
q_c = Delta/(2*np.sqrt(mu*gamma))
q_values = np.sort(np.append(np.linspace(-3*q_c, 3*q_c), [0]))
K, Theta = np.meshgrid(k_values, theta_values)

Delta = 0.08
B = 0*Delta
varphi = np.pi/2
q_D_values = [0, 0.5*q_c, 1.2*q_c]

fig, ax = plt.subplots()
E_q_parallel = np.zeros((3, len(q_values)))

for i, q_D in enumerate(q_D_values):
    s = [r"$0$", r"$0.5q_c$", r"$1.2q_c$"]
    E_q_parallel[i, :] = get_ground_state_energy_vs_q_with_Doppler(K, Theta, q_values, varphi, mu, Delta, gamma, B, q_D)
    ax.plot(q_values, E_q_parallel[i, :], label=r"$q_D=$" + f"{s[i]}")
ax.legend()
ax.set_xlabel(r"$q_y$")
ax.set_ylabel(r"$E_0$")

#%% Stiffness parallel

q_D_values = np.linspace(0, 3*q_c)
h = 1e-5
q_y_values = [-h, 0, h]
E_q_D = np.zeros((3, len(q_D_values)))
B = 0.*Delta

# fig, ax = plt.subplots()
for i, q_y in enumerate(q_y_values):
    E_q_D[i, :] = get_ground_state_energy_vs_q_with_Doppler(K, Theta, q_D_values, np.pi/2, mu, Delta, gamma, B, q_y)
D_perp = (E_q_D[2, :] - 2*E_q_D[1, :] + E_q_D[0, :])/h**2
fig, ax = plt.subplots()
ax.plot(q_D_values, D_perp)
ax.set_xticks([0, q_c, 2*q_c, 3*q_c], [r"$0$", r"$q_c$", r"$2q_c$", r"$3q_c$"])
plt.grid()
plt.show()