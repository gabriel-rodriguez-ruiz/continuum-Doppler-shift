#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:00:51 2026

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from diagonalization import get_Energies_in_polars, get_Energies, get_Eigenvectors_in_polars
import scipy.optimize

c = 3e17 # nm/s  #3e8 # m/s
m_e =  5.1e8 / c**2 # meV s²/(nm)²
m = 0.0403 * m_e # meV s²/(nm)²
hbar = 6.58e-13 # meV s
gamma = hbar**2 / (2*m) # meV (nm)²
E_F = 50.6 #50.6 # meV
k_F = np.sqrt(E_F / gamma ) # 1/nm
v_F = hbar*k_F/m * 1e-9  # m/s
mu_B = 5.788e-2 # meV/TT


Delta =  0.08  #2*0.122 # 0.08 #0.08   #  meVs
mu = E_F  # 623 Delta #50.6  #  meV
# gamma = 9479 # meV (nm)²
Lambda = 15  #15 #187*Delta/2 # meV*nm    # 8 * Delta  #0.644 meV 
theta = np.pi/2

B = 3*Delta   #0.28*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
theta_values = np.array([0])
q_B_constant = 0 #0.024/8
phi_x = 0  #1e-3 * k_F #q_B_constant * B  #0.0004  #0.024 * 0.5 * Delta
phi_y = 0

m_Al = 1.4 * m_e # meV s²/(nm)²
gamma_Al = hbar**2 / (2*m_Al) # meV (nm)²
k_F_Al = np.sqrt(E_F / gamma_Al ) # 1/nm
v_F_Al = hbar*k_F_Al/m_Al * 1e-9  # m/s
Delta_Al = 0

k_values = np.linspace(0.9*k_F, 1.1*k_F, 100)

E = get_Energies_in_polars(k_values, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)

def f(x):
    return get_Energies_in_polars(x, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[:,0,2]

def g(x):
    return get_Energies_in_polars(x, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[:,0,0]

def h(x):
    return get_Energies_in_polars(x, theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[:,0,3]

# 1. Definir los intervalos (brackets) donde hay cambio de signo
x_range = k_values
f_vals = f(x_range)
idx = np.where(np.diff(np.sign(f_vals)))[0]

a = x_range[idx]      # Límites inferiores (arreglo)
b = x_range[idx + 1]  # Límites superiores (arreglo)

# 2. Bisección Vectorizada (Máximo rendimiento)
def vectorized_bisect(func, a, b, tol=1e-7, max_iter=100):
    for _ in range(max_iter):
        mid = (a + b) / 2
        f_mid = func(mid)
        
        # Si f(a) * f(mid) < 0, la raíz está en [a, mid], si no en [mid, b]
        # Usamos np.where para actualizar todos los intervalos a la vez
        condition = np.sign(func(a)) * np.sign(f_mid) < 0
        a = np.where(condition, a, mid)
        b = np.where(condition, mid, b)
        
        # Criterio de parada: si la diferencia es menor a la tolerancia
        if np.max(np.abs(b - a)) < tol:
            break
    return (a + b) / 2

# Ejecución
roots = vectorized_bisect(f, a, b)

x_range = k_values
g_vals = g(x_range)
idx = np.where(np.diff(np.sign(g_vals)))[0]

a = x_range[idx]      # Límites inferiores (arreglo)
b = x_range[idx + 1]  # Límites superiores (arreglo)
roots_g = vectorized_bisect(g, a, b)

x_range = k_values
h_vals = h(x_range)
idx = np.where(np.diff(np.sign(h_vals)))[0]

a = x_range[idx]      # Límites inferiores (arreglo)
b = x_range[idx + 1]  # Límites superiores (arreglo)
roots_h = vectorized_bisect(h, a, b)

print(f"Raíces encontradas: {roots}")

fig, ax = plt.subplots()
ax.plot(k_values/k_F, E[:, 0], marker="v", markersize=3)
# ax.plot(x_range/k_F, f(x_range), "-")
# ax.plot(x_range/k_F, g(x_range), "-", linewidth=5)

ax.scatter(roots/k_F, [0, 0])
ax.scatter(roots_g/k_F, np.zeros_like(roots_g))
ax.scatter(roots_h/k_F, np.zeros_like(roots_h))


from scipy.optimize import root_scalar
f_values = f(x_range)

# Identificar dónde cambia el signo de la función
sign_changes = np.where(np.diff(np.sign(f_values)))[0]
brackets = [(x_range[i], x_range[i+1]) for i in sign_changes]

roots = []
for bracket in brackets:
    g = lambda x: get_Energies_in_polars([x], theta_values, mu, B_y, Delta, phi_x, gamma, Lambda, B_x, phi_y)[:,0,2][0]
    sol = root_scalar(g, bracket=bracket, method='brentq')
    if sol.converged:
        roots.append(sol.root)

ax.scatter(roots/k_F, np.zeros_like(roots))
