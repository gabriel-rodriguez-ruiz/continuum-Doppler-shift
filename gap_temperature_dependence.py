#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:00:56 2026

@author: gabriel
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# --- Parámetros del Material (Ejemplo: Aluminio) ---
Tc = 1.86        # Temperatura crítica (K)
kb = 8.617e-5    # Constante de Boltzmann (eV/K)
delta_0 = 1.764 * kb * Tc  # Gap a T=0 según BCS
wd = 10.0 * kb * Tc        # Frecuencia de Debye aproximada (límite de integración)

def bcs_integral(delta, T):
    """Ecuación integral BCS: res - 1 = 0"""
    if T == 0: return 0 # Manejo manual para T=0
    
    # Integrando: tanh(E / 2kbT) / E  donde E = sqrt(xi^2 + delta^2)
    integrand = lambda xi: (np.tanh(np.sqrt(xi**2 + delta**2) / (2 * kb * T)) / 
                            np.sqrt(xi**2 + delta**2))
    
    integral, _ = quad(integrand, 0, wd)
    
    # La condición de consistencia BCS simplificada:
    # integral(0->wd) [tanh(E/2kT)/E] dxi = 1/V*N(0)
    # A T=0, delta=delta_0, por lo que determinamos 1/VN(0) constante:
    const_vn0 = quad(lambda xi: 1/np.sqrt(xi**2 + delta_0**2), 0, wd)[0]
    
    return integral - const_vn0

def calcular_gap_temperatura(T_array):
    gaps = []
    gap_guess = delta_0 # Valor inicial para el solver
    
    for T in T_array:
        if T >= Tc:
            gaps.append(0)
        elif T < 0.01: # Cerca de T=0
            gaps.append(delta_0)
        else:
            # Resolvemos para encontrar el gap delta que hace la integral = cte
            sol = fsolve(bcs_integral, gap_guess, args=(T))
            gap_actual = max(0, sol[0])
            gaps.append(gap_actual)
            gap_guess = gap_actual # Usar el resultado previo para acelerar convergencia
            
    return np.array(gaps)

# --- Ejecución y Gráfico ---
T_vals = np.linspace(0.01, Tc + 0.1, 50)
gap_vals = calcular_gap_temperatura(T_vals)

plt.figure(figsize=(8, 5))
plt.plot(T_vals/Tc, gap_vals/delta_0, 'o-', label='Cálculo Numérico BCS')
plt.axvline(1, color='r', linestyle='--', label='$T = T_c$')
plt.title("Dependencia del Gap con la Temperatura (Modelo BCS)")
plt.xlabel("Temperatura Normalizada ($T/T_c$)")
plt.ylabel("Gap Normalizado ($\Delta(T)/\Delta_0$)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
