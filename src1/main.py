#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import numpy as np
from samplers import Metropolis2VMC, MetropolisHastingsVMC, MetropolisVMC, MetropolisOptimizerVMC
from wavefunction import JaxWaveFunction, SimpleGaussian


def exact_energy(n_particles, dim, omega):
    """
    Minimal local energy found by setting alpha = omega / 2, which yields
    E_L = (omega * dim * n_particles) / 2
    """
    return (omega * dim * n_particles) / 2


N = 5
d = 1
dt = 0.5/(np.sqrt(N))
omega = 1
wf = SimpleGaussian(N, d, omega)

exact_E = exact_energy(N, d, omega)
ncycles = 30000
alpha = 0.2
print(f"Exact energy: {exact_E}")
"""

vmc_sampler = MetropolisHastingsVMC(wf)


initial_time = time.time()
energies, variances = vmc_sampler.sample(ncycles,
                                         alphas,
                                         dt=dt)
final_time = time.time()


E_min = np.min(energies)
alpha_min = alphas[np.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")
print("Time spent metropolis hastings: ", final_time - initial_time)

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
"""
vmc_sampler = MetropolisOptimizerVMC(wf)
initial_time = time.time()
energy, variance = vmc_sampler.sample(ncycles,
                                         alpha,
                                         scale=0.5,
                                         tune=True,
                                         tune_iter=5000,
                                         tune_interval=250,
                                         optimize=True,
                                         optim_iter=10000,
                                         optim_runs=100,
                                         learning_rate = 1e-3,
                                         tolerance = 1e-12,
                                         momentum = 0.0,
                                         )
final_time = time.time()
print("Final energy: ", energy)
print("Final variance: ", variance)
"""
E_min = np.min(energies)
alpha_min = alphas[np.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")
print("Time spent metropolis: ", final_time-initial_time)

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
"""
