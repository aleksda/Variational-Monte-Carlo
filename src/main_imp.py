#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from samplers import ImportanceVMC
from wavefunction import SimpleGaussian


def exact_energy(n_particles, dim, omega):
    """
    Minimal local energy found by setting alpha = omega / 2, which yields
    E_L = (omega * dim * n_particles) / 2
    """
    return (omega * dim * n_particles) / 2


N = 500
d = 3
omega = 1
wf = SimpleGaussian(N, d, omega)

exact_E = exact_energy(N, d, omega)
print(f"Exact energy: {exact_E}")

# Importance VMC
dt = 0.0006
#dt = 0.5 / np.sqrt(N)
ncycles = 30000

imp_sampler = ImportanceVMC(wf)
alpha_step = 0.05
alphas = np.arange(0.1, 1. + alpha_step, alpha_step)
energies, variances = imp_sampler.sample(ncycles,
                                         alphas,
                                         dt=dt
                                         )


E_min = np.min(energies)
alpha_min = alphas[np.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
