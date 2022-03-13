import matplotlib.pyplot as plt
import numpy as np
from samplers import MetropolisVMC2
from wavefunction import HOWF


def exact_energy(n_particles, dim, omega):
    return (omega * dim * n_particles) / 2


def safe_initial_state(wavefunction, alpha, seed=None):
    rng = np.random.default_rng(seed=seed)
    N = wavefunction.n_particles
    dim = wavefunction.dim
    positions = rng.random(size=(N, dim))

    # safe initialization
    wf2 = wavefunction.density(positions, alpha)
    while np.sum(wf2) <= 1e-14:
        positions *= 0.5
        wf2 = wavefunction.density(positions, alpha)

    return positions


N = 1
d = 1
omega = 1
wf = HOWF(N, d, omega)

exact_E = exact_energy(N, d, omega)
print(f"Exact energy: {exact_E}")

vmc_sampler = MetropolisVMC2(wf)
ncycles = int(1e4)
initial_alpha = 0.7
eta = 0.001  # learning rate

initial_state = safe_initial_state(wf, initial_alpha)
energy = vmc_sampler.sample(ncycles,
                            initial_state,
                            initial_alpha,
                            scale=1.,
                            burn=0,
                            tune=True,
                            tune_iter=5000,
                            tune_interval=250,
                            tol_scale=1e-5,
                            optimize=True,
                            max_iter=100000,
                            batch_size=1000,
                            gradient_method='gdmom',
                            eta=eta,
                            tol_optim=1e-8
                            )

print("Optimal scale:", vmc_sampler.scale)
print("Optimal alpha:", vmc_sampler.alpha)
print("Sampler accept rate:", vmc_sampler.accept_rate)
print("VMC energy:", energy)
