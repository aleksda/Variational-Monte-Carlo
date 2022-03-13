import matplotlib.pyplot as plt
import numpy as np
from samplers import MetropolisVMC2
from wavefunction import SimpleGaussian


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


N = 500
d = 3
omega = 1
wf = SimpleGaussian(N, d, omega)

exact_E = exact_energy(N, d, omega)
print(f"Exact energy: {exact_E}")

vmc_sampler = MetropolisVMC2(wf)
ncycles = 10000
alpha_step = 0.1
alphas = np.arange(0.1, 1 + alpha_step, alpha_step)


energies = np.zeros(alphas.size)

#fig, ax = plt.subplots()

for i, alpha in enumerate(alphas):
    initial_state = safe_initial_state(wf, alpha)
    energies[i] = vmc_sampler.sample(ncycles,
                                     initial_state,
                                     alpha,
                                     scale=0.5,
                                     burn=0,
                                     tune=True,
                                     tune_iter=8000,
                                     tune_interval=250,
                                     optimize=False,
                                     tol_scale=1e-5
                                     )
    accept_rate = vmc_sampler.accept_rate
    print(f"{alpha=:.2f}, E={energies[i]:.2f}, {accept_rate=:.2f}")
    #ax.plot(vmc_sampler.energy_samples, label=f'{alpha=:.1f}')
    # ax.legend()

E_min = np.min(energies)
alpha_min = alphas[np.argmin(energies)]
print(f"{alpha_min=:.2f}, {E_min=:.2f}")

fig, ax = plt.subplots()
ax.plot(alphas, energies, label='VMC')
ax.axhline(exact_E, ls='--', color='r', label='Exact')
ax.set(xlabel=r'$\alpha$', ylabel='Energy')
ax.legend()
plt.show()
