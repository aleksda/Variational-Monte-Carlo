from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.config import config

config.update("jax_enable_x64", True)


class JaxWF:

    def __init__(self, n_particles, dim, omega):
        # Error handling
        if not isinstance(n_particles, int):
            msg = "The number of particles in the system must be passed as int"
            raise TypeError(msg)
        if not isinstance(dim, int):
            msg = "The dimensionality of the system must be passed as int"
            raise TypeError(msg)
        if not n_particles > 0:
            msg = "The number of particles must be > 0"
            raise ValueError(msg)
        if not 1 <= dim <= 3:
            msg = "Dimensionality must be between 1D, 2D or 3D"
            raise ValueError(msg)

        self._N = n_particles
        self._d = dim
        self._omega = omega

    @partial(jit, static_argnums=(0,))
    def __call__(self, r, alpha):
        """Evaluate the trial wave function.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Evaluated trial wave function
        """
        return jnp.exp(-alpha * jnp.sum(r**2))

    @partial(jit, static_argnums=(0,))
    def evaluate(self, r, alpha):
        return jnp.exp(-alpha * jnp.sum(r**2))

    @partial(jit, static_argnums=(0,))
    def local_energy(self, r, alpha):
        """Compute the local energy.

        Parameters
        ----------
        r : np.ndarray, shape=(n_particles, dim)
            Particle positions
        alpha : float
            Variational parameter

        Returns
        -------
        float
            Computed local energy
        """
        E_L = self._N * self._d * alpha + \
            (0.5 * self._omega**2 - 2 * alpha**2) * jnp.sum(r**2)
        return E_L

    @partial(jit, static_argnums=(0,))
    def drift_force_analytical(self, r, alpha):
        return -4 * alpha * jnp.sum(r, axis=0)

    @partial(jit, static_argnums=(0,))
    def drift_force_jax(self, r, alpha):
        grad_wf = grad(self.evaluate)
        F = 2 * jnp.sum(grad_wf(r, alpha), axis=0) / self.evaluate(r, alpha)
        return F


if __name__ == "__main__":
    import numpy as np
    from jax import random

    N = 2
    d = 3
    alpha = 0.5
    omega = 1

    key = random.PRNGKey(0)
    r = random.normal(key, (N, d)) * 0.001
    print(r.shape)

    # print(r)
    assert np.any(np.array(r) < 0)

    r_np = np.array(r)

    wf = JaxWF(N, d, omega)
    print("Wf:", wf(r, alpha))
    print("Local energy:", wf.local_energy(r, alpha))
    print("Drift F analytical:", wf.drift_force_analytical(r, alpha))
    print("Drift F Jax:", wf.drift_force_jax(r, alpha))

    F_np = -4 * alpha * np.sum(r_np)
    print("Drift F Numpy:", F_np)
