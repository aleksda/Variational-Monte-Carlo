from functools import partial
from . import WaveFunction
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.config import config

config.update("jax_enable_x64", True)


class JaxWaveFunction(WaveFunction):
    """
    Solver for general wave function using
    automatic differentiation using Jax.
    """

    def __init__(self, n_particles, dim, omega):
        super().__init__(n_particles, dim)
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
        return -4 * alpha * jnp.sum(r)

    @partial(jit, static_argnums=(0,))
    def drift_force_jax(self, r, alpha):
        grad_wf = grad(self.evaluate)
        F = (2 * grad_wf(r, alpha)) / self.evaluate(r, alpha)
        return F.sum()

    @partial(jit, static_argnums=(0,))
    def drift_force_jax2(self, r, alpha):
        grad_wf = grad(self.evaluate)
        logF = jnp.log(2. * grad_wf(r, alpha)) - \
            jnp.log(self.evaluate(r, alpha))
        F = jnp.exp(logF)
        return F.sum()
