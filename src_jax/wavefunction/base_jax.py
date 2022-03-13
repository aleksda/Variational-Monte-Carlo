from abc import ABCMeta, abstractmethod
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, hessian
import jax as jax
from jax.config import config

config.update("jax_enable_x64", True)






class BaseJaxWF:

    def __init__(self, wavefunction, potential, n_particles, dim):
        """Base class for computing properties of trial wave functon,
        using jax and autodifferentiation.

        Parameters
        ----------
        wavefunction    : function
            Trial wavefunction
        potential       : function
            Potential used in Hamiltonian of system
        n_particles     : int
            Number of particles in system
        dim             : int
            Dimensionality of system
        """
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

        # Initialise system
        self._N = n_particles
        self._d = dim

        self.wf = wavefunction
        self.potential = potential

        # Finding functions needed to evaluate properties of system.
        self.grad_wf = grad(self.wf, argnums=0, holomorphic=False)
        self.hessian = hessian(self.wf, argnums=0, holomorphic=False)
        self.lap = self.laplacian
        self.lap_vector = self.laplacian_vector

    def hamiltonian(self, r, alpha):
        #kinetic = -0.5*jnp.sum(jnp.diag(jnp.sum(jnp.sum(self.hessian(r, alpha), axis=1), axis=-1)))
        kinetic = -0.5*jnp.sum(self.lap_vector(r, alpha))
        return kinetic + self.potential(r, self._omega)

    def density(self, r, alpha):
        return self.wf(r, alpha)*self.wf(r, alpha)

    def gradient_wf(self, r, alpha):
        return jnp.sum(self.grad_wf(r, alpha), axis=0)

    def logdensity(self, r, alpha):
        return jnp.log(self.wf(r, alpha)*self.wf(r, alpha))

    def reduced_hessian(self, r, alpha):
        return jnp.sum(jnp.sum(self.hessian(r, alpha), axis=1), axis=-1)

    def laplacian_vector(self, r, alpha):
        return jnp.diag(self.reduced_hessian(r, alpha))

    def laplacian(self, r, alpha):
        return jnp.sum(self.lap_vector(r, alpha))

    def local_energy(self, r, alpha):
        H = self.hamiltonian(r, alpha)
        return H / self.wf(r, alpha)

    def drift_force(self, r, alpha):
        #grad_wf = grad(self.evaluate)
        F = 2 *self.gradient_wf(r, alpha) / self.wf(r, alpha)
        return F

    @property
    def dim(self):
        return self._d

    @property
    def n_particles(self):
        return self._N


class SG(BaseJaxWF):
    """Single particle wave function with Gaussian kernel.

    Parameters
    ----------
    n_particles : int
        Number of particles in system
    dim : int
        Dimensionality of system
    omega : float
        Harmonic oscillator frequency
    """

    def __init__(self, wavefunction, potential, _particles, dim, omega):
        super().__init__(wavefunction, potential, _particles, dim)
        self._omega = omega


def wavefunction(r, alpha):
    return jnp.exp(-alpha*jnp.sum(r**2))

def potential(r, _omega):
    return 0.5*_omega*_omega*jnp.sum(r**2)

def drift_force_analytical(r, alpha):
    return -4 * alpha * jnp.sum(r, axis=0)


if __name__ == "__main__":
    from jax import random
    N = 500
    d = 3
    alpha = 0.5
    omega = 1

    key = random.PRNGKey(0)
    r = random.normal(key, (N, d)) * 0.001
    print(r.shape)

    # print(r)
    assert np.any(np.array(r) < 0)

    r_np = np.array(r)

    wf = SG(wavefunction, potential, N, d, omega)
    print("Local energy:", wf.local_energy(r, alpha))
    print("Drift F analytical:", drift_force_analytical(r, alpha))
    print("Drift F Jax:", wf.drift_force(r, alpha))
    print("Hessian F shape Jax: ", wf.hessian(r, alpha).shape)
