from abc import ABCMeta, abstractmethod
from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
import jax as jax
from jax.config import config

config.update("jax_enable_x64", True)

class BaseJaxWF(metaclass=ABCMeta):

    def __init__(self, n_particles, dim):
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


    def precompute(self):
        self.grad_wf = grad(self.wavefunction, argnums=0)
        self.laplace = jax.jacfwd(self.grad_wf, argnums=0)
        self.grad2_wf = grad(self.grad_wf, argnums=0)



        print("Testing second derivative.. ")
        print("Jax df/dr([1.0, 1.0], 0.5): ", jnp.sum(self.grad_wf(jnp.array([1.0, 1.0, 1.0], 0.5)))
        print("Analytical: ", self.analytical_dfdr(1.0, 0.5))
        print("Jax d2fdr2(1.0, 0.5): ", jnp.sum(self.hessian(1.0, 0.5)))
        print("Analytical: ", self.analytical_d2fdr2(1.0, 0.5))



    def analytical_dfdr(self, r, alpha):
        return -2.0*alpha*np.sum(r)*np.exp(-alpha*np.sum(r**2))

    def analytical_d2fdr2(self, r, alpha):
        return -2.0*alpha*np.exp(-alpha*np.sum(r**2)) + 4.0*alpha*alpha*np.sum(r**2)*np.exp(-alpha*np.sum(r**2))

    @abstractmethod
    def wavefunction(self):
        raise NotImplementedError


    def hamiltonian(self, r, alpha):
        return -0.5*jnp.sum(self.laplace(r, alpha)) + self.potential(r)*self.wavefunction(r, alpha)

    @abstractmethod
    def potential(self):
        raise NotImplementedError

    def density(self, r, alpha):
        return self.wavefunction(r, alpha)*self.wavefunction(r, alpha)

    def logdensity(self, r, alpha):
        return jnp.log(self.wavefunction(r, alpha)*self.wavefunction(r, alpha))

    def local_energy(self, r, alpha):
        H = self.hamiltonian(r, alpha)
        return H / self.wavefunction(r, alpha)

    @partial(jit, static_argnums=(0,))
    def drift_force_jax(self, r, alpha):
        #grad_wf = grad(self.evaluate)
        F = (2 * self.grad_wf(r, alpha)) / self.wavefunction(r, alpha)
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

    def __init__(self, n_particles, dim, omega):
        super().__init__(n_particles, dim)
        self._omega = omega

    def wavefunction(self, r, alpha):
        return jnp.exp(-alpha*jnp.sum(r**2))

    def potential(self, r):
        return 0.5*self._omega*self._omega*jnp.sum(r**2)
