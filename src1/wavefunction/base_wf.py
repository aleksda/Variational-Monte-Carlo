#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

class WaveFunction(metaclass=ABCMeta):
    """Base class for constructing trial wave functions.

    Parameters
    ----------
    n_particles : int
        Number of particles in system
    dim : int
        Dimensionality of system
    """
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

    @abstractmethod
    def __call__(self):
        """Evaluate the many body trial wave function.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    @abstractmethod
    def local_energy(self):
        """Compute the local energy.

        Must be overwritten by sub-class.
        """
        raise NotImplementedError

    def density(self, *args, **kwargs):
        """Compute the square of the many body trial wave function

        Parameters
        ----------
        *args
            args are passed to the call method
        **kwargs
            kwargs are passed to the call method

        Returns
        -------
        float
            The squared trial wave function
        """
        return self(*args, **kwargs)**2

    @property
    def dim(self):
        return self._d

    @property
    def n_particles(self):
        return self._N
