#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from . import WaveFunction


class SimpleGaussian(WaveFunction):
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
        # return np.exp(-alpha * np.sum(r**2))
        return np.exp(-alpha * r**2)

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
            (0.5 * self._omega**2 - 2 * alpha**2) * np.sum(r**2)
        return E_L

    def grad_local_energy(self, r, alpha):
        """W.r.t. alpha"""
        return self._N * self._d - 4 * alpha * np.sum(r**2)

    def gradient(self, r, alpha):
        # return -4 * alpha * np.sum(r) * self(r, alpha)
        return - np.sum(r**2) * self.local_energy(r, alpha)

    def laplacian(self, r, alpha):
        grad2 = (-2 * self._N * self._d * alpha + 4 *
                 alpha**2 * np.sum(r**2)) * self(r, alpha)
        return grad2

    def drift_force(self, r, alpha):
        # return -4 * alpha * np.sum(r)
        return -4 * alpha * r
