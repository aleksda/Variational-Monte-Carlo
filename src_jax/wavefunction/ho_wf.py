#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from . import WaveFunction


class HOWF(WaveFunction):
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

    def __call__(self, x, alpha):
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
        return np.exp(-0.5 * alpha**2 * x**2)

    def local_energy(self, x, alpha):
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
        E_L = 0.5 * (alpha**2 + x**2 * (1 - alpha**4))
        return E_L

    def grad_local_energy(self, r, alpha):
        """W.r.t. alpha"""
        return 0.5 * alpha - 1 / (2 * alpha**3)
