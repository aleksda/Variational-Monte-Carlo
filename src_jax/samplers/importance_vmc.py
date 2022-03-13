#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class ImportanceVMC:

    def __init__(self, wavefunction, seed=None):
        self._wf = wavefunction
        self._seed = seed

        self._N = self._wf.n_particles
        self._dim = self._wf.dim
        self._rng = np.random.default_rng(seed=self._seed)

    def sample(
        self,
        ncycles,
        alphas,
        dt=0.01,
    ):
        self._ncycles = ncycles
        self._alphas = alphas
        self._dt = dt
        self._D = 0.5
        # precompute quantities
        self._sqrt_dt = np.sqrt(self._dt)
        self._Ddt = self._D * self._dt
        self._4Ddt = 4 * self._Ddt

        energies = np.zeros(len(self._alphas))
        variances = np.zeros(len(self._alphas))

        for i, alpha in enumerate(alphas):

            initial_state = self._safe_initialization(alpha)

            results = self._importance_sampler(alpha, initial_state)
            energies[i] = results[0]
            variances[i] = results[1]

        return energies, variances

    def _importance_sampler(self, alpha, positions):

        u = self._rng.random(size=self._ncycles)
        wf2 = self._wf.density(positions, alpha)
        F = self._wf.drift_force(positions, alpha)

        n_accepted = 0
        energy = 0
        energy2 = 0

        for i in range(self._ncycles):
            trial_positions = self._propose_move(positions, F)
            trial_wf2 = self._wf.density(trial_positions, alpha)
            trial_F = self._wf.drift_force(trial_positions, alpha)

            greens = self._greens_function(positions,
                                           trial_positions,
                                           F,
                                           trial_F)
            ratio = greens * trial_wf2 / wf2

            # Metropolis acceptance criterion
            if u[i] <= ratio:
                positions = trial_positions
                wf2 = trial_wf2
                F = trial_F
                n_accepted += 1

            local_energy = self._wf.local_energy(positions, alpha)
            energy += local_energy
            energy2 += local_energy**2

        # acceptance rate
        acc_rate = n_accepted / self._ncycles
        print(f"{alpha=:.2f}, {acc_rate=:.2f}")
        # Calculate mean, variance
        energy /= self._ncycles
        energy2 /= self._ncycles
        variance = energy - energy2

        return energy, variance

    def _greens_function(self, r_old, r_new, F_old, F_new):
        """Calculate Green's function MH ratio.
        Normalizing factors omitted as they cancel each other out
        """
        old_term = -(np.sum(r_new - r_old) - self._Ddt * F_old)**2
        new_term = -(np.sum(r_old - r_new) - self._Ddt * F_new)**2
        exponent = (old_term - new_term) / self._4Ddt
        return np.exp(exponent)

    def _initial_positions(self):

        initial_state = self._rng.random(size=(self._N, self._dim))  # * 0.001

        return initial_state

    def _safe_initialization(self, alpha):

        positions = self._initial_positions()
        wf2 = self._wf.density(positions, alpha)

        while wf2 <= 1e-14:
            positions /= 2
            wf2 = self._wf.density(positions, alpha)

        return positions

    def _propose_move(self, positions, F):

        new_positions = positions + self._Ddt * F + \
            self._rng.normal(loc=0, scale=self._sqrt_dt, size=(
                self._N, self._dim))

        return new_positions
