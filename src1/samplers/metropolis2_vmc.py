#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Metropolis2VMC:
    """
    Metropolis algorithm moving only one particle at a time
    in configuration space.
    """

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
        scale=0.5,
        tune=True,
        tune_iter=5000,
        tune_interval=250,
    ):
        self._ncycles = ncycles
        self._alphas = alphas
        self._scale = scale

        initial_state = None
        energies = np.zeros(len(self._alphas))
        variances = np.zeros(len(self._alphas))

        self._propsal_dist = self._draw_proposal_gaussian

        for i, alpha in enumerate(alphas):
            if tune:
                self._tune_iter = tune_iter
                self._tune_interval = tune_interval
                #initial_state_tune = self._initial_positions()
                initial_state_tune = self._safe_initialization(alpha)
                initial_state = self._tune(alpha, initial_state_tune)

            results = self._metropolis(alpha, initial_state)
            energies[i] = results[0]
            variances[i] = results[1]

        return energies, variances

    def _metropolis(self, alpha, initial_state):
        if initial_state is None:
            positions = self._initial_positions()
        else:
            positions = initial_state

        u = self._rng.random(size=(self._ncycles, self._N))
        wf2 = self._wf.density_per_particle(positions, alpha)

        n_accepted = 0
        energy = 0
        energy2 = 0

        for i in range(self._ncycles):
            trial_positions = self._propsal_dist(positions)
            trial_wf2 = self._wf.density_per_particle(trial_positions, alpha)
            for j in range(self._N):
                # Metropolis acceptance criterion
                if u[i,j] <= trial_wf2[j] / wf2[j]:
                    positions[j] = trial_positions[j]
                    #wf2 = trial_wf2
                    n_accepted += 1

                local_energy = self._wf.local_energy(positions, alpha)/self._N
                energy += local_energy
                energy2 += local_energy**2

        # acceptance rate
        acc_rate = n_accepted / self._ncycles
        print(f"{alpha=:.2f}, {acc_rate=:.2f}")
        # Calculate mean, variance, error
        energy /= self._ncycles
        energy2 /= self._ncycles
        variance = energy - energy2
        # error = np.sqrt(variance / self._ncycles)

        return energy, variance

    def _tune(self, alpha, positions):

        n_accepted_tune = 0
        wf2 = self._wf.density_per_particle(positions, alpha)
        u = self._rng.random(size=(self._tune_iter, self._N))

        for i in range(self._tune_iter):
            trial_positions = self._propsal_dist(positions)
            trial_wf2 = self._wf.density_per_particle(trial_positions, alpha)
            for j in range(self._N):
                # Metropolis acceptance criterion
                if u[i, j] <= trial_wf2[j] / wf2[j]:
                    positions[j] = trial_positions[j]
                    n_accepted_tune += 1

            if i % self._tune_interval == 0:
                acc_rate = n_accepted_tune / self._tune_interval
                self._tune_table(acc_rate)
                n_accepted_tune = 0

        return positions

    def _tune_table(self, acc_rate):
        """Proposal scale lookup table.

        Retrieved from PyMC3 source.

        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        if acc_rate < 0.001:
            # reduce by 90 percent
            self._scale *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            self._scale *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            self._scale *= 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            self._scale *= 10.0
        elif acc_rate > 0.75:
            # increase by double
            self._scale *= 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            self._scale *= 1.1

    def _initial_positions(self):
        '''
        initial_state = self._rng.normal(
            loc=np.zeros((self._n_particles, self._dim)),
            scale=self._scale
        )
        '''
        initial_state = self._rng.random(size=(self._N, self._dim))
        #initial_state = np.zeros(shape=(self._N, self._dim))
        return initial_state

    def _safe_initialization(self, alpha):

        positions = self._initial_positions()
        wf2 = np.prod(self._wf.density_per_particle(positions, alpha))

        while wf2 <= 1e-14:
            positions /= 2
            wf2 = np.prod(self._wf.density_per_particle(positions, alpha))

        return positions

    def _draw_proposal_gaussian(self, old_positions):
        return self._rng.normal(loc=old_positions, scale=self._scale)

    def _draw_proposal_uniform(self, old_positions):
        return old_positions + (self._rng.random(size=(self._N, self._dim)) - 0.5)
