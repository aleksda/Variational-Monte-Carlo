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
        dt=0.3
    ):
        self._ncycles = ncycles
        self._alphas = alphas
        initial_state = None
        energies = np.zeros(len(self._alphas))
        variances = np.zeros(len(self._alphas))


        self._propsal_dist = self._draw_prop_gaussian_importance

        for i, alpha in enumerate(alphas):
            results = self._importance(alpha, initial_state, dt)
            energies[i] = results[0]
            variances[i] = results[1]
            densities = results[2]
            E_propagation = results[3]

        return energies, variances, densities, E_propagation

    def _importance(self, alpha, initial_state, dt):
        if initial_state is None:
            positions = self._initial_positions()
        else:
            positions = initial_state

        D = 0.5
        u = self._rng.random(size=self._ncycles)
        wf2 = self._wf.density(positions, alpha)
        print("Energy at init positions: ", self._wf.local_energy(positions, alpha))

        n_accepted = 0
        energy = 0
        energy2 = 0
        energies = np.zeros(self._ncycles)
        densities = np.zeros(self._ncycles)
        qforce = self._wf.drift_force(positions, alpha)
        #print(qforce)
        for i in range(self._ncycles):
            trial_positions = positions + self._rng.normal(loc=0.0, scale=1.0)*dt + qforce*dt*D
            #print(trial_positions)
            trial_wf2 = self._wf.density(positions, alpha)

            P = self._greens_relative(trial_positions, positions, dt, alpha)*trial_wf2/wf2

            if u[i] <= P:
                positions = trial_positions
                qforce = self._wf.drift_force(positions, alpha)
                wf2 = trial_wf2
                n_accepted += 1

            local_energy = self._wf.local_energy(positions, alpha)
            energy += local_energy
            energies[i] = local_energy
            energy2 += local_energy**2
            densities[i] = wf2
        print("Final positins: ", positions)

        # acceptance rate
        acc_rate = n_accepted / self._ncycles
        print(f"{alpha=:.2f}, {acc_rate=:.2f}")
        # Calculate mean, variance, error
        energy /= self._ncycles
        energy2 /= self._ncycles
        variance = energy - energy2
        # error = np.sqrt(variance / self._ncycles)

        return energy, variance, densities, energies

    def _tune(self, alpha, positions):

        n_accepted_tune = 0
        wf2 = self._wf.density(positions, alpha)
        u = self._rng.random(size=self._tune_iter)

        for i in range(self._tune_iter):
            trial_positions = self._propsal_dist(positions)
            trial_wf2 = self._wf.density(trial_positions, alpha)

            # Metropolis acceptance criterion
            if u[i] <= trial_wf2 / wf2:
                positions = trial_positions
                wf2 = trial_wf2
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
        wf2 = self._wf.density(positions, alpha)

        while wf2 <= 1e-14:
            positions /= 2
            wf2 = self._wf.density(positions, alpha)

        return positions

    def _greens_relative(self,new_positions, old_positions, delta_t, alpha):
        D = 0.5
        drift_new = self._wf.drift_force(new_positions, alpha)
        drift_old = self._wf.drift_force(old_positions, alpha)
        exponent = np.sum(0.5*(drift_old+drift_new)*((old_positions-new_positions)+0.5*(drift_old-drift_new)*D*delta_t))
        return np.exp(exponent)

    def _draw_prop_gaussian_importance(self, old_positions, dt):
        return old_positions + self._rng.normal(loc=0.0, scale=1.0)*dt

    def _draw_proposal_gaussian(self, old_positions):
        return self._rng.normal(loc=old_positions, scale=self._scale)

    def _draw_proposal_uniform(self, old_positions):
        return old_positions + (self._rng.random(size=(self._N, self._dim)) - 0.5)
