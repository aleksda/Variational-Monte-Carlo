#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import jax.numpy as jnp
import jax as jax
from jax import random
from numpy.random import default_rng
from jax.config import config

config.update("jax_enable_x64", True)


warnings.filterwarnings("ignore", message="divide by zero encountered")


def rw_metropolis_step(logp, positions, *model_args, scale=0.5, iteration):
    """Random walk Metropolis step generator"""
    logp_current = logp(positions, *model_args)
    #log_unif = jnp.log(rng.random(size=positions.shape))
    key = random.PRNGKey(iteration)
    key1, key2 = random.split(key)
    log_unif = jnp.log(random.uniform(key1, shape=(positions.shape)))
    proposals = positions + random.normal(key2, shape=(positions.shape))*self.scale
    logp_proposal = logp(proposals, *model_args)
    accept = log_unif < logp_proposal - logp_current
    positions = jnp.where(accept, proposals, positions)
    yield positions


class MetropolisVMC2jax:

    def __init__(self, wavefunction):
        self._wf = wavefunction
        self._logp = self._wf.logdensity
        self._Eloc = self._wf.local_energy
        self._rng_key = random.PRNGKey(0)

    def sample(
        self,
        ncycles,
        initial_state,
        alpha,
        scale=0.5,
        burn=500,
        warm=True,
        warmup_iter=500,
        tune=True,
        tune_iter=5000,
        tune_interval=250,
        tol_scale=1e-5,
        optimize=True,
        max_iter=10000,
        batch_size=500,
        gradient_method='adam',
        eta=0.001,
        tol_optim=1e-5
    ):
        self._alpha = alpha
        self._scale = scale
        self._eta = eta
        self._tol_scale = tol_scale
        self._tol_optim = tol_optim
        self._ncycles = ncycles

        if gradient_method == 'gd':
            self._gradient_method = self._gradient_descent
        elif gradient_method == 'gdmom':
            self._gamma = 0.9
            self._v = None
            self._gradient_method = self._gradient_descent_momentum
        elif gradient_method == 'adam':
            self._beta1 = 0.9
            self._beta2 = 0.999
            self._epsilon = 1e-8
            self._m = 0
            self._v = 0
            self._t = 0
            self._gradient_method = self._adam

        if warm:
            initial_state = self._warmup(warmup_iter,
                                         initial_state,
                                         alpha)

        if tune:
            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            '''
            if optimize:
                intitial_state, alpha = self._tune_optimize(tune_iter,
                                                            tune_interval,
                                                            initial_state,
                                                            alpha)
            else:
                intitial_state = self._tune(tune_iter,
                                            tune_interval,
                                            initial_state,
                                            alpha)
            '''

        if optimize:

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            # Tune again for good measure?
            '''
            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)
            '''

            # Check
            '''
            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)

            intitial_state, alpha = self._optimize(max_iter,
                                                   batch_size,
                                                   initial_state,
                                                   alpha)

            intitial_state = self._tune(tune_iter,
                                        tune_interval,
                                        initial_state,
                                        alpha)
            '''

        energies = self._rw_metropolis(ncycles, initial_state, alpha)

        self._energy_samples = jax.lax.slice(energies, start_indices=(burn+1,), limit_indices=(self._ncycles-1,))
        energy = jnp.mean(self._energy_samples)

        return energy

    def _rw_metropolis_step(self, positions, logp_current, alpha, iteration):
        #log_unif = jnp.log(jnp.asarray(self._rng.random(size=positions.shape)))
        key = random.PRNGKey(iteration)
        key1, key2 = random.split(key)
        log_unif = jnp.log(random.uniform(key1, shape=(positions.shape)))
        proposals = positions + random.normal(key2, shape=(positions.shape))*self.scale
        #proposals = jnp.asarray(self._rng.normal(loc=positions, scale=self._scale))
        logp_proposal = self._logp(proposals, alpha)
        accept = log_unif < logp_proposal - logp_current
        # Where accept is True, yield proposal, otherwise yield old position
        positions = jnp.where(accept, proposals, positions)
        logp_current = jnp.where(accept, logp_proposal, logp_current)
        accepted = jnp.sum(accept)
        return positions, logp_current, accepted

    def _warmup(self, warmup_iter, positions, alpha):
        """Warm-up chain"""
        logp_current = self._logp(positions, alpha)
        for _ in range(warmup_iter):
            positions, logp_current, accepted = self._rw_metropolis_step(
                positions,
                logp_current,
                alpha,
                _)
        return positions

    def _tune(self, tune_iter, tune_interval, positions, alpha):
        """Tune proposal scale"""
        steps_before_tune = tune_interval
        n_accepted = 0
        logp_current = self._logp(positions, alpha)

        for i in range(tune_iter):
            positions, logp_current, accepted = self._rw_metropolis_step(
                positions,
                logp_current,
                alpha,
                i)

            n_accepted += accepted
            steps_before_tune -= 1

            if steps_before_tune == 0:
                scale_old = self._scale
                accept_rate = n_accepted / tune_interval
                self._tune_scale_table(accept_rate)

                dL2 = jnp.linalg.norm(self._scale - scale_old)
                if dL2 < self._tol_scale:
                    print(f"Tune early stopping at iter {i+1}/{tune_iter}")
                    break

                # reset
                n_accepted = 0
                steps_before_tune = tune_interval

        return positions

    def _optimize(self, max_iter, batch_size, positions, alpha):
        steps_before_optimize = batch_size
        logp_current = self._logp(positions, alpha)
        grad_energies = []

        for i in range(max_iter):
            positions, logp_current, accepted = self._rw_metropolis_step(
                positions,
                logp_current,
                alpha,
                i)
            grad_energies.append(self._grad_Eloc(positions, alpha))

            steps_before_optimize -= 1
            if steps_before_optimize == 0:
                alpha_old = alpha
                alpha = self._gradient_method(alpha, np.mean(grad_energies))
                dL2_alpha = jnp.linalg.norm(alpha - alpha_old)

                if dL2_alpha < self._tol_optim:
                    print(f"Optimize early stopping at iter {i+1}/{max_iter}")
                    break

                # reset
                grad_energies = []
                steps_before_optimize = batch_size
        self._alpha = alpha
        return positions, alpha

    def _tune_optimize(self, tune_iter, tune_interval, positions, alpha):
        """Tune proposal scale"""
        steps_before_tune = tune_interval
        n_accepted = 0
        logp_current = self._logp(positions, alpha)
        grad_energies = []

        for i in range(tune_iter):
            positions, logp_current, accepted = self._rw_metropolis_step(
                positions,
                logp_current,
                alpha,
                i)

            grad_energies.append(self._grad_Eloc(positions, alpha))

            n_accepted += accepted
            steps_before_tune -= 1

            if steps_before_tune == 0:
                # Tune scale
                scale_old = self._scale
                accept_rate = n_accepted / tune_interval
                self._tune_scale_table(accept_rate)

                # Optimize alpha
                alpha_old = alpha
                alpha = self._gradient_method(alpha, jnp.mean(grad_energies))

                # early stopping?
                dL2_scale = jnp.linalg.norm(self._scale - scale_old)
                dL2_alpha = jnp.linalg.norm(alpha - alpha_old)

                if dL2_scale < self._tol_scale and dL2_alpha < self._tol_optim:
                    print(f"Early stopping at iter {i+1}/{tune_iter}")
                    print(f"Optimal: scale={self._scale}, {alpha=}")
                    break

                # reset
                n_accepted = 0
                grad_energies = []
                steps_before_tune = tune_interval

        return positions, alpha

    def _gradient_descent(self, alpha, gradient):
        alpha -= self._eta * gradient
        return alpha

    def _gradient_descent_momentum(self, alpha, gradient):
        if self._v is None:
            self._v = self._eta * gradient
        self._v *= self._gamma
        alpha -= self._v - self._eta * gradient
        return alpha

    def _adam(self, alpha, gradient):
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * gradient
        self._v = self._beta2 * self._v + (1 - self._beta2) * gradient**2
        m_hat = self._m / (1 - self._beta1**self._t)
        v_hat = self._v / (1 - self._beta2**self._t)
        alpha -= self._eta * m_hat / (jnp.sqrt(v_hat) - self._epsilon)
        return alpha

    def _rw_metropolis(self, ncycles, positions, alpha):
        """Sampling"""
        energies = jnp.zeros(ncycles)
        n_accepted = 0
        logp_current = self._logp(positions, alpha)

        for i in range(ncycles):
            positions, logp_current, accepted = self._rw_metropolis_step(
                positions,
                logp_current,
                alpha,
                i)
            energies = energies.at[i].set(self._Eloc(positions, alpha))
            n_accepted += accepted

        self._acc_rate = n_accepted / ncycles
        return energies

    def _tune_scale_table(self, acc_rate):
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

    @property
    def energy_samples(self):
        return self._energy_samples

    @property
    def accept_rate(self):
        return self._acc_rate

    @property
    def scale(self):
        return self._scale

    @property
    def alpha(self):
        return self._alpha
