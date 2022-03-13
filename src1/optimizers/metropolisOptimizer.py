# Metropolis optimizer.
import random
import numpy as np


class Optimizer:
    def __init__(self, initial_guess, samples_derivative):
        self._alpha = initial_guess
        self._samples_derivative = samples_derivatives
        self._num_datapoints = len(samples_derivative)

    def gradient_descent(self, guess, learning_rate, momentum_term, momentum_factor=0.0):
        """
        Gradient descent with momentum. Equates to SGD with batch_size = _num_datapoints.

        Parameters
        ----------
        guess : float (generally np.ndarray with size=(len(parameters)))
        learning_rate : float
        Returns
        -------
        new_alpha : float (generally np.ndarray with size=(len(parameters)))
        m : float (generally np.ndarray with size=(len(parameters)))
            momentum_term from this iteration
        """

        derivative = np.sum(self._samples_derivative)/self._num_datapoints
        m = momentum_factor*momentum_term + learning_rate*derivative
        next_guess = guess - m
        return next_guess, m

    def SGD(self, guess, learning_rate, batch_size, momentum_term, momentum_factor=0.0):
        """
        Stochastic gradient descent.

        Parameters
        ----------
        guess               : float (generally np.ndarray with size=(len(guess)))
        learning_rate       : float
        batch_size          : int
        momentum_term       : float (weighted accumulation of previous iterations)
        momentum_factor     : float
        Returns
        -------
        guess               : float (generally np.ndarray with size=(len(guess)))
        next_momentum_term  : float (generally np.ndarray with size=(len(guess)))
        """
        num_updates = int(self._num_datapoints/batch_size)
        samples = self._samples_derivative
        sample_order = random.sample(samples, num_updates*batch_size)
        accumulated_m = 0
        for i in range(num_updates):
            batch = sample_order[i:(i+batch_size)]
            batch_gradient = np.sum(batch)/batch_size
            batch_m = momentum_factor*momentum_term + learning_rate*batch_gradient
            accumulated_m += batch_m
            guess = guess - batch_m
        new_momentum_term = accumulated_m/num_updates
        return guess, next_momentum_term
