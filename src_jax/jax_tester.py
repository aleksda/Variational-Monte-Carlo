import warnings
import numpy as np
import jax.numpy as jnp
import jax as jax
from jax import random
from numpy.random import default_rng
from jax.config import config

config.update("jax_enable_x64", True)


x = jnp.ones(10)

energies = jnp.zeros(10)

for i in range(10):
    if i%2 == 0:
        energies = energies.at[i].set(10)

print(energies)
