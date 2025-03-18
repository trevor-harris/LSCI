import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

def risk(lower, upper, residual):
    return jnp.mean((residual > lower)*(residual < upper))