import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

def uqno_band(ytest_quant, lam_uqn):
    band_uqn = (lam_uqn * ytest_quant)
    return -band_uqn, band_uqn