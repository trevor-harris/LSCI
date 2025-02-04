import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

# conformal prediction bands (supremum method)
def modulator(r):
    mod_r = jnp.max(jnp.abs(r), axis = 0)
    return mod_r / jnp.mean(mod_r)

def band_nonconf(r, s):
    mod_r = r / s[None,:]
    return jnp.max(jnp.abs(mod_r), axis = 1)

def band_predset(r ,s):
    mod_r = s[None,:]
    cval_band = band_nonconf(rval)[:,None]
    return cval_band * mod_r

def supr_band(residuals, alpha):
    s_fn = modulator(residuals)
    cval_supr = band_nonconf(residuals, s_fn)[:,None]
    qval_supr = jnp.quantile(cval_supr, 1-alpha)
    band_supr = (qval_supr * s_fn)[None,:]
    return -band_supr, band_supr