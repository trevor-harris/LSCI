import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax
from lsci import *

def gaussian_score(res, mu, sd):
    nll = jnp.mean((mu - res)**2 / (2*sd**2) + 0.5 * jnp.log(sd**2))
    return jnp.exp(-nll)
gaussian_score = vmap(gaussian_score, (0, None, None))

def gaus_band(residuals, pca_state, alpha):
    res_proj = pcax.transform(pca_state, residuals)
    mu = jnp.mean(res_proj, axis = 0)
    sd = jnp.std(res_proj, axis = 0)
    dval_conf = gaussian_score(res_proj, mu, sd)
    qval_conf = jnp.quantile(dval_conf, alpha)
    conf_ens = pcax.recover(pca_state, res_proj[dval_conf > qval_conf])
    conf_lower = jnp.min(conf_ens, axis = 0)
    conf_upper = jnp.max(conf_ens, axis = 0)
    return conf_lower, conf_upper