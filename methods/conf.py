import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax
from lsci import *

def conf_band(residuals, pca_state, alpha):
    res_proj = pcax.transform(pca_state, residuals)
    dval_conf = jnp.mean(local_tukey_self(res_proj, res_proj, 1/res_proj.shape[0]), axis = 0)
    qval_conf = jnp.quantile(dval_conf, alpha)
    conf_ens = pcax.recover(pca_state, res_proj[dval_conf > qval_conf])
    conf_lower = jnp.min(conf_ens, axis = 0)
    conf_upper = jnp.max(conf_ens, axis = 0)
    return conf_lower, conf_upper