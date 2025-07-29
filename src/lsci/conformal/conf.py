import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax
import lsci

def conf_band(residuals, pca_state, alpha):
    n = residuals.shape[0]
    adj_alpha = jnp.ceil((n+1)*(1-alpha))/n

    res_proj = pcax.transform(pca_state, residuals)
    weights = (1/res_proj.shape[0]) * jnp.ones((1, 1))
    dval_conf = lsci.phi_tukey(res_proj, res_proj, weights).squeeze()
    qval_conf = jnp.quantile(dval_conf, 1-adj_alpha)
    conf_ens = pcax.recover(pca_state, res_proj[dval_conf > qval_conf])
    conf_lower = jnp.min(conf_ens, axis = 0)
    conf_upper = jnp.max(conf_ens, axis = 0)
    return conf_lower, conf_upper