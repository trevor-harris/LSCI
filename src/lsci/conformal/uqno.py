import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

# def estimate_lambda(scores, alpha = 0.1, delta = 0.01, tau_factor = 1.1):
    
#     # scores = jnp.abs(yval - yval_hat) / yval_quant
#     m = scores.shape[-1]
#     tau = tau_factor * jnp.sqrt(-jnp.log(delta)/(2*m))
#     scores = jnp.quantile(scores, 1-alpha+tau, axis = (1))
#     nval = scores.shape[0]

#     adj_alpha = 1 - jnp.ceil((nval + 1) * (delta - jnp.exp(-2*m*tau**2)))/nval
#     lam_uqno = jnp.quantile(scores, adj_alpha)
#     return lam_uqno


def uqno_lambda(yval, yval_hat, yval_quant, delta, alpha):
    n = yval.shape[0]
    m = yval.shape[2]
    tau = 1e-2 + jnp.sqrt(-jnp.log(delta)/(2*m))
    
    uqno_scores = jnp.abs(yval.squeeze() - yval_hat) / yval_quant
    sg = jnp.quantile(uqno_scores, jnp.clip(1-alpha+tau, 0, 1), axis = 1)
    q = 1 - jnp.ceil((n+1)*(delta - jnp.exp(-2*m*tau**2)))/n
    return jnp.quantile(sg, q)


def uqno_band(ytest_quant, lam_uqn):
    band_uqn = (lam_uqn * ytest_quant)
    return -band_uqn, band_uqn