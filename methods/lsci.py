import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

def warp01(cdf):
    _min = jnp.min(cdf, axis = 0)[None,]
    _max = jnp.max(cdf, axis = 0)[None,]
    return (cdf - _min) / (_max - _min)

def local_weights(yval, ytest, frac = 0.1):
    dmat = jnp.max(jnp.abs(yval - ytest), axis = 1)
    quant = jnp.quantile(dmat, frac, axis = 0)
    indx = dmat < quant
    inner_max = jnp.max(dmat * indx)
    dmat = inner_max - dmat
    w = dmat * indx
    return w / jnp.sum(w, axis = 0)[None,]
local_weights = jit(vmap(local_weights, (None, 0, None)))

def local_cdf(rval, q, weight):
    lcdf = jnp.sum(weight * (rval < q))
    return lcdf
local_cdf = jit(local_cdf)
local_cdf = jit(vmap(local_cdf, (None, None, 0)))

def local_quantile(rval, p, weight):
    rval_sort = jnp.sort(rval, axis = 0)
    lcdf = jnp.sum(weight[None,:] * (rval[None,:] < rval_sort[:,None]), axis = 1)
    lcdf = warp01(lcdf)
    ind = jnp.argmax(1.0*(lcdf >= p))
    return rval_sort[ind]
local_quantile = jit(local_quantile)
local_quantile_vmap = jit(vmap(local_quantile, (1, None, None)))
local_quantile_proj = vmap(local_quantile_vmap, (None, None, 0))

def local_tukey(rval, q, weight):
    lcdf_l = jnp.sum(weight * (rval < q))
    lcdf_r = jnp.sum(weight * (rval > q))
    return 2 * jnp.min(jnp.array([lcdf_l, lcdf_r]))
local_tukey = jit(local_tukey)
local_tukey_vmap = vmap(local_tukey, (1, 1, None))

local_tukey_self = vmap(local_tukey, (None, 0, None))
local_tukey_self = jit(vmap(local_tukey_self, (1, 1, None)))

def warped_unif(rng, a, b):
    u = jax.random.uniform(rng)
    return (b - a) * u + a

def conformal_resample(rng, x, w):
    phi = local_quantile(x, warped_unif(rng, 0.0, 1.0), w)
    return phi
conformal_resample = vmap(conformal_resample, (0, 1, None))

def conformal_ens(rng, x, w):
    rngs = random.split(rng, x.shape[1])
    return conformal_resample(rngs, x, w)
conformal_ens = vmap(conformal_ens, (0, None, None))

def lsci_sample(residuals, local_weight, pca_state, alpha, nsamp, rng = random.PRNGKey(0)):
    res_proj = pcax.transform(pca_state, residuals)
    rngs = random.split(rng, nsamp)
    phi_t = conformal_ens(rngs, res_proj, local_weight)
    ens_t = pcax.recover(pca_state, phi_t)
    
    dval = jnp.mean(local_tukey_self(res_proj, res_proj, local_weight), axis = 0)
    qval = jnp.quantile(dval, alpha)
    dphi = jnp.mean(local_tukey_self(res_proj, phi_t, local_weight), axis = 0)
    return ens_t[dphi > qval]

def lsci_band(residuals, local_weight, pca_state, alpha, nsamp, rng = random.PRNGKey(0)):
    lsci_conf = lsci_sample(residuals, local_weight, pca_state, alpha, nsamp, rng)
    lsci_lower = jnp.min(lsci_conf, axis = 0)
    lsci_upper = jnp.max(lsci_conf, axis = 0)
    return lsci_lower, lsci_upper