import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax

## random slicer
def phi_slice(rng, p1, p2):
    phi = random.normal(rng, (p1, p2))
    phi = phi / jnp.sum(phi, axis = 1)[:,None]
    return phi

# scaled metrics
def scaled_linf_dist(x, y):
    d = jnp.max(jnp.abs(x - y))
    return d/(1+d)
scaled_linf_dist = jit(vmap(vmap(scaled_linf_dist, (0, None)), (None, 0)))

def scaled_l2_dist(x, y):
    d = jnp.sqrt(jnp.mean((x - y)**2))
    return d/(1+d)
scaled_l2_dist = jit(vmap(vmap(scaled_l2_dist, (0, None)), (None, 0)))

# localizers
def linf_localizer(xval, xtest, lam):
    dist = scaled_linf_dist(xval, xtest)
    dist = jnp.exp(-lam * dist)
    local_weights = dist / (jnp.exp(-lam) + jnp.sum(dist, axis = 1, keepdims=True))
    return local_weights

def l2_localizer(xval, xtest, lam):
    dist = scaled_l2_dist(xval, xtest)
    dist = jnp.exp(-lam * dist)
    local_weights = dist / (jnp.exp(-lam) + jnp.sum(dist, axis = 1, keepdims=True))
    return local_weights

def knn_localizer(xval, xtest, lam):
    dist = scaled_linf_dist(xval, xtest)
    qdist = jnp.quantile(dist, 1/(1+lam))
    dist = jnp.exp(-dist.at[dist > qdist].set(qdist))
    local_weights = dist / (jnp.exp(-1/lam) + jnp.sum(dist, axis = 1, keepdims=True))
    return local_weights

# phi depths
def weighted_tukey1d(phi_val, phi_test, weights):
    f1 = jnp.sum(weights * (phi_val < phi_test))
    f2 = jnp.sum(weights * (phi_val > phi_test))
    f = jnp.vstack([f1, f2])
    return 2*jnp.min(f)
weighted_tukey1d = vmap(weighted_tukey1d, (None, 0, None))
weighted_tukey1d = vmap(weighted_tukey1d, (1, 1, None))
weighted_tukey1d = jit(weighted_tukey1d)

def weighted_norm1d(phi_val, phi_test, weights):
    f = jnp.sum(weights * jnp.abs(phi_val - phi_test))
    f = jnp.nan_to_num(f)
    return 1/(1 + f)
weighted_norm1d = vmap(weighted_norm1d, (None, 0, None))
weighted_norm1d = vmap(weighted_norm1d, (1, 1, None))
weighted_norm1d = jit(weighted_norm1d)

def weighted_mahal1d(phi_val, phi_test, weights):
    f = jnp.sum(weights * jnp.abs(phi_val - phi_test) / jnp.std(phi_val))
    f = jnp.nan_to_num(f)
    return 1/(1 + f)
weighted_mahal1d = vmap(weighted_mahal1d, (None, 0, None))
weighted_mahal1d = vmap(weighted_mahal1d, (1, 1, None))
weighted_mahal1d = jit(weighted_mahal1d)

def phi_tukey(phi_val, phi_test, weights):
    depth = jnp.min(weighted_tukey1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
phi_tukey = vmap(phi_tukey, (None, None, 0))
phi_tukey = jit(phi_tukey)

def phi_norm(phi_val, phi_test, weights):
    depth = jnp.min(weighted_norm1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
phi_norm = vmap(phi_norm, (None, None, 0))
phi_norm = jit(phi_norm)

def phi_mahal(phi_val, phi_test, weights):
    depth = jnp.min(weighted_mahal1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
phi_mahal = vmap(phi_mahal, (None, None, 0))
phi_mahal = jit(phi_mahal)

## diagonal versions
def vphi_tukey(phi_val, phi_test, weights):
    depth = jnp.min(weighted_tukey1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
vphi_tukey = vmap(vphi_tukey, (None, 0, 0))
vphi_tukey = jit(vphi_tukey)

def vphi_norm(phi_val, phi_test, weights):
    depth = jnp.min(weighted_norm1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
vphi_norm = vmap(vphi_norm, (None, 0, 0))
vphi_norm = jit(vphi_norm)

def vphi_mahal(phi_val, phi_test, weights):
    depth = jnp.min(weighted_mahal1d(phi_val, phi_test, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
vphi_mahal = vmap(vphi_mahal, (None, 0, 0))
vphi_mahal = jit(vphi_mahal)


### sampler
def local_cdf(x, q, weights):
    f = jnp.sum(weights * (x < q))
    return f
local_cdf = jit(vmap(local_cdf, (None, 0, None)))

def local_quantile(x, cdf, unif):
    return x[jnp.argmax(cdf > unif)]
local_quantile = jit(vmap(local_quantile, (None, None, 0)))

def local_sampler(x, weights, n_samp, rng):
    quants = jnp.linspace(jnp.min(x) - 0.1, jnp.max(x) + 0.1, x.shape[0]+1)
    unif = random.uniform(rng, n_samp)
    lcdf = local_cdf(x, quants, weights.T)
    return local_quantile(x, lcdf, unif)
local_sampler = vmap(local_sampler, in_axes = (1, None, None, 0), out_axes = 1)
local_sampler = vmap(local_sampler, (None, 0, None, 1), out_axes = 0)

def pcax_recover(state, ens):
    return pcax.recover(state, ens)
pcax_recover = vmap(pcax_recover, (None, 0))

def phi_project(ens, phi):
    return ens @ phi
vphi_project = vmap(phi_project, (0, None))

### not vectorized, have to loop
def phi_band(ens, depths, thresh):
    sub_ens = ens
    sub_ens = sub_ens.at[depths < thresh].set(jnp.nan)
    lower = jnp.nanmin(sub_ens, axis = 0)
    upper = jnp.nanmax(sub_ens, axis = 0)
    return jnp.stack([lower, upper])