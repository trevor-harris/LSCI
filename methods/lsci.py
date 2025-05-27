import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax

## random slicer
def phi_slice(rng, p1, p2):
    phi = random.normal(rng, (p1, p2))
    phi = phi / jnp.sum(phi, axis = 1)[:,None]
    return phi

#  localization metrics 
def linf_dist(x, y):
    return jnp.max(jnp.abs(x - y))
linf_dist = jit(vmap(vmap(linf_dist, (0, None)), (None, 0)))

def l2_dist(x, y):
    return jnp.sqrt(jnp.mean((x - y)**2))
l2_dist = jit(vmap(vmap(l2_dist, (0, None)), (None, 0)))

# localizers
def linf_localizer(xval, xtest, lam):
    dist = linf_dist(xval, xtest)
    dist = jnp.exp(-lam * dist)
    local_weights = dist / (jnp.exp(-lam) + jnp.sum(dist, axis = 1, keepdims=True))
    return local_weights

def l2_localizer(xval, xtest, lam):
    dist = l2_dist(xval, xtest)
    dist = jnp.exp(-lam * dist)
    local_weights = dist / (jnp.exp(-lam) + jnp.sum(dist, axis = 1, keepdims=True))
    return local_weights

def knn_localizer(xval, xtest, lam):
    dist = linf_dist(xval, xtest)
    qdist = jnp.quantile(dist, 1/(1+lam))
    dist = jnp.exp(-dist.at[dist > qdist].set(jnp.inf))
    local_weights = dist / (jnp.exp(-1/(1+lam)) + jnp.sum(dist, axis = 1, keepdims=True))
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
def weighted_fpca(x, w):
    mu = x.T @ w
    x = (x - mu[None])
    sig = x.T @ jnp.diag(w) @ x
    vals, comps = jnp.linalg.eigh(sig)
    comps = jnp.flip(comps, axis = 1)
    state = [mu, vals, comps]
    return state
weighted_fpca = jit(weighted_fpca)

def transform_fpca(x, state):
    mu, vals, comps = state
    x = (x - mu[None])
    return x @ comps
transform_fpca = jit(transform_fpca)

def recover_fpca(fpc, state):
    mu, vals, comps = state
    x = fpc @ comps.T
    x = x + mu[None]
    return x
recover_fpca = jit(recover_fpca)

def phi_project(ens, phi):
    return ens @ phi
vphi_project = vmap(phi_project, (0, None))

def local_sampler(x, weights, n_samp, rng):
    idx = jnp.argsort(x)
    quants = x[idx]
    weights = weights.squeeze()[idx]
    local_cdf = jnp.cumsum(weights/jnp.sum(weights))
    unif = random.uniform(rng, n_samp)
    samples = quants[jnp.argmax(local_cdf[:,None] >= unif, axis = 0)]
    return samples
local_sampler = vmap(local_sampler, in_axes = (1, None, None, 0), out_axes = 1)

def fpca_sampler(rval, weights, n_samp, rng):
    state = weighted_fpca(rval, weights.squeeze())
    rval_phi = transform_fpca(rval, state)
    n_proj = rval_phi.shape[1]
    
    ens_fpc = local_sampler(rval_phi, weights, n_samp, random.split(rng, n_proj))
    ens_fpc = ens_fpc.squeeze()
    ens = recover_fpca(ens_fpc, state)
    return ens

def depth_reject(rval, ens, weights, n_phi, alpha, rng):
    p = rval.shape[1]
    phi = phi_slice(rng, p, n_phi)
    rval_phi = phi_project(rval, phi)
    ens_phi = phi_project(ens, phi)

    depth_val = phi_tukey(rval_phi, rval_phi, weights).squeeze()
    quant_val = jnp.quantile(depth_val, 1-alpha)
    depth_ens = phi_tukey(rval_phi, ens_phi, weights).squeeze()
    return ens[depth_ens >= quant_val]


# ### sampler
# def local_cdf(x, q, weights):
#     f = jnp.sum(weights * (x < q))
#     return f
# local_cdf = jit(vmap(local_cdf, (None, 0, None)))

# def local_quantile(x, cdf, unif):
#     return x[jnp.argmax(cdf > unif)]
# local_quantile = jit(vmap(local_quantile, (None, None, 0)))

# def local_sampler(x, weights, n_samp, rng):
#     quants = jnp.linspace(jnp.min(x) - 0.1, jnp.max(x) + 0.1, x.shape[0]+1)
#     unif = random.uniform(rng, n_samp)
#     lcdf = local_cdf(x, quants, weights.T)
#     return local_quantile(x, lcdf, unif)
# local_sampler = vmap(local_sampler, in_axes = (1, None, None, 0), out_axes = 1)
# local_sampler = vmap(local_sampler, (None, 0, None, 1), out_axes = 0)

# def pcax_recover(state, ens):
#     return pcax.recover(state, ens)
# pcax_recover = vmap(pcax_recover, (None, 0))

# def phi_project(ens, phi):
#     return ens @ phi
# vphi_project = vmap(phi_project, (0, None))

# ### not vectorized, have to loop
# def phi_band(ens, depths, thresh):
#     sub_ens = ens
#     sub_ens = sub_ens.at[depths < thresh].set(jnp.nan)
#     lower = jnp.nanmin(sub_ens, axis = 0)
#     upper = jnp.nanmax(sub_ens, axis = 0)
#     return jnp.stack([lower, upper])