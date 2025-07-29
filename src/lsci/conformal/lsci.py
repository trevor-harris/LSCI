import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import jaxwt as jwt
import pcax


## localizers
##### helpers
def linf_dist(x, y):
    return jnp.max(jnp.abs(x - y))
linf_dist = jit(vmap(vmap(linf_dist, (0, None)), (None, 0)))

def l2_dist(x, y):
    return jnp.sqrt(jnp.mean((x - y)**2))
l2_dist = jit(vmap(vmap(l2_dist, (0, None)), (None, 0)))

##### main code
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
    
##### main function
def localize(xval, xtest, lam, local_fn = 'linf', rng = random.key(0), sigma = 0.0):
    noise = sigma*random.normal(rng, xtest.shape)
    xtest += noise
    
    if local_fn == 'linf':
        local_weights = linf_localizer(xval, xtest, lam)
    elif local_fn == 'l2':
        local_weights = l2_localizer(xval, xtest, lam)
    elif local_fn == 'knn':
        local_weights = knn_localizer(xval, xtest, lam)
    else:
        local_weights = None
    return local_weights

## projectors
##### helpers
def phi_slice(rng, p1, p2):
    phi = random.normal(rng, (p1, p2))
    phi = phi / jnp.sum(phi, axis = 1)[:,None]
    return phi

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

##### main code 
def rand_project(x, n_phi, rng):
    dim = x.shape[-1]
    phi = phi_slice(rng, dim, n_phi)
    return x @ phi

def fpca_project(x, n_fpc):
    unif_w = jnp.ones(n)/n
    state = weighted_fpca(rval_fno, unif_w)
    state[2] = state[2][:,0:n_fpc]
    return transform_fpca(x, state)

def wave_project(x, n_level):
    return jwt.wavedec(x, 'haar', mode='zero', level=n_level)[0]

##### main function
def project(x, n_proj, rng, proj_fn = 'rand'):
    if proj_fn == 'rand':
        proj = rand_project(x, n_proj, rng)
    elif proj_fn == 'fpca':
        proj = fpca_project(x, n_proj)
    elif proj_fn == 'wave':
        proj = wave_project(x, n_proj)
    elif proj_fn == 'rand-fpca':
        proj = fpca_project(x, n_proj)
        proj = rand_project(proj, n_proj, rng)
    elif proj_fn == 'rand-wave':
        proj = wave_project(x, n_proj)
        proj = rand_project(proj, 5*n_proj, rng)
    else: return None
    return proj

## Depths
##### helpers
def weighted_tukey1d(x_ref, x_query, weights):
    f1 = jnp.sum(weights * (x_ref < x_query))
    f2 = jnp.sum(weights * (x_ref > x_query))
    f = jnp.vstack([f1, f2])
    return 2*jnp.min(f)
weighted_tukey1d = vmap(weighted_tukey1d, (None, 0, None))
weighted_tukey1d = vmap(weighted_tukey1d, (1, 1, None))
weighted_tukey1d = jit(weighted_tukey1d)

def weighted_norm1d(x_ref, x_query, weights):
    f = jnp.sum(weights * jnp.abs(x_ref - x_query))
    f = jnp.nan_to_num(f)
    return 1/(1 + f)
weighted_norm1d = vmap(weighted_norm1d, (None, 0, None))
weighted_norm1d = vmap(weighted_norm1d, (1, 1, None))
weighted_norm1d = jit(weighted_norm1d)

def weighted_mahal1d(x_ref, x_query, weights):
    f = jnp.sum(weights * jnp.abs(x_ref - x_query) / jnp.std(x_ref))
    f = jnp.nan_to_num(f)
    return 1/(1 + f)
weighted_mahal1d = vmap(weighted_mahal1d, (None, 0, None))
weighted_mahal1d = vmap(weighted_mahal1d, (1, 1, None))
weighted_mahal1d = jit(weighted_mahal1d)

##### main code
def local_tukey(x_ref, x_query, weights):
    depth = jnp.min(weighted_tukey1d(x_ref, x_query, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
local_tukey = vmap(local_tukey, (None, None, 0))
local_tukey = jit(local_tukey)

def local_norm(x_ref, x_query, weights):
    depth = jnp.min(weighted_norm1d(x_ref, x_query, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
local_norm = vmap(local_norm, (None, None, 0))
local_norm = jit(local_norm)

def local_mahal(x_ref, x_query, weights):
    depth = jnp.min(weighted_mahal1d(x_ref, x_query, weights), axis = 0)
    return jnp.clip(depth, 0, 1)
local_mahal = vmap(local_mahal, (None, None, 0))
local_mahal = jit(local_mahal)

#### main functions
def local_depth(x_ref, x_query, weights, depth_fn = 'tukey'):
    if depth_fn == 'tukey':
        depth = local_tukey(x_ref, x_query, weights)
    elif depth_fn == 'norm':
        depth = local_norm(x_ref, x_query, weights)
    elif depth_fn == 'mahal':
        depth = local_mahal(x_ref, x_query, weights)
    else:
        return None
    return depth

def local_phi_depth(x_ref, x_qry, weights, n_proj, reduce = False, rng = random.key(0), proj_fn = 'rand', depth_fn = 'tukey'):
    phi_ref = project(x_ref, n_proj, rng, proj_fn = proj_fn)
    phi_qry = project(x_qry, n_proj, rng, proj_fn = proj_fn)
    depths = local_depth(phi_ref, phi_qry, weights, depth_fn = depth_fn)

    if reduce:
        depths = jnp.diag(depths)
    return depths

def local_quantile(depths, alpha):
    n = depths.shape[0]
    alpha_conf = jnp.ceil((n+1)*(1-alpha))/n
    conf_k = jnp.quantile(depths, 1-alpha_conf, axis = 1)
    return conf_k

## Sampler
#### helper
def local_1dsampler(x, weights, n_samp, rng):
    idx = jnp.argsort(x)
    quants = x[idx]
    weights = weights.squeeze()[idx]
    local_cdf = jnp.cumsum(weights/jnp.sum(weights))
    unif = random.uniform(rng, n_samp)
    samples = quants[jnp.argmax(local_cdf[:,None] >= unif, axis = 0)]
    return samples
local_1dsampler = vmap(local_1dsampler, in_axes = (1, None, None, 0), out_axes = 1)

def spectral_sampler(rval, weights, n_samp, rng):
    state = weighted_fpca(rval, weights.squeeze())
    rval_phi = transform_fpca(rval, state)
    n_proj = rval_phi.shape[1]
    
    ens_fpc = local_1dsampler(rval_phi, weights, n_samp, random.split(rng, n_proj))
    ens_fpc = ens_fpc.squeeze()
    ens = recover_fpca(ens_fpc, state)
    return ens

def depth_filter(rval, ens, weights, alpha, n_proj, rng, proj_fn = 'rand', depth_fn = 'tukey'):
    depth_val = local_phi_depth(rval, rval, weights, n_proj, reduce = False, rng = rng, proj_fn = proj_fn, depth_fn = depth_fn)
    depth_ens = local_phi_depth(rval, ens, weights, n_proj, reduce = False, rng = rng, proj_fn = proj_fn, depth_fn = depth_fn)

    depth_val = depth_val.squeeze()
    depth_ens = depth_ens.squeeze()
    
    quant_val = local_quantile(depth_val[None], alpha)
    return ens[depth_ens >= quant_val]

def local_sampler(rval, weight, alpha, n_samp, n_proj, rng = random.key(0), proj_fn = 'rand', depth_fn = 'tukey'):
    
    # sample candidate ensemble
    weight = weight[None]
    ens = spectral_sampler(rval, weight, n_samp, rng)

    # apply depth filter
    ens = depth_filter(rval, ens, weight, alpha, n_proj, rng, proj_fn = proj_fn, depth_fn = depth_fn)
    return ens














    