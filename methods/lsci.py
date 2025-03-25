import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax

def lcdf(rval, q, weight):
    return jnp.sum(weight[:,None] * (rval < q), axis = 0)
lcdf = jit(vmap(lcdf, (None, 0, None)))

def local_weights(yval, ytest, frac = 0.1):
    dmat = jnp.max(jnp.abs(yval - ytest), axis = 1)
    quant = jnp.quantile(dmat, frac, axis = 0)
    indx = dmat < quant
    inner_max = jnp.max(dmat * indx)
    dmat = inner_max - dmat
    w = dmat * indx
    return w / jnp.sum(w, axis = 0)[None,]
local_weights = jit(vmap(local_weights, (None, 0, None)))

def local_tukey(rval, q, weight):
    lcdf_l = jnp.sum(weight * (rval < q))
    lcdf_r = jnp.sum(weight * (rval > q))
    return 2 * jnp.min(jnp.array([lcdf_l, lcdf_r]))
local_tukey = jit(local_tukey)
local_tukey_vmap = vmap(local_tukey, (1, 1, None))
local_tukey_self = vmap(local_tukey, (None, 0, None))
local_tukey_self = jit(vmap(local_tukey_self, (1, 1, None)))

def lsci_state(xval, rval, npc, localization = 'pca'):
    nval = xval.shape[0]
    xval = xval.squeeze()
    rval = rval.squeeze()
    pca_state = pcax.fit(rval, n_components=npc)
    rval2 = pcax.transform(pca_state, rval)
    
    if localization == 'pca':
        xval2 = pcax.transform(pca_state, xval)
    else:
        xval2 = xval.reshape(nval, -1)
    return [rval2, xval2, localization, pca_state]

def lsci(xtest, state, alpha, nsamp, gamma = 0.1, rng = random.PRNGKey(0)):
    rval2, xval2, localization, pca_state = state
    
    nval = rval2.shape[0]
    npc = rval2.shape[1]

    if localization == 'pca':
        xtest2 = pcax.transform(pca_state, xtest[None,])
    else:
        xtest2 = xtest.flatten()[None,]
    
    alpha = 1 - jnp.ceil((gamma*nval + 1) * (1 - alpha))/(gamma*nval)

    weight = local_weights(xval2, xtest2, gamma).squeeze()
    unif = random.uniform(rng, (nsamp, npc))
    quants = jnp.linspace(jnp.min(rval2) - 0.1, jnp.max(rval2) + 0.1, nval+1)

    local_cdfs = lcdf(rval2, quants, weight)
    local_quants = jnp.argmax(local_cdfs[:,:,None] > unif.T[None,:,:], axis = 0)
    local_phi = quants[local_quants].T
    
    dval = jnp.mean(local_tukey_self(rval2, rval2, weight), axis = 0) * jnp.min(local_tukey_self(rval2, rval2, weight), axis = 0)
    qval = jnp.quantile(dval, alpha)
    dphi = jnp.mean(local_tukey_self(rval2, local_phi, weight), axis = 0)

    local_ens = pcax.recover(pca_state, local_phi)
    local_ens = local_ens[dphi > qval]
    return local_ens

def lsci_band(xtest, state, alpha, nsamp, gamma = 0.1):
    local_ens = lsci(xtest, state, alpha, nsamp, gamma = gamma)
    return jnp.min(local_ens, axis = 0), jnp.max(local_ens, axis = 0)