import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax
from lsci import *

def quant_don(quant_model, xval, xtest, yval, alpha):
    
    nval = xval.shape[0]
    ntest = xtest.shape[0]
    
    yval = yval.reshape(nval, -1)
    
    quant_val = quant_model(xval).reshape(nval, -1)
    quant_test = quant_model(xtest).reshape(ntest, -1)
    
    alpha_adj = jnp.ceil((1-alpha) * (nval + 1))/(nval)
    
    lower = -quant_val - yval
    upper = yval - quant_val
    score = jnp.max(jnp.stack([lower, upper], axis = 2), axis=2)
    q = jnp.quantile(score, alpha_adj, axis = 0)
    
    lower = -quant_test - q[None,]
    upper = quant_test + q[None,]
    return lower, upper