import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import pcax
from lsci import *

def prob_don(prob_model, xval, xtest, yval, alpha):
    
    nval = xval.shape[0]
    ntest = xtest.shape[0]
    
    mu_val, sd_val = prob_model(xval)
    mu_test, sd_test = prob_model(xtest)
    
    yval = yval.reshape(nval, -1)
    mu_val = mu_val.reshape(nval, -1)
    sd_val = sd_val.reshape(nval, -1)
    mu_test = mu_test.reshape(ntest, -1)
    sd_test = sd_test.reshape(ntest, -1)
    
    alpha_adj = jnp.ceil((1-alpha) * (nval + 1))/(nval)
    
    score = jnp.abs(yval - mu_val)/sd_val
    q = jnp.quantile(score, alpha_adj, axis = 0)
    
    lower = -sd_test * q[None,]
    upper = sd_test * q[None,]
    return lower, upper