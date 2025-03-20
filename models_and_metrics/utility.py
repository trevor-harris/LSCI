import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

import torch

def split_data(data, lag, horizon):
    horizon = horizon-1
    y_t = data[(lag + horizon):][:,None]
    x_t = np.stack([data[(lag-i-1):(-(i+1+horizon))] for i in range(lag)], axis = 1)
    return x_t.copy(), y_t.copy()

def torch2jax(x):
    return jnp.array(x.numpy())

def jax2torch(x):
    return torch.Tensor(np.array(x))

def conv(x, k):
    return jnp.convolve(x, k, mode = 'same')
conv = jit(vmap(conv, (0, None)))