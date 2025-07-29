import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random

def risk(lower, upper, residual):
    return jnp.mean((residual > lower)*(residual < upper))


def dcorr(x, y, eps = 1e-6):
    var_x = jnp.var(x)
    var_y = jnp.var(y)
    if (var_x < eps) & (var_y > eps):
        return 0
    elif (var_y < eps) & (var_x > eps):
        return 0
    elif (var_x < eps) & (var_y < eps):
        return 0
    else:
        a = jnp.abs(x[:,None] - x[None,:])
        b = jnp.abs(y[:,None] - y[None,:])
        a = a - jnp.mean(a, axis = 0)[None,] - jnp.mean(a, axis = 1)[:,None] + jnp.mean(a)
        b = b - jnp.mean(b, axis = 0)[None,] - jnp.mean(b, axis = 1)[:,None] + jnp.mean(b)
        return jnp.mean(a*b) / jnp.sqrt(jnp.mean(a**2)*jnp.mean(b**2))