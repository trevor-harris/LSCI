import jax
import jax.numpy as jnp
from jax import vmap, grad, jit, random
from flax import nnx

import torch
import torch.nn as nn
import neuralop as op
from neuralop.models import FNO, SFNO, TFNO

# class ANO_layer(nnx.Module):
#     def __init__(self, width, rngs: nnx.Rngs):
#         self.linear = nnx.Linear(width, width, rngs=rngs)
#         # self.bn = nnx.BatchNorm(dmid, rngs=rngs)
#         # self.dropout = nnx.Dropout(0.2, rngs=rngs)
#         self.linear_out = nnx.Linear(width, width, rngs=rngs)
        
#     def __call__(self, x):
#         # channel mix
#         h = self.linear(x)

#         # spatial mix
#         g = jnp.mean(x, axis = (1, 2))[:,None,None,:]

#         # sum
#         x = h + g
#         x = nnx.relu(x)

#         return self.linear_out(x)

# class encode_layer(nnx.Module):
#     def __init__(self, in_dim, out_dim, rngs):
#         self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)

#     def __call__(self, x):
#         return self.linear(x)

    
# class DeepANO(nnx.Module):
#     def __init__(self, in_dim, width, out_dim, rngs):
#         self.encode_layer = encode_layer(in_dim, width, rngs)
#         self.ano1 = ANO_layer(width, rngs)
#         self.ano2 = ANO_layer(width, rngs)
#         self.ano3 = ANO_layer(width, rngs)
#         self.decode_layer = encode_layer(width, out_dim, rngs)

#     def __call__(self, x):
#         x = self.encode_layer(x)
#         x = self.ano1(x)
#         x = self.ano2(x)
#         x = self.ano3(x)
#         x = self.decode_layer(x)
#         return x
    
    
# class ProbANO(nnx.Module):
#     def __init__(self, in_dim, width, out_dim, rngs):
#         self.mu_ano = DeepANO(in_dim, width, out_dim, rngs)
#         self.sd_ano = DeepANO(in_dim, width, out_dim, rngs)

#     def __call__(self, x):
#         mu = self.mu_ano(x)
#         log_sd = self.sd_ano(x)
#         return mu, nnx.softplus(log_sd)
    
    
# class DropANO(nnx.Module):
#     def __init__(self, in_dim, width, out_dim, drop_prob, rngs):
#         self.encode_layer = encode_layer(in_dim, width, rngs)
#         self.ano1 = ANO_layer(width, rngs)
#         self.ano2 = ANO_layer(width, rngs)
#         self.ano3 = ANO_layer(width, rngs)
#         self.decode_layer = encode_layer(width, out_dim, rngs)
#         self.dropout = nnx.Dropout(drop_prob, rngs=rngs)

#     def __call__(self, x):
#         x = self.encode_layer(x)
#         x = self.ano1(x)
#         x = self.dropout(x)
#         x = self.ano2(x)
#         x = self.dropout(x)
#         x = self.ano3(x)
#         x = self.decode_layer(x)
#         return x
    
# def train_step(model, optimizer, x, y):
#     def loss_fn(model):
#         y_pred = model(x)
#         return jnp.mean((y_pred - y) ** 2)

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(grads)

#     return loss

# def quant_step(model, optimizer, x, y):
#     def loss_fn(model):
#         quant = 1 - 0.1
#         y_pred = model(x)
#         y_abs = jnp.abs(y)
#         resid = y_abs - y_pred
#         loss = jnp.max(jnp.concat([quant * resid, -(1-quant) * resid], axis = 3), axis = 3)
#         return jnp.mean(loss)
    
#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(grads)

#     return loss

# def prob_step(model, optimizer, x, y):
#     def loss_fn(model):
#         mu, sd = model(x)
#         var = sd**2
#         nll = jnp.mean((mu - y)**2 / (2*var) + 0.5 * jnp.log(var))
#         return nll

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     optimizer.update(grads)

#     return loss

# train_step = nnx.jit(train_step)
# quant_step = nnx.jit(quant_step)
# prob_step = nnx.jit(prob_step)



class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, y_hat, y):
        res = y - y_hat
        return torch.mean(torch.max((self.quantile - 1) * res, self.quantile * res))
    
class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, y):
        mu, sd = outputs
        nll = 0.5 * torch.log(2 * torch.pi * sd**2) + 0.5*(y - mu)**2 / sd**2
        return nll.mean()
    
class ProbFNO(nn.Module):
    def __init__(self, in_chan = 1):
        super().__init__()
        self.mu_fno = FNO(n_modes=(16,), hidden_channels=64, in_channels=in_chan, out_channels=1)
        self.sd_fno = FNO(n_modes=(16,), hidden_channels=64, in_channels=in_chan, out_channels=1)

    def __call__(self, x):
        mu = self.mu_fno(x)
        log_sd = self.sd_fno(x)
        return mu, torch.nn.Softplus()(log_sd)


class ProbSFNO(nn.Module):
    def __init__(self, in_chan = 1):
        super().__init__()
        self.mu_fno = FNO(n_modes=(16,32), hidden_channels=64, in_channels=in_chan, out_channels=1)
        self.sd_fno = FNO(n_modes=(16,32), hidden_channels=64, in_channels=in_chan, out_channels=1)

    def __call__(self, x):
        mu = self.mu_fno(x)
        log_sd = self.sd_fno(x)
        return mu, torch.nn.Softplus()(log_sd)


class ProbTFNO(nn.Module):
    def __init__(self, in_chan = 1):
        super().__init__()
        self.mu_fno = TFNO(n_modes=(16,32), hidden_channels=64, in_channels=in_chan, out_channels=1)
        self.sd_fno = TFNO(n_modes=(16,32), hidden_channels=64, in_channels=in_chan, out_channels=1)

    def __call__(self, x):
        mu = self.mu_fno(x)
        log_sd = self.sd_fno(x)
        return mu, torch.nn.Softplus()(log_sd)