# LSCI

Local Spectral Conformal Inference for Operator Models

## Overview

`LSCI` implements the Local Spectral Conformal Inference method [1] in JAX. LSCI was developed to implement locally adaptive conformal inference for operator models, allowing them quantify prediction uncertainty in local exchangeable settings, i.e when the distribution changes over time.

### Installation
```bash
git clone https://github.com/trevor-harris/LSCI
pip install LSCI/
```
### Examples

```python
import jax
import jax.numpy as jnp
from lsci.conformal import lsci

n, p = 2000, 100

data_rng = random.key(0)
data_keys = random.split(data_rng, 6)

xval = random.normal(data_keys[0], (n, p))
yval = 1 + random.normal(data_keys[1], (n, p))
yval_hat = 1.2 + random.normal(data_keys[2], (n, p))

xtest = random.normal(data_keys[3], (n//2, p))
ytest = 1 + random.normal(data_keys[4], (n//2, p))
ytest_hat = 1.2 + random.normal(data_keys[5], (n//2, p))

rval = yval - yval_hat
rtest = ytest - ytest_hat


# pre-compute local weights
local_weights = lsci.localize(xval, xtest, 5)

# evaluate phi-depth
n_proj = 5
depth_val = lsci.local_phi_depth(rval, rval, local_weights, n_proj)
depth_test = lsci.local_phi_depth(rval, rtest, local_weights, n_proj, reduce = True)

# conformal quantiles
alpha = 0.1
quant_val = lsci.local_quantile(depth_val, alpha)

# check coverage
jnp.mean(depth_test > quant_val) # 0.901

# sample ensemble at test point X_i
i = 10
n_samp = 5000
conf_ens = lsci.local_sampler(rval, local_weights[i], alpha, n_samp, n_proj)
```

### Notes

This package is currently under heavy development. This paper has not yet been peer-reviewed and exists as-is as a preprint currently.

### Cite us

If you use `LSCI` in an academic paper, please cite [1]

### References
<a id='1'>[1]</a>
Harris T., Liu, Y.; 
Locally Adaptive Conformal Inference for Operator Models;
[arxiv link](https://arxiv.org/abs/2507.20975)
