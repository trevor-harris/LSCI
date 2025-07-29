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

First, we sample synethic regression data to mimic a standard conformal inference scenario. In this scenario, each covariate function is observed at `p = 100` sample points and consists of pure white noise. The residuals are simulated as pure white noise with a small bias. 
```python
import jax
import jax.numpy as jnp
from lsci.conformal import lsci

# generate data and predictions
n, p = 2000, 100
data_rng = random.key(0)
data_keys = random.split(data_rng, 4)

# covariates
xval = random.normal(data_keys[0], (n, p))
xtest = random.normal(data_keys[1], (n//2, p))

# prediction residuals
rval = 0.5 + random.normal(data_keys[2], (n, p))
rtest = 0.5 + random.normal(data_keys[3], (n//2, p))
```

Then, we compute the theoretical prediction sets and evaluate the out of sample coverage. This does not require any sampling.
```python
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
```

Finally, we can generate a ensemble representation of the prediction set using spectral sampling and a depth filter to reject out-of-band samples. The ensemble is computed at a single time point.
```python
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
