---
layout: default
title: Install
permalink: /docs/install/
nav_order: 2
has_children: false

---

# Installing Stan (with Python)

{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }

1. TOC
{:toc}
</details>

---
The common python stan interfaces are [pystan](https://pystan.readthedocs.io/en/latest/) or [cmdstanpy](https://cmdstanpy.readthedocs.io/en/stable-0.9.65/getting_started.html) which offer slightly different architectural designs & APIs.

This guide assumes you are using anaconda to manage your python.




## Local

Preinstall requirements:
* Python 3 (via conda).
* a linux-based OS, for Pystan
* If using mac, you may need to update your Xcode command line tools

### Pystan2
Notice, pystan2 is a deprecated software. However pystan3 is not yet stable & less commonly used.
1. Install a C++ compiler: [GCC 9](https://gcc.gnu.org/install/) (or newer) or [Clang 10](https://clang.llvm.org/get_started.html) (or newer).
    a. If using mac, make sure to update your environment variables.
2. Install pystan via your python environment ``` $ conda install pystan```.
3. To run with jupyter notebooks, install [nest-asyncio](https://pypi.org/project/nest-asyncio/).


### CmdStanPy
1. Install cmdstan itself following sections 1.3 - 1.5 of the [cmdstan user guide](https://mc-stan.org/docs/2_26/cmdstan-guide/cmdstan-installation.html#git-clone.section).
2. Install the cmdstanpy package on python ```$ conda install cmdstanpy```.


Technically, you can install cmdstanpy first and call the install_cmdstan utility from python. I would not recommend doing that.


## Module-based
1. Load anaconda ```$ module load anacondapy/5.3.1```
2. Activate your designated stan environment.

### Pystan2
* Install pystan ```$ conda install pystan```

### CmdStanPy
* Load cmdstanpy ```$ module load cmdstan/2.26.1```

See the full CmdStanPy instructions from IT [here](https://gist.github.com/karnigili/c5519b3b62ab494dedf5a0a5a4aebdeb).


## Let's check it is working

We will use the build-in Bernoulli estimator. In python, run (we will go over the meaning of these lines of code later):

### Pystan2

```python
import pystan
model_code = 'bernoulli.stan'
model = pystan.StanModel(file=model_code, verbose=True)
```

### CmdStanPy

```python
from cmdstanpy import CmdStanModel
model_code = 'bernoulli.stan'
model = CmdStanModel(stan_file=model_code)
```


Expect to see a lot of output, but no failure codes. Got it? Amazing, you are all set up. Let’s dive into it:)

If you are facing compiling error, please refer to the [compilation error section](https://nivlab.github.io/nivstan/getting_started/#11-compilation-errors) of this tutorial.
