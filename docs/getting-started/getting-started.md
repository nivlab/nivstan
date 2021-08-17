---
layout: default
title: Getting started
nav_order: 3
has_children: true
permalink: /getting_started/
---

## Getting started

### 1. Syntax

A Stan program defines a statistical model through a conditional probability function $p(\theta|y, x)$, where $\theta$ is a sequence of modeled unknown values (e.g., model parameters, latent variables, missing data, future predictions), y is a sequence of modeled
known values, and x is a sequence of unmodeled predictors and constants (e.g., sizes,
hyperparameters).

#### General
* Comments in stan follow a double backslash `\\`.
* Every line ends with a semicolon `;`.
* Stan supports arithmetic and matrix operations on expressions as well as loops & conditions.
* Stan indexing starts at 1.



#### Program blocks
Stan programs feature several distinct parts named blocks:
1. `functions{}` contains user-defined functions
2. `data{}` declares the required data for the model, i.e the components of the observational space
3. `transformed data{}` defines the changes to the format or structure of the data, as required by the model (if required)
4. `parameters{}` declares the (unconstrained version of the) model’s parameters, which are sampled or optimized.
5. `transformed parameters{}` defines the changes to the format or structure of the parameters, as required by the model (if required)  
6. `model{}` defines the log probability function
7. `generated quantities{}` derives user-defined quantities based on the data and sampled parameters

All of the blocks are optional, but when used - the blocks appear in the above order. Commonly, one uses at least the data, parameters, and model blocks in a basic stan program (see the Bernoulli example).

Program flow (based on the block design):
***TBD***


#### Variable declaration
You have to declare every variable, including its type and shape. Following the structure:
`var_type variable ?dims ?('=' expression);`

Some examples:
* `int N; \\ an integer n (minimal declaration)`
* `int P=4; \\ an integer P of value 4.`
* `int<lower=0, upper=1> M[N]; \\ a constrained integer array of size N.`

See all of Stan’s data types [here](https://mc-stan.org/docs/2_27/reference-manual/overview-of-data-types.html).

#### Sampling statements

Defining a sampling statement can be done in one of two ways:
1. In sampling notation, following `var_name ~ probability_function (parameters);`
or
2. Via an explicit incrementation of the log probability function, following `target += probability_function(var_name| parameters);`

Notice, `target` is NOT a variable but a global parameter, representing the calculated density of the model’s log probability function. It does not need to (but can) be explicitly accessed.

For example, sampling y from a normal distribution with $\mu=0$ and $\sigma=1$.
* `y ~ normal(0,1);`
* `target += normal_lpdf(y | 0, 1);`


### 1.1 Compilation errors

* If facing a compiler error (```CompileError: command 'gcc' failed with exit status 1```), reinstall gcc in conda  `$ conda install -c brown-data-science gcc`.


* Arithmetic operation variable type error(```SYNTAX ERROR, MESSAGE(S) FROM PARSER: No matches for:  real[ ]*real[ ] Expression is ill formed.```). Usually indicate that you are using the wrong input types for a specific operator.



### 2. Compiling & running a stan program:

#### Steps:
1. define the model
2. define the data, in a dictionary
3. complie the model
4. sample
5. optional - diagnose
6. extract the posterior samples

Below in pystan and cmdstanpy

#### Pystan

```python
# import libraries
import pystan

# define model & data
model_code = 'bernoulli.stan'
model_data = {"N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1]}

# compile the model
model = pystan.StanModel(file=model_code, verbose=True)

# sample
model_fit = model.sampling(data=model_data)

# diagnose - there is no diagnose command but make sure to look at the diagnosis params via the model summary.

# results summary
model_fit.stansummary()

# extract posterior samples & create a dataframe
posterior_samples_df = model_fit.draws_as_dataframe()

# calculate the generated quantities:
new_quantities = model.generate_quantities(data=model_data, mcmc_sample=model_fit)

#OR as a dataframe:

new_quantities_df = model.generated_quantities_pd(data=model_data, mcmc_sample=model_fit)
```

#### CmdStanPy

```python
# import libraries
from cmdstanpy import CmdStanModel

# define model & data
model_code = 'bernoulli.stan'
model_data = {"N" : 10, "y" : [0,1,0,0,0,0,0,0,0,1]}

# compile the model
model = CmdStanModel(stan_file=model_code)

# sample
model_fit = model.sample(data=model_data)

# diagnose
model_fit.diagnose()

# results summary
print(model_fit.summary())

# extract posterior samples & create a dataframe
posterior_samples = model_fit.extract()
posterior_samples_df = pd.DataFrame(posterior_samples)
```

tbd - REWRITE A REGRESSION MODEL

### 2.1 Data passing errors

* Assigning wrong value size into a variable(```
SYNTAX ERROR, MESSAGE(S) FROM PARSER: Variable definition base type mismatch```). Make sure the constraints are defined to match the value assigned.



### 3. Debugging & diagnostics
#### A word on debugging
Stan provides very informative warnings and errors. Debugging a failed-to-compile model is relatively straightforward (but not always easy). Notice that the error message includes the link and character when it faces the issue.

See 1.1 and 2.1 for some warning examples.

#### Diagnosis
Diagnosis, unlike debugging, deals with cases where the program finishes running, but there are validity issues with the sampler’s outcome. There is no easy troubleshooting, as these usually indicate some fundamental problem with the model itself.

Most important/common:

* `Rhat`  is an estimate of whether the MCMC chain (the sequence of samples) has converged. Briefly, `rhat` is the approximation of the average variance of 1 chain divided by the variance of all samples from all chains. If the Markov chains converge, these variances should be equal and Rhat~=1. This is a ratio, hence starting rhat>1.1 indicates poor convergence.

* `N_eff` is the effective number of samples (total samples = chains * (iter - warmup)). The effective number of samples is an estimate of how many of these total samples are uncorrelated (independent).

And

* Reaching maximum `treedepth` indicates that NUTS is terminating prematurely to avoid excessively long execution time. This warning poses an efficiency concern (rather than a validity one) BUT it could indicate that you should try rewriting the model or reparameterization it.

* `Divergent transitions` - sampler returns biased estimates (usually due to a mismatch between the step size and the model’s parameter space).

* A low estimated Bayesian Fraction of Missing Information (`E-BFMI`) implies that the adaptation chains likely did not explore the posterior distribution efficiently.

[Some more about stan warnings](https://mc-stan.org/misc/warnings.html).

### Parameterization tricks

#### Recap on hierarchical models
This is a simple normal hierarchical model:

$$eq 1: x_i \sim N(\mu_i,\sigma_i)$$
$$eq 2: \mu_i \sim N(\mu_T,\sigma_T)$$

Where $x_i$ represents a datapoint for subject $i$ , distributed normally with a matching mean and standard deviation. The mean for each subject is, in turn, distributed normally with group level mean and standard deviation (indicated with $T$).

#### Parameterization
**A centered parameterization** implies that the subject-level parameters ($i$) are centered around the group-level parameter ($T$) as seen in equation 2. This makes sense, but it does not always work.

It will work well in case that their data is in abundance, and the data is highly heterogeneous. In this case, the group-level sigma will be large, and the posterior distribution space will be smooth. See fig 1, left below.

However, if we have little data, the group-level sigma will be small, creating a situation where the subject-level parameters become very similar to one another, and the posterior distribution space will be quite sharp. See figure 1, right below.

Why do we care about the posterior distribution geometry? HMC uses global step size independent of location in parameter space. Thus, a smooth geometry allows sampling accurately from the space, while a sharp geometry will probably lead to divergence or biased estimation of the group sigma.
