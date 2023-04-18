```@meta
CurrentModule = BayesianSVD
```

# BayesianSVD

Documentation for [BayesianSVD](https://github.com/jsnowynorth/BayesianSVD.jl).

Our goal is to sample from the model

``Z = X \beta + UDV' + \epsilon,``

where ``Z`` is a n (space) by m (time) matrix, ``X \beta`` is the mean of the process where ``X`` is a matrix of covariates, ``U`` is a ``n \times k`` matrix, ``V`` is a ``m \times k`` matrix, ``D`` is a ``k \times k`` and ``\epsilon \sim N(0, \sigma^2)``.
Here, ``M = X \beta`` is a fixed effect and ``Y = UDV'`` is a random effect.


## General Workflow

The model is broken down into two pieces - `Data` and `Pars` - which are then passed in as arguments to a sampling function which performs MCMC and returns `Posterior`.


### Setting up the data
To set up the model, we first set up our `Data` class, which contains subtypes `MixedEffectData` and `RandomEffectData`.
The difference in the two is `RandomEffectData` assumes ``M \equiv 0`` and is not to be estimated.

To set up a `MixedEffectData` model, we would use
```
data = Data(Z, X, rowLocations, columnLocations, numberBasisFunctions)
```
and To set up a `RandomEffectData` model, we would use
```
data = Data(Z, rowLocations, columnLocations, numberBasisFunctions)
```
where now `X` is omitted from the function call.


### Setting up the parameters
The second thing is to set up the `Pars` class. This is the same for both the `MixedEffectData` and `RandomEffectData`, and requires three arguments: data, the correlation matrix for ``U``, and the correlation matrix for ``V``. The function `Pars()` is simply a constructor for the `Pars` class and will generate ``k`` correlation matrices for ``U`` and ``V`` based on the supplied matrices.

```
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
pars = Pars(data, ΩU, ΩV)
```

### Sampling
To sample from the model, use the `SampleSVD()` function, supplying the `data` and `pars` as inputs. This will return a structure of class `Posterior` which has its own properties.
```
posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)
```