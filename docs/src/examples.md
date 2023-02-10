```@meta
CurrentModule = BayesianSVD
```

# Examples

## Simulating Random Basis Functions

```@setup 1
using BayesianSVD, Random, Plots, Distances

# set seed
Random.seed!(2)
```

We start by simulating a random orthonormal matrix.

```@example 1
# size of matrix is n (locations) by k (basis functions)
n = 100
k = 5

# domain
x = range(-5, 5, n)

# covariance matrix
Σ = MaternKernel(x, ρ = 3, ν = 4, metric = Distances.Euclidean())

# random n by k matrix with structure
Φ = PON(n, k, Σ.K)
nothing # hide
```

```@example 1
# plot the basis functions
Plots.plot(Φ)
savefig("plot.svg")
nothing
```

![](plot.svg)