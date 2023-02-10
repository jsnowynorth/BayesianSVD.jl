```@meta
CurrentModule = BayesianSVD
```

# Examples

## Simulating Random Basis Functions


Load some packages that we will need and set a seed.
```@setup 1d
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

# set seed
Random.seed!(2)
```

We start by simulating some random data from the model
$$Z = UDV' + \epsilon,$$
where $Z$ is a n (space) by m (time) matrix, $U$ is a $n \times k$ matrix, $V$ is a $m \times k$ matrix, and $\epsilon \sim N(0, \sigma^2)$.

```@example 1d
# set dimensions
n = 100
m = 50
k = 5

# domains
x = range(-5, 5, n)
t = range(0, 10, m)

# covariance matrix
ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Distances.Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Distances.Euclidean())

# random n by k matrix with structure
Φ = PON(n, k, ΣU.K)
Ψ = PON(m, k, ΣV.K)
```

```@example 1d
Plots.plot(x, Φ, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
```

```@example 1d
Plots.plot(t, Ψ, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])
```
