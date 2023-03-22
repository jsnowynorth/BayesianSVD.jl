```@meta
CurrentModule = BayesianSVD
```

# Examples

## 1-D space and 1-D

This example goes over how to simulate random basis functions.

Load some packages that we will need and set a seed.
```@setup 1d
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

# set seed
Random.seed!(2)
```

The function `GenerateData()` will simulate data from the model

$$Z = UDV' + \epsilon,$$

where $Z$ is a n (space) by m (time) matrix, $U$ is a $n \times k$ matrix, $V$ is a $m \times k$ matrix, and $\epsilon \sim N(0, \sigma^2)$.
We start by setting up the desired dimensions for the simulated data.

```@example 1d
# set dimensions
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

# covariance matrices
ΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40 ,20 ,10 ,5 ,2] # sqrt of eigenvalues
k = 5 # number of basis functions 
ϵ = 0.01 # noise

U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
nothing # hide
```

We can then plot the spatial basis functions

Here is a plot of the spatial basis functions.
```@example 1d
Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
```

and temporal basis functions.
```@example 1d
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])
```

Last, we can look at the simulated smooth and noisy data.
```@example 1d
l = Plots.@layout [a b]
p1 = Plots.contourf(x, t, Y', clim = (-1.05, 1.05).*maximum(abs, Z), title = "Smooth", c = :balance)
p2 = Plots.contourf(x, t, Z', clim = (-1.05, 1.05).*maximum(abs, Z), title = "Noisy", c = :balance)
Plots.plot(p1, p2, layout = l, size = (1000, 400), margin = 5Plots.mm, xlabel = "Space", ylabel = "Time")
```
