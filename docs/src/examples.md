```@meta
CurrentModule = BayesianSVD
```

# Examples

## 1-D space and 1-D

This example goes over how to simulate random basis functions and how to parameterize and fit the Bayesian SVD model.

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
m = 100
k = 5

# domains
x = range(-5, 5, n)
t = range(0, 10, m)

# covariance matrix
ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Distances.Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Distances.Euclidean())

# random n by k matrix with structure
U = PON(n, k, ΣU.K)
V = PON(m, k, ΣV.K)
nothing # hide
```

Here is a plot of the spatial basis functions.
```@example 1d
Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
```

Here is a plot of the temporal basis functions.
```@example 1d
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])
```

We just need to specify the diagonal components, noise, and put it all together.
```@example 1d
# sqrt of eigenvalues
D = diagm([40, 20, 10, 5, 2])
ϵ = rand(Normal(0, sqrt(0.01)), n, m)
Z = U * D * V' + ϵ # n × m
nothing # hide
```

Plot of the smooth "true" data and the noisy data.
```@example 1d
l = Plots.@layout [a b]
p1 = Plots.contourf(x, t, (U * D * V')', clim = (-1.05, 1.05).*maximum(abs, Z), title = "Smooth", c = :balance)
p2 = Plots.contourf(x, t, Z', clim = (-1.05, 1.05).*maximum(abs, Z), title = "Noisy", c = :balance)
Plots.plot(p1, p2, layout = l, size = (1000, 400), margin = 5Plots.mm, xlabel = "Space", ylabel = "Time")
```


To sample from the Bayesian SVD model we use the `SampleSVD()` function (see `?SampleSVD()` for help). This function requires two parameters - `data::Data` and `pars::Pars`. The parameter function is easy, is requres the one arguemt `data::Data`. The data function requires 4 arguments - (1) the data matrix, (2) covariance matrix `ΩU::KernelFunction` for the U basis matrix, (3) covariance matrix `ΩV::KernelFunction` for the V basis matrix, and (4) the number of basis functions `k`.

```@example 1d
ΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Euclidean()) # U covariance matrix
ΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Euclidean()) # V covariance matrix
data = Data(Z, ΩU, ΩV, k) # data structure
pars = Pars(data) # parameter structure
nothing # hide
```

We are now ready to sample from the model. Note, we recommend `show_progress = false` when running in a notebook and `show_progress = true` if you have output print in the REPL. Also, the sampler is slow in the notebooks but considerably faster outside of them.

```@example 1d
posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = false)
nothing # hide
```

We can now plot the output of the spatial basis functions
```@example 1d
Plots.plot(posterior, x, size = (1000, 400), basis = 'U', linewidth = 2, c = [:red :green :purple :blue :orange], xlabel = "Space", ylabel = "Value", title = "Spatial Basis Functions", margin = 5Plots.mm)
Plots.plot!(x, (U' .* [1, 1, 1, -1, -1])', label = false, color = "black", linewidth = 2)
Plots.plot!(x, svd(Z).U[:,1:data.k], label = false, linestyle = :dash, linewidth = 2, c = [:red :green :purple :blue :orange])
```

And the temporal basis functions.
```@example 1d
Plots.plot(posterior, t, size = (1000, 400), basis = 'V', c = [:red :green :purple :blue :orange], xlabel = "Time", ylabel = "Value", title = "Temporal Basis Functions", margin = 5Plots.mm)
Plots.plot!(t, (V' .* [1, 1, 1, -1, -1])', label = false, color = "black", linewidth = 2)
Plots.plot!(t, svd(Z).V[:,1:data.k], c = [:red :green :purple :blue :orange], linestyle = :dash, label = false, linewidth = 2)
```