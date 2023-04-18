```@meta
CurrentModule = BayesianSVD
```

```@setup 1d
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

# set seed
Random.seed!(2)
```

# Examples

## Simulate Data

The function `GenerateData()` will simulate data from the model

``Z = UDV' + \epsilon,``

where ``Z`` is a n (space) by m (time) matrix, ``U`` is a ``n \times k`` matrix, ``V`` is a ``m \times k`` matrix, ``D`` is a ``k \times k`` and ``\epsilon \sim N(0, \sigma^2)``.
We start by setting up the desired dimensions for the simulated data.

```@example 1d
# set dimensions
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

# covariance matrices
ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 20, 10, 5, 2] # sqrt of eigenvalues
k = 5 # number of basis functions 
ϵ = 2 # noise

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)
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

## 1-D space and 1-D

To sample from the Bayesian SVD model we use the `SampleSVD()` function (see `?SampleSVD()` for help). This function requires two parameters - `data::Data` and `pars::Pars`. The parameter function is easy, is requres the one arguemt `data::Data`. The data function requires 4 arguments - (1) the data matrix, (2) covariance matrix `ΩU::KernelFunction` for the U basis matrix, (3) covariance matrix `ΩV::KernelFunction` for the V basis matrix, and (4) the number of basis functions `k`.

```@example 1d
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean()) # U covariance matrix
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean()) # V covariance matrix
data = Data(Z, x, t, k) # data structure
pars = Pars(data, ΩU, ΩV) # parameter structure
nothing # hide
```

We are now ready to sample from the model. Note, we recommend `show_progress = false` when running in a notebook and `show_progress = true` if you have output print in the REPL. Also, the sampler is slow in the notebooks but considerably faster outside of them.

```@example 1d
posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = false)
nothing # hide
```

We can now plot the output of the spatial basis functions
```@example 1d
Plots.plot(posterior, x, basis = 'U', linewidth = 2, c = [:red :green :purple :blue :orange], xlabel = "Space", ylabel = "Value", title = "Spatial Basis Functions", margin = 5Plots.mm)
Plots.plot!(x, (U' .* [1, 1, 1, 1, 1])', label = false, color = "black", linewidth = 2)
Plots.plot!(x, svd(Z).U[:,1:data.k], label = false, linestyle = :dash, linewidth = 2, c = [:red :green :purple :blue :orange])
```

And the temporal basis functions.
```@example 1d
Plots.plot(posterior, t, basis = 'V', c = [:red :green :purple :blue :orange], xlabel = "Time", ylabel = "Value", title = "Temporal Basis Functions", margin = 5Plots.mm)
Plots.plot!(t, (V' .* [1, 1, 1, 1, 1])', label = false, color = "black", linewidth = 2)
Plots.plot!(t, svd(Z).V[:,1:data.k], c = [:red :green :purple :blue :orange], linestyle = :dash, label = false, linewidth = 2)
```

Last, we can look at the difference between the target smooth surface, our estimate, and the algorithmic estimate.

```@example 1d

svdZ = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'
Z_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
Z_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-1.05, 1.05).*maximum(abs, Z)

l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Z_hat, title = "Y Hat", c = :balance, clim = lims)
p2 = Plots.contourf(t, x, Y, title = "Truth", c = :balance, clim = lims)
p3 = Plots.contourf(t, x, svdZ, title = "Algorithm", c = :balance, clim = lims)
p4 = Plots.contourf(t, x, Z_hat .- Y, title = "Y Hat - Truth", c = :balance, clim = (-0.2, 0.2))
p5 = Plots.contourf(t, x, Z, title = "Observed", c = :balance, clim = lims)
p6 = Plots.contourf(t, x, svdZ .- Y, title = "Algorithm - Truth", c = :balance, clim = (-0.2, 0.2))
Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))
```