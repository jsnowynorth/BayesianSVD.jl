```@meta
CurrentModule = BayesianSVD
```

# Examples

## Simulating Random Basis Functions

We start by simulating a random orthonormal matrix.

```julia
julia> n = 100
julia> x = range(-5, 5, n)
julia> Σ = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
julia> k = 5
julia> Φ = PON(n, k, Σ.K)
julia> Plots.plot(Φ)
```