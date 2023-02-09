```@meta
CurrentModule = BayesianSVD
```

# Examples

## Simulating Random Basis Functions

We start by simulating a random orthonormal matrix.

```@example
Random.seed!(2)

n = 100
x = range(-5, 5, n)
Σ = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
k = 5
Φ = PON(n, k, Σ.K)
Plots.plot(Φ)


```


<!-- ```@eval
Random.seed!(2)

n = 100
x = range(-5, 5, n)
Σ = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
k = 5
Φ = PON(n, k, Σ.K)
Plots.plot(Φ)
savefig("plot.png")

nothing
```
![](plot.png) -->