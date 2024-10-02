
######################################################################
#### Structure Sets
######################################################################

abstract type Correlation end
abstract type IndependentCorrelation <: Correlation end
abstract type DependentCorrelation <: Correlation end


mutable struct IdentityCorrelation <: IndependentCorrelation
    ρ::Float64
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct ExponentialCorrelation <: DependentCorrelation
    d::Matrix{Float64}
    ρ::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct GaussianCorrelation <: DependentCorrelation
    d::Matrix{Float64}
    ρ::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct MaternCorrelation <: DependentCorrelation
    d::Matrix{Float64}
    ρ::Float64
    ν::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct SparseCorrelation <: DependentCorrelation
    d::Matrix{Float64}
    ρ::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct ARCorrelation <: DependentCorrelation
    d::Matrix{Float64}
    ρ::Float64
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end


######################################################################
#### Identity Functions
######################################################################
"""
    IdentityCorrelation(x)
    IdentityCorrelation(x, y)
    IdentityCorrelation(X)

Create an Identity correlation matrix of type `IdentityCorrelation <: IndependentCorrelation <: Correlation`.

See also [`ExponentialCorrelation`](@ref), [`GaussianCorrelation`](@ref), [`MaternCorrelation`](@ref), and [`SparseCorrelation`](@ref).

# Arguments
- x: vector of locations
- y: vector of locations for a second dimension
- X: matrix of all pairwise locations

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = IdentityCorrelation(x)
Ω = IdentityCorrelation(x, y)
Ω = IdentityCorrelation(X')
``` 
"""
function IdentityCorrelation(x)

    K = diagm(ones(length(x)))
    IdentityCorrelation(1, K, K, logdet(K))

end

function IdentityCorrelation(x, y)

    K = diagm(ones(length(x)*length(y)))
    IdentityCorrelation(1, K, K, logdet(K))
    
end

function IdentityCorrelation(X::Matrix{Float64})

    K = diagm(ones(size(X, 1)))
    IdentityCorrelation(1, K, K, logdet(K))
    
end

function IdentityCorrelation(C::IdentityCorrelation)

    K = diagm(ones(size(C.K, 1)))
    IdentityCorrelation(1, K, K, logdet(K))
    
end

Base.copy(C::IdentityCorrelation) = IdentityCorrelation(C)

######################################################################
#### Exponential Functions
######################################################################

"""
    ExponentialCorrelation(x; ρ = 1, metric = Euclidean())
    ExponentialCorrelation(x, y; ρ = 1, metric = Euclidean())
    ExponentialCorrelation(X; ρ = 1, metric = Euclidean())

Create an Exponential correlation matrix of type `ExponentialCorrelation <: DependentCorrelation <: Correlation`.

See also [`IdentityCorrelation`](@ref), [`GaussianCorrelation`](@ref), [`MaternCorrelation`](@ref), and [`SparseCorrelation`](@ref).

# Arguments
- x: vector of locations
- y: vector of locations for a second dimension
- X: matrix of all pairwise locations

# Optional Arguments
- ρ = 1: length-scale parameter
- metric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = ExponentialCorrelation(x, ρ = 1, metric = Euclidean())
Ω = ExponentialCorrelation(x, y, ρ = 1, metric = Euclidean())
Ω = ExponentialCorrelation(X', ρ = 1, metric = Euclidean())
``` 
"""
function ExponentialCorrelation(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = exp.(-d ./ ρ)
    ExponentialCorrelation(d, ρ, metric, K, inv(K), logdet(K))

end

function ExponentialCorrelation(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = exp.(-d./ρ)
    ExponentialCorrelation(d, ρ, metric, K, inv(K), logdet(K))
    
end

function ExponentialCorrelation(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = exp.(-(d.^2)./(ρ^2))
    ExponentialCorrelation(d, ρ, metric, K, inv(K), logdet(K))
    
end

function ExponentialCorrelation(C::ExponentialCorrelation)

    K = exp.(-(C.d)./(C.ρ))
    ExponentialCorrelation(C.d, C.ρ, C.metric, K, inv(K), logdet(K))

end

Base.copy(C::ExponentialCorrelation) = ExponentialCorrelation(C)


######################################################################
#### Gaussian Functions
######################################################################

"""
    GaussianCorrelation(x; ρ = 1, metric = Euclidean())
    GaussianCorrelation(x, y; ρ = 1, metric = Euclidean())
    GaussianCorrelation(X; ρ = 1, metric = Euclidean())

Create an Gaussian correlation matrix of type `GaussianCorrelation <: DependentCorrelation <: Correlation`.

See also [`IdentityCorrelation`](@ref), [`ExponentialCorrelation`](@ref), [`MaternCorrelation`](@ref), and [`SparseCorrelation`](@ref).

# Arguments
- x: vector of locations
- y: vector of locations for a second dimension
- X: matrix of all pairwise locations

# Optional Arguments
- ρ = 1: length-scale parameter
- metric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = GaussianCorrelation(x, ρ = 1, metric = Euclidean())
Ω = GaussianCorrelation(x, y, ρ = 1, metric = Euclidean())
Ω = GaussianCorrelation(X', ρ = 1, metric = Euclidean())
``` 
"""
function GaussianCorrelation(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = exp.(-(d .^2 ./ (ρ^2)))
    K = K + diagm(0.00000001*ones(size(d,1)))
    GaussianCorrelation(d, ρ, metric, K, inv(K), logdet(K))

end

function GaussianCorrelation(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = exp.(-(d .^2 ./ (ρ^2)))
    K = K + diagm(0.00000001*ones(size(d,1)))
    GaussianCorrelation(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianCorrelation(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = exp.(-(d .^2 ./ (ρ^2)))
    K = K + diagm(0.00000001*ones(size(d,1)))
    GaussianCorrelation(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianCorrelation(C::GaussianCorrelation)
    
    K = exp.(-(C.d .^2 ./ (C.ρ^2)))
    K = K + diagm(0.00000001*ones(length(diag(K))))
    GaussianCorrelation(C.d, C.ρ, C.metric, K, inv(K), logdet(K))

end

Base.copy(C::GaussianCorrelation) = GaussianCorrelation(C)


######################################################################
#### Matern Functions
######################################################################

"""
    MaternCorrelation(x; ρ = 1, ν = 3.5 metric = Euclidean())
    MaternCorrelation(x, y; ρ = 1, ν = 3.5, metric = Euclidean())
    MaternCorrelation(X; ρ = 1, ν = 3.5, metric = Euclidean())

Create an Matern correlation matrix of type `MaternCorrelation <: DependentCorrelation <: Correlation`.

```math
K_{\\nu, \\rho}(s, s') = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}\\left(2\\nu \\frac{||s-s'||}{\\rho}\\right)^{\\nu}J_{\\nu}\\left(2\\nu \\frac{||s-s'||}{\\rho}\\right),
```
for ``s,s' \\in \\mathcal{S}``, where ``\\Gamma`` is the gamma function,
``J_{\\nu}`` is the Bessel function of the second kind, and ``\\{\\rho, \\nu\\}`` 
are hyperparameters that describe the length-scale and differentiability, respectively.

See also [`IdentityCorrelation`](@ref), [`ExponentialCorrelation`](@ref), [`GaussianCorrelation`](@ref), and [`SparseCorrelation`](@ref).

# Arguments
- x: vector of locations
- y: vector of locations for a second dimension
- X: matrix of all pairwise locations

# Optional Arguments
- ρ = 1: length-scale parameter
- ν = 3.5: smoothness parameter
- metric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = MaternCorrelation(x, ρ = 1, ν = 3.5, metric = Euclidean())
Ω = MaternCorrelation(x, y, ρ = 1, ν = 3.5, metric = Euclidean())
Ω = MaternCorrelation(X', ρ = 1, ν = 3.5, metric = Euclidean())
``` 
"""
function MaternCorrelation(x; ρ = 1, ν = 3.5, metric = Euclidean())

    d = pairwise(metric, x)
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternCorrelation(d, ρ, ν, metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

function MaternCorrelation(x, y; ρ = 1, ν = 3.5, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternCorrelation(d, ρ, ν, metric, K, inv(K), logdet(K))
    
end

function MaternCorrelation(X::Matrix{Float64}; ρ = 1, ν = 3.5, metric = Euclidean())

    d = pairwise(metric, X')
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternCorrelation(d, ρ, ν, metric, K, inv(K), logdet(K))
    
end

function MaternCorrelation(C::MaternCorrelation)

    d = C.d
    K = (2.0^(1-C.ν))/gamma(C.ν) .* (sqrt(2*C.ν) .* d ./ C.ρ) .^ C.ν .* [d[i,j] == 0 ? 1 : besselk(C.ν, (sqrt(2*C.ν) * d[i,j] / C.ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternCorrelation(C.d, C.ρ, C.ν, C.metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

Base.copy(C::MaternCorrelation) = MaternCorrelation(C)


######################################################################
#### Sparse Functions
######################################################################

function CompKern(d, ρ)
    return ((3*(d/ρ)^2) * log((d/ρ) / (1 + sqrt(1 - (d/ρ)^2))) + (2*(d/ρ)^2 + 1) * sqrt(1 - (d/ρ)^2))
end

"""
    SparseCorrelation(x; ρ = 1, metric = Euclidean())
    SparseCorrelation(x, y; ρ = 1, metric = Euclidean())
    SparseCorrelation(X; ρ = 1, metric = Euclidean())

Create an Gaussian correlation matrix of type `SparseCorrelation <: DependentCorrelation <: Correlation`.

See also [`IdentityCorrelation`](@ref), [`ExponentialCorrelation`](@ref), [`GaussianCorrelation`](@ref), and [`MaternCorrelation`](@ref).

# Arguments
- x: vector of locations
- y: vector of locations for a second dimension
- X: matrix of all pairwise locations

# Optional Arguments
- ρ = 1: length-scale parameter
- metric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = SparseCorrelation(x, ρ = 1, metric = Euclidean())
Ω = SparseCorrelation(x, y, ρ = 1, metric = Euclidean())
Ω = SparseCorrelation(X', ρ = 1, metric = Euclidean())
``` 
"""
function SparseCorrelation(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    d[d .> ρ] .= ρ

    K = CompKern.(d, ρ)
    K = replace(K, NaN => 1)

    SparseCorrelation(d, ρ, metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

function SparseCorrelation(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = CompKern.(d, ρ)
    K = replace(K, NaN => 1)

    SparseCorrelation(d, ρ, metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))
    
end

function SparseCorrelation(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = CompKern.(d, ρ)
    K = replace(K, NaN => 1)

    SparseCorrelation(d, ρ, metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))
    
end

function SparseCorrelation(C::SparseCorrelation)

    d = C.d
    K = CompKern.(d, r)
    K = replace(K, NaN => 1)
    SparseCorrelation(C.d, C.ρ, C.metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

Base.copy(C::SparseCorrelation) = SparseCorrelation(C)



######################################################################
#### Autoregressive Correlation Functions
######################################################################

"""
    ARCorrelation(t; ρ = 1, metric = Euclidean())

Create an AR(1) correlation matrix of type `ARCorrelation <: DependentCorrelation <: Correlation`.

See also [`IdentityCorrelation`](@ref), [`ExponentialCorrelation`](@ref), [`GaussianCorrelation`](@ref), and [`MaternCorrelation`](@ref).

# Arguments
- t: vector of times

# Optional Arguments
- ρ = 1: correlation parameter
- metric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)
X = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'

Ω = SparseCorrelation(x, ρ = 1, metric = Euclidean())
Ω = SparseCorrelation(x, y, ρ = 1, metric = Euclidean())
Ω = SparseCorrelation(X', ρ = 1, metric = Euclidean())
``` 
"""
function ARCorrelation(t; ρ = 0.5)

    T = range(1, length(t))
    d = [abs(T[i] - T[j]) for i in axes(T, 1), j in axes(T, 1)]
    K = ρ .^ d

    ARCorrelation(d, ρ, K, inv(K), logdet(K))

end

function ARCorrelation(C::ARCorrelation)

    K = C.ρ .^ C.d
    ARCorrelation(C.d, C.ρ, K, inv(K), logdet(K))

end

Base.copy(C::ARCorrelation) = ARCorrelation(C)



######################################################################
#### Update Correlation Kernel
######################################################################

function updateCorrelation(C::ExponentialCorrelation)
    return ExponentialCorrelation(C)
end

function updateCorrelation(C::MaternCorrelation)
    return MaternCorrelation(C)
end

function updateCorrelation(C::GaussianCorrelation)
    return GaussianCorrelation(C)
end

function updateCorrelation(C::SparseCorrelation)
    return SparseCorrelation(C)
end

function updateCorrelation(C::ARCorrelation)
    return ARCorrelation(C)
end

######################################################################
#### Base Show
######################################################################
Base.show(io::IO, C::IdentityCorrelation) =
  print(io, "Kernel: Identity")
#

Base.show(io::IO, C::ExponentialCorrelation) =
  print(io, "Kernel: Exponential\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " └─── Effective range: ", C.ρ, '\n')
#

Base.show(io::IO, C::GaussianCorrelation) =
  print(io, "Kernel: Gaussian\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " └─── Effective Range: ", C.ρ, '\n')
#

Base.show(io::IO, C::MaternCorrelation) =
  print(io, "Kernel: Matern\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " ├─── Effective Range: ", C.ρ, '\n',
    " └─── ν: ", C.ν, '\n')
#


Base.show(io::IO, C::SparseCorrelation) =
  print(io, "Kernel: Sparse\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " └─── Range: ", C.r, '\n')
#


Base.show(io::IO, C::ARCorrelation) =
  print(io, "Kernel: AR(1)\n",
    " └─── Correlation: ", C.ρ, '\n')
#

######################################################################
#### Plotting
######################################################################
@recipe function f(C::GaussianCorrelation; Inv = false)
    seriestype  :=  :contourf
    if Inv
        return C.Kinv
    else
        return C.K
    end
end

