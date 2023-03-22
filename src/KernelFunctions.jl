
######################################################################
#### Structure Sets
######################################################################

abstract type KernelFunction end


mutable struct IdentityKernel <: KernelFunction
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct ExponentialKernel <: KernelFunction
    d::Matrix{Float64}
    ρ::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct GaussianKernel <: KernelFunction
    d::Matrix{Float64}
    ρ::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end

mutable struct MaternKernel <: KernelFunction
    d::Matrix{Float64}
    ρ::Float64
    ν::Float64
    metric::Metric
    K::Array{Float64}
    Kinv::Array{Float64}
    logdet::Float64
end


######################################################################
#### Identity Functions
######################################################################

"""
    IdentityKernel(x)
    IdentityKernel(x, y)
    IdentityKernel(X::Matrix{Float64})

Creates `IdentityKernel <: KernelFunction`

See also [`Data`](@ref) and [`MaternKernel`](@ref).

# Arguments
- x: vector of values at which to evaluate the kernel (dimension 1)
- y: vector of values at which to evaluate the kernel (dimension 2)
- X::Matrix{Float64}: matrix of values at which to evaluate the kernel (all possible combinations)

# Return
- K::Array{Float64}
- Kinv::Array{Float64}
- logdet::Float64

# Examples
```@example
n = 100
x = range(-5, 5, n)
ΩU = IdentityKernel(x)

n = 100
x = range(-5, 5, n)
y = range(-5, 5, n)
ΩU = IdentityKernel(x, y)

locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
ΩU = IdentityKernel(locs')
``` 
"""
function IdentityKernel(x)

    K = diagm(ones(length(x)))
    IdentityKernel(K, K, logdet(K))

end

function IdentityKernel(x, y)

    K = diagm(ones(length(x)*length(y)))
    IdentityKernel(K, K, logdet(K))
    
end

function IdentityKernel(X::Matrix{Float64})

    K = diagm(ones(size(X, 1)))
    IdentityKernel(K, K, logdet(K))
    
end

Base.copy(C::IdentityKernel) = IdentityKernel(C)

######################################################################
#### Exponential Functions
######################################################################
function ExponentialKernel(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = exp.(-d ./ ρ)
    ExponentialKernel(d, ρ, metric, K, inv(K), logdet(K))

end

function ExponentialKernel(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = exp.(-d./ρ)
    ExponentialKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function ExponentialKernel(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = exp.(-(d.^2)./(ρ^2))
    ExponentialKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function ExponentialKernel(C::ExponentialKernel)

    K = exp.(-(C.d)./(C.ρ))
    ExponentialKernel(C.d, C.ρ, C.metric, K, inv(K), logdet(K))

end

Base.copy(C::ExponentialKernel) = ExponentialKernel(C)


######################################################################
#### Gaussian Functions
######################################################################
function GaussianKernel(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = exp.(-(d .^2 ./ (2*ρ^2)))
    K = K + diagm(0.00000001*ones(length(x)))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))

end

function GaussianKernel(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = exp.(-(d .^2 ./ (2*ρ^2)))
    K = K + diagm(0.00000001*ones(size(d,1)))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianKernel(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = exp.(-(d .^2 ./ (2*ρ^2)))
    K = K + diagm(0.00000001*ones(size(d,1)))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianKernel(C::GaussianKernel)
    
    K = exp.(-(C.d .^2 ./ (2*C.ρ^2)))
    K = K + diagm(0.00000001*ones(size(D.d,1)))
    GaussianKernel(C.d, C.ρ, C.metric, K, inv(K), logdet(K))

end

Base.copy(C::GaussianKernel) = GaussianKernel(C)


######################################################################
#### Matern Functions
######################################################################

"""
    MaternKernel(x; ρ = 1, ν = 1, metric = Euclidean())
    MaternKernel(x, y; ρ = 1, ν = 1, metric = Euclidean())
    MaternKernel(X::Matrix{Float64}; ρ = 1, ν = 1, metric = Euclidean())

Creates `MaternKernel <: KernelFunction` with correlation function

```math
K_{\\nu, \\rho}(s, s') = \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}\\left(2\\nu \\frac{||s-s'||}{\\rho}\\right)^{\\nu}J_{\\nu}\\left(2\\nu \\frac{||s-s'||}{\\rho}\\right),
```

for ``s,s' \\in \\mathcal{S}``, where ``\\Gamma`` is the gamma function,
``J_{\\nu}`` is the Bessel function of the second kind, and ``\\{\\rho, \\nu\\}`` 
are hyperparameters that describe the length-scale and differentiability, respectively.


See also [`Data`](@ref) and [`IdentityKernel`](@ref).

# Arguments
- x: vector of values at which to evaluate the kernel (dimension 1)
- y: vector of values at which to evaluate the kernel (dimension 2)
- X::Matrix{Float64}: matrix of values at which to evaluate the kernel (all possible combinations)

# Optional Arguments
- ρ = 1: defines the effective range
- ν = 1: smoothing parameter
- metric = Euclidean(): metric to compute the distance

# Return
- d::Matrix{Float64}
- ρ::Float64
- ν::Float64
- metric::Metric
- K::Array{Float64}
- Kinv::Array{Float64}
- logdet::Float64

# Examples
```@example
n = 100
x = range(-5, 5, n)
ΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())

n = 100
x = range(-5, 5, n)
y = range(-5, 5, n)
ΩU = MaternKernel(x, y, ρ = 4, ν = 4, metric = Distances.Euclidean())

locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
ΩU = MaternKernel(locs', ρ = 4, ν = 4, metric = Distances.Euclidean())
``` 
"""
function MaternKernel(x; ρ = 1, ν = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    # MaternKernel(d, ρ, ν, metric, K, inv(K))
    MaternKernel(d, ρ, ν, metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

function MaternKernel(x, y; ρ = 1, ν = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternKernel(d, ρ, ν, metric, K, inv(K), logdet(K))
    
end

function MaternKernel(X::Matrix{Float64}; ρ = 1, ν = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = (2.0^(1-ν))/gamma(ν) .* (sqrt(2*ν) .* d ./ ρ) .^ ν .* [d[i,j] == 0 ? 1 : besselk(ν, (sqrt(2*ν) * d[i,j] / ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    MaternKernel(d, ρ, ν, metric, K, inv(K), logdet(K))
    
end

function MaternKernel(C::MaternKernel)

    d = C.d
    K = (2.0^(1-C.ν))/gamma(C.ν) .* (sqrt(2*C.ν) .* d ./ C.ρ) .^ C.ν .* [d[i,j] == 0 ? 1 : besselk(C.ν, (sqrt(2*C.ν) * d[i,j] / C.ρ)) for i in axes(d,1), j in axes(d,2)]
    K = K + diagm(ones(size(d,1)))
    # MaternKernel(C.d, C.ρ, C.ν, C.metric, K, inv(K))
    MaternKernel(C.d, C.ρ, C.ν, C.metric, K, Hermitian(inv(cholesky(K).L)'*inv(cholesky(K).L)), logdet(K))

end

Base.copy(C::MaternKernel) = MaternKernel(C)

######################################################################
#### Base Show
######################################################################
Base.show(io::IO, C::IdentityKernel) =
  print(io, "Kernel: Identity")
#

Base.show(io::IO, C::ExponentialKernel) =
  print(io, "Kernel: Exponential\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " └─── Effective range: ", C.ρ, '\n')
#

Base.show(io::IO, C::GaussianKernel) =
  print(io, "Kernel: Gaussian\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " └─── Effective Range: ", C.ρ, '\n')
#

Base.show(io::IO, C::MaternKernel) =
  print(io, "Kernel: Matern\n",
    " ├─── Metric: ", typeof(C.metric), '\n',
    " ├─── Effective Range: ", C.ρ, '\n',
    " └─── ν: ", C.ν, '\n')
#

######################################################################
#### Plotting
######################################################################
@recipe function f(C::KernelFunction; Inv = false)
    seriestype  :=  :contourf
    if Inv
        return C.Kinv
    else
        return C.K
    end
end

# C = IdentityKernel(x)
# C = ExponentialKernel(x; ρ = 1, metric = Euclidean())
# C = GaussianKernel(x; ρ = 1, metric = Euclidean())
# C = MaternKernel(x; ρ = 1, ν = 1, metric = Euclidean())

# C = IdentityKernel(x, y)
# C = ExponentialKernel(x, y; ρ = 1, metric = Euclidean())
# C = GaussianKernel(x, y; ρ = 1, metric = Euclidean())
# C = MaternKernel(x, y; ρ = 1, ν = 1, metric = Euclidean())

