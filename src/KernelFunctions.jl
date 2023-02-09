
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


######################################################################
#### Gaussian Functions
######################################################################
function GaussianKernel(x; ρ = 1, metric = Euclidean())

    d = pairwise(metric, x)
    K = exp.(-(d.^2)./(ρ^2))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))

end

function GaussianKernel(x, y; ρ = 1, metric = Euclidean())

    Nx = length(x)
    Ny = length(y)
    locs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'
    d = pairwise(metric, locs')
    K = exp.(-(d.^2)./(ρ^2))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianKernel(X::Matrix{Float64}; ρ = 1, metric = Euclidean())

    d = pairwise(metric, X')
    K = exp.(-(d.^2)./(ρ^2))
    GaussianKernel(d, ρ, metric, K, inv(K), logdet(K))
    
end

function GaussianKernel(C::GaussianKernel)

    K = exp.(-(C.d.^2)./(C.ρ^2))
    GaussianKernel(C.d, C.ρ, C.metric, K, inv(K), logdet(K))

end


######################################################################
#### Matern Functions
######################################################################
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

