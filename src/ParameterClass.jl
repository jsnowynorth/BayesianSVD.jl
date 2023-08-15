######################################################################
#### Parameter Structure Sets
######################################################################

abstract type Pars end

mutable struct MixedEffectPars <: Pars

    # linear trend
    β::Array{Float64}
    M::Array{Float64}
    m::Vector{Float64}

    # basis functions
    U::Array{Float64}
    UZ::Array{Float64}
    NU::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    NV::Array{Float64}

    # diagonal matrix
    D::Array{Float64}

    # data variance
    σ::Float64

    # basis function variance
    σU::Array{Float64}
    σV::Array{Float64}
    ΩU::Vector{Correlation}
    ΩV::Vector{Correlation}
    NΩUN::Array{Float64}
    NΩVN::Array{Float64}
    NΩUNinv::Array{Float64}
    NΩVNinv::Array{Float64}

    # MCMC tuning parameters
    propSD::Array{Float64}
    Daccept::Array{Float64}
    propSU::Array{Float64}
    Uaccept::Array{Float64}
    propSV::Array{Float64}
    Vaccept::Array{Float64}
end

mutable struct RandomEffectPars <: Pars

    # basis functions
    U::Array{Float64}
    UZ::Array{Float64}
    NU::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    NV::Array{Float64}

    # diagonal matrix
    D::Array{Float64}

    # data variance
    σ::Float64

    # basis function variance
    σU::Array{Float64}
    σV::Array{Float64}
    ΩU::Vector{Correlation}
    ΩV::Vector{Correlation}
    NΩUN::Array{Float64}
    NΩVN::Array{Float64}
    NΩUNinv::Array{Float64}
    NΩVNinv::Array{Float64}

    # MCMC tuning parameters
    propSD::Array{Float64}
    Daccept::Array{Float64}
    propSU::Array{Float64}
    Uaccept::Array{Float64}
    propSV::Array{Float64}
    Vaccept::Array{Float64}
end


"""
    Pars(data::Data, ΩU::Correlation, ΩV::Correlation)

Creates the parameter class of type `typeof(data)`.'

See also [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- data: data structure of type `Data`
- ΩU: data structure of type `Correlation`
- ΩV: data structure of type `Correlation`

# Examples
```
k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)
``` 
"""
function Pars(data::MixedEffectData, ΩU::Correlation, ΩV::Correlation)

    # linear parameters
    # β = zeros(data.p)
    β = inv(data.X' * data.X) * data.X' * reshape(data.Z,:)
    m = data.X * β
    M = reshape(m, data.n, data.m)

    # inital values for basis functions
    svdY = svd(data.Z .- M)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    NU = Array{Float64}(undef, data.n, data.n - data.k + 1, data.k)
    NV = Array{Float64}(undef, data.m, data.m - data.k + 1, data.k)

    NΩUN = Array{Float64}(undef, data.n - data.k + 1, data.n - data.k + 1, data.k)
    NΩVN = Array{Float64}(undef, data.m - data.k + 1, data.m - data.k + 1, data.k)

    NΩUNinv = Array{Float64}(undef, data.n - data.k + 1, data.n - data.k + 1, data.k)
    NΩVNinv = Array{Float64}(undef, data.m - data.k + 1, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        NU[:,:,i] = nullspace(U[:,inds]')
        NV[:,:,i] = nullspace(V[:,inds]')

        UZ[:,i] = NU[:,:,i]'*U[:,i]
        VZ[:,i] = NV[:,:,i]'*V[:,i]

        NΩUN[:,:,i] = NU[:,:,i]' * ΩU.K * NU[:,:,i]
        NΩVN[:,:,i] = NV[:,:,i]' * ΩV.K * NV[:,:,i]

        NΩUNinv[:,:,i] = NU[:,:,i]' * ΩU.K * NU[:,:,i]
        NΩVNinv[:,:,i] = NV[:,:,i]' * ΩV.K * NV[:,:,i]
    end

    # inital values for D
    D = svdY.S[1:data.k]
    σ = 1.0

    # basis function correlation and variance
    σU = ones(data.k)
    σV = ones(data.k)

    ΣU = [copy(ΩU) for i in 1:data.k]
    ΣV = [copy(ΩV) for i in 1:data.k]

    # MCMC tuning parameters
    propSD = ones(data.k)
    Daccept = zeros(data.k)
    propSU = 0.1*ones(data.k)
    Uaccept = zeros(data.k)
    propSV = 0.1*ones(data.k)
    Vaccept = zeros(data.k)

    MixedEffectPars(β, M, m, U, UZ, NU, V, VZ, NV, D, σ, σU, σV, ΣU, ΣV, NΩUN, NΩVN, NΩUNinv, NΩVNinv, propSD, Daccept, propSU, Uaccept, propSV, Vaccept)
end

function Pars(data::RandomEffectData, ΩU::Correlation, ΩV::Correlation)

    # inital values for basis functions
    svdY = svd(data.Z)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    NU = Array{Float64}(undef, data.n, data.n - data.k + 1, data.k)
    NV = Array{Float64}(undef, data.m, data.m - data.k + 1, data.k)

    NΩUN = Array{Float64}(undef, data.n - data.k + 1, data.n - data.k + 1, data.k)
    NΩVN = Array{Float64}(undef, data.m - data.k + 1, data.m - data.k + 1, data.k)

    NΩUNinv = Array{Float64}(undef, data.n - data.k + 1, data.n - data.k + 1, data.k)
    NΩVNinv = Array{Float64}(undef, data.m - data.k + 1, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        NU[:,:,i] = nullspace(U[:,inds]')
        NV[:,:,i] = nullspace(V[:,inds]')

        UZ[:,i] = NU[:,:,i]'*U[:,i]
        VZ[:,i] = NV[:,:,i]'*V[:,i]

        NΩUN[:,:,i] = NU[:,:,i]' * ΩU.K * NU[:,:,i]
        NΩVN[:,:,i] = NV[:,:,i]' * ΩV.K * NV[:,:,i]

        NΩUNinv[:,:,i] = NU[:,:,i]' * ΩU.K * NU[:,:,i]
        NΩVNinv[:,:,i] = NV[:,:,i]' * ΩV.K * NV[:,:,i]
    end

    # inital values for D
    D = svdY.S[1:data.k]
    σ = 1.0

    # basis function correlation and variance
    σU = ones(data.k)
    σV = ones(data.k)

    ΣU = [copy(ΩU) for i in 1:data.k]
    ΣV = [copy(ΩV) for i in 1:data.k]

    # MCMC tuning parameters
    propSD = ones(data.k)
    Daccept = zeros(data.k)
    propSU = 0.1*ones(data.k)
    Uaccept = zeros(data.k)
    propSV = 0.1*ones(data.k)
    Vaccept = zeros(data.k)

    RandomEffectPars(U, UZ, NU, V, VZ, NV, D, σ, σU, σV, ΣU, ΣV, NΩUN, NΩVN, NΩUNinv, NΩVNinv, propSD, Daccept, propSU, Uaccept, propSV, Vaccept)
end


Base.show(io::IO, pars::MixedEffectPars) =
  print(io, "Mixed Effect Parameter Structure\n",
  " ├─── U correlation: ", typeof(pars.ΩU[1]), '\n',
  " └─── V correlation: ", typeof(pars.ΩV[1]))
#

Base.show(io::IO, pars::RandomEffectPars) =
  print(io, "Random Effect Parameter Structure\n",
  " ├─── U correlation: ", typeof(pars.ΩU[1]), '\n',
  " └─── V correlation: ", typeof(pars.ΩV[1]))
#