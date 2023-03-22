######################################################################
#### Parameter Structure Sets
######################################################################

# IdentityPars
# ExponentialPars
# GaussianPars
# MaternPars


abstract type Pars end

mutable struct IdentityPars <: Pars 
    U::Array{Float64}
    UZ::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    D::Array{Float64}
    σ::Float64
    σU::Array{Float64}
    σV::Array{Float64}
    propSD::Array{Float64}
    Daccept::Array{Float64}
end

mutable struct ExponentialPars <: Pars 
    U::Array{Float64}
    UZ::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    D::Array{Float64}
    σ::Float64
    σU::Array{Float64}
    σV::Array{Float64}
    ρ::Float64
    propSD::Array{Float64}
    Daccept::Array{Float64}
end

mutable struct GaussianPars <: Pars 
    U::Array{Float64}
    UZ::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    D::Array{Float64}
    σ::Float64
    σU::Array{Float64}
    σV::Array{Float64}
    ρ::Float64
    propSD::Array{Float64}
    Daccept::Array{Float64}
end

mutable struct MaternPars <: Pars 
    U::Array{Float64}
    UZ::Array{Float64}
    V::Array{Float64}
    VZ::Array{Float64}
    D::Array{Float64}
    σ::Float64
    σU::Array{Float64}
    σV::Array{Float64}
    ρU::Vector{Float64}
    ρV::Vector{Float64}
    ν::Float64
    propSD::Array{Float64}
    Daccept::Array{Float64}
    propSU::Array{Float64}
    Uaccept::Array{Float64}
    propSV::Array{Float64}
    Vaccept::Array{Float64}
end



"""
    Pars(data::Data)

Creates the parameter class of type `typeof(data)`.'

See also [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- data: data structure of type `Data`

# Examples
```@example
To Do.
``` 
"""
function Pars(data::IdentityData)

    svdY = svd(data.Y)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    # U = rand(Normal(), data.n, data.k)
    # V = rand(Normal(), data.m, data.k)

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        Nu = nullspace(U[:,inds]')
        Nv = nullspace(V[:,inds]')

        UZ[:,i] = Nu'*U[:,i]
        VZ[:,i] = Nv'*V[:,i]
    end

    D = svdY.S[1:data.k]
    σ = 1.0

    σU = ones(data.k)
    σV = ones(data.k)

    propSD = ones(data.k)
    Daccept = zeros(data.k)

    IdentityPars(U, UZ, V, VZ, D, σ, σU, σV, propSD, Daccept)
end

function Pars(data::ExponentialData)

    svdY = svd(data.Y)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    # U = rand(Normal(), data.n, data.k)
    # V = rand(Normal(), data.m, data.k)

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        Nu = nullspace(U[:,inds]')
        Nv = nullspace(V[:,inds]')

        UZ[:,i] = Nu'*U[:,i]
        VZ[:,i] = Nv'*V[:,i]
    end

    D = svdY.S[1:data.k]
    # D = ones(data.k)
    σ = 1.0

    σU = ones(data.k)
    σV = ones(data.k)

    ρ = 1.0

    propSD = ones(data.k)
    Daccept = zeros(data.k)

    ExponentialPars(U, UZ, V, VZ, D, σ, σU, σV, ρ, propSD, Daccept)
end

function Pars(data::GaussianData)

    svdY = svd(data.Y)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    # U = rand(Normal(), data.n, data.k)
    # V = rand(Normal(), data.m, data.k)

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        Nu = nullspace(U[:,inds]')
        Nv = nullspace(V[:,inds]')

        UZ[:,i] = Nu'*U[:,i]
        VZ[:,i] = Nv'*V[:,i]
    end

    D = svdY.S[1:data.k]
    # D = ones(data.k)
    σ = 1.0

    σU = ones(data.k)
    σV = ones(data.k)

    ρ = 1.0

    propSD = ones(data.k)
    Daccept = zeros(data.k)

    GaussianPars(U, UZ, V, VZ, D, σ, σU, σV, ρ, propSD, Daccept)
end

function Pars(data::MaternData)

    svdY = svd(data.Y)
    U = svdY.U[:,1:data.k]
    V = svdY.V[:,1:data.k]

    # U = rand(Normal(), data.n, data.k)
    # V = rand(Normal(), data.m, data.k)

    UZ = Array{Float64}(undef, data.n - data.k + 1, data.k)
    VZ = Array{Float64}(undef, data.m - data.k + 1, data.k)

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        Nu = nullspace(U[:,inds]')
        Nv = nullspace(V[:,inds]')

        UZ[:,i] = Nu'*U[:,i]
        VZ[:,i] = Nv'*V[:,i]
    end

    D = svdY.S[1:data.k]
    # D = ones(data.k)
    σ = 1.0

    σU = ones(data.k)
    σV = ones(data.k)

    ρU = ones(data.k)
    ρV = ones(data.k)
    ν = 1.0

    propSD = ones(data.k)
    Daccept = zeros(data.k)

    propSU = 0.1*ones(data.k)
    Uaccept = zeros(data.k)
    propSV = 0.1*ones(data.k)
    Vaccept = zeros(data.k)

    MaternPars(U, UZ, V, VZ, D, σ, σU, σV, ρU, ρV, ν, propSD, Daccept, propSU, Uaccept, propSV, Vaccept)
end


