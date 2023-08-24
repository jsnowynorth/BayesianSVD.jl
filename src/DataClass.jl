

######################################################################
#### Data Structure Sets
######################################################################

abstract type Data end

struct MixedEffectData <: Data
    Z::Array{Float64}
    X::Array{Float64}
    Ps::Array{Float64}
    Pt::Array{Float64}
    ulocs::Array{Float64}
    vlocs::Array{Float64}
    n::Int
    m::Int
    p::Int
    k::Int
end

struct RandomEffectData <: Data
    Z::Array{Float64}
    ulocs::Array{Float64}
    vlocs::Array{Float64}
    n::Int
    m::Int
    k::Int
end


"""
    Data(Y, ulocs, vlocs, k)
    Data(Y, X, ulocs, vlocs, k)

Creates the data class.

See also [`Pars`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- Y: data of dimension n × m
- X: covariate matrix of dimension nm × p
- ulocs: locations for the U basis functions, corresponds to the locations for the rows of Y
- vlocs: locations for the V basis functions, corresponds to the locations for the columns of Y
- k: number of basis functions to keep

# Examples
```
k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)
``` 
"""
function Data(Z::Array{Float64, 2}, X::Array{Float64, 2}, Ps::Array{Float64, 2}, Pt::Array{Float64, 2}, ulocs, vlocs, k::Int)
    n, m = size(Z)
    p = size(X, 2)

    MixedEffectData(Z, X, Ps, Pt, ulocs, vlocs, n, m, p, k)
end


function Data(Z::Array{Float64, 2}, ulocs, vlocs, k::Int)
    n, m = size(Z)

    RandomEffectData(Z, ulocs, vlocs, n, m, k)
end

Base.show(io::IO, data::MixedEffectData) =
  print(io, "Mixed Effect Data Structure\n",
    " ├─── U location range: ", extrema(data.ulocs), '\n',
    " ├─── V locations range: ", extrema(data.vlocs), '\n',
    " ├─── Number of covariates: ", data.p, '\n',
    " ├─── Number of U locations: ", data.n, '\n',
    " ├─── Number of V locations: ", data.m, '\n',
    " └─── Number of basis functions: ", data.k, '\n')
#

Base.show(io::IO, data::RandomEffectData) =
  print(io, "Random Effect Data, Structure\n",
    " ├─── U location range: ", extrema(data.ulocs), '\n',
    " ├─── V locations range: ", extrema(data.vlocs), '\n',
    " ├─── Number of U locations: ", data.n, '\n',
    " ├─── Number of V locations: ", data.m, '\n',
    " └─── Number of basis functions: ", data.k, '\n')
#
