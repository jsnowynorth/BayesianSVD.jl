

######################################################################
#### Data Structure Sets
######################################################################

abstract type Data end

struct IdentityData <: Data
    Y::Array{Float64}
    n::Int
    m::Int
    k::Int
    ΩU::IdentityKernel
    ΩV::IdentityKernel
end

struct ExponentialData <: Data
    Y::Array{Float64}
    n::Int
    m::Int
    k::Int
    ΩU::Vector{ExponentialKernel}
    ΩV::Vector{ExponentialKernel}
end

struct GaussianData <: Data
    Y::Array{Float64}
    n::Int
    m::Int
    k::Int
    ΩU::Vector{GaussianKernel}
    ΩV::Vector{GaussianKernel}
end

struct MaternData <: Data
    Y::Array{Float64}
    n::Int
    m::Int
    k::Int
    ΩU::Vector{MaternKernel}
    ΩV::Vector{MaternKernel}
end


"""
    Data(Y, ΩU::KernelFunction, ΩV::KernelFunction, k)

Creates the data class.

See also [`Pars`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- Y: data of dimension n × m
- ΩU::KernelFunction: Kernel for U matrix, of dimension n × n
- ΩV::KernelFunction: Kernel for V matrix, of dimension m × m
- k: number of basis functions to keep

# To Do
allow for ΩU and ΩV to be of different types (i.e., ΩU is Matern and ΩV is Identity)

# Examples
```@example
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

# Identity Covariances
ΩU = IdentityKernel(x, metric = Distances.Euclidean())
ΩV = IdentityKernel(t, metric = Distances.Euclidean())
data = Data(Y, ΩU, ΩV, k)

# Matern Covariances
ΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())
ΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Distances.Euclidean())
data = Data(Y, ΩU, ΩV, k)
``` 
"""
function Data(Y, ΩU::IdentityKernel, ΩV::IdentityKernel, k)
    n, m = size(Y)

    if isodd(k)
        indstart = Int((k-1)/2+1)
        nend = (n-Int((k-1)/2))
        mend = (m-Int((k-1)/2))
    else
        indstart = Int(ceil((k-1)/2)+1)
        nend = (n-Int(floor((k-1)/2)))
        mend = (m-Int(floor((k-1)/2)))
    end

    ΩU.K = ΩU.K[indstart:nend, indstart:nend]
    ΩU.Kinv = ΩU.Kinv[indstart:nend, indstart:nend]
    ΩV.K = ΩV.K[indstart:mend, indstart:mend]
    ΩV.Kinv = ΩV.Kinv[indstart:mend, indstart:mend]

    IdentityData(Y, n, m, k, ΩU, ΩV)
end

function Data(Y, ΩU::ExponentialKernel, ΩV::ExponentialKernel, k)
    n, m = size(Y)

    if isodd(k)
        indstart = Int((k-1)/2+1)
        nend = (n-Int((k-1)/2))
        mend = (m-Int((k-1)/2))
    else
        indstart = Int(ceil((k-1)/2)+1)
        nend = (n-Int(floor((k-1)/2)))
        mend = (m-Int(floor((k-1)/2)))
    end

    ΩU.d = ΩU.d[indstart:nend, indstart:nend]
    ΩV.d = ΩV.d[indstart:mend, indstart:mend]
    ΩU = ExponentialKernel(ΩU)
    ΩV = ExponentialKernel(ΩV)

    ΣU = [copy(ΩU) for i in 1:k]
    ΣV = [copy(ΩV) for i in 1:k]
    
    ExponentialData(Y, n, m, k, ΣU, ΣV)
end

function Data(Y, ΩU::GaussianKernel, ΩV::GaussianKernel, k)
    n, m = size(Y)

    if isodd(k)
        indstart = Int((k-1)/2+1)
        nend = (n-Int((k-1)/2))
        mend = (m-Int((k-1)/2))
    else
        indstart = Int(ceil((k-1)/2)+1)
        nend = (n-Int(floor((k-1)/2)))
        mend = (m-Int(floor((k-1)/2)))
    end

    ΩU.d = ΩU.d[indstart:nend, indstart:nend]
    ΩV.d = ΩV.d[indstart:mend, indstart:mend]
    ΩU = GaussianKernel(ΩU)
    ΩV = GaussianKernel(ΩV)

    ΣU = [copy(ΩU) for i in 1:k]
    ΣV = [copy(ΩV) for i in 1:k]
    
    GaussianData(Y, n, m, k, ΣU, ΣV)
end

function Data(Y, ΩU::MaternKernel, ΩV::MaternKernel, k)
    n, m = size(Y)

    if isodd(k)
        indstart = Int((k-1)/2+1)
        nend = (n-Int((k-1)/2))
        mend = (m-Int((k-1)/2))
    else
        indstart = Int(ceil((k-1)/2)+1)
        nend = (n-Int(floor((k-1)/2)))
        mend = (m-Int(floor((k-1)/2)))
    end

    ΩU.d = ΩU.d[indstart:nend, indstart:nend]
    ΩV.d = ΩV.d[indstart:mend, indstart:mend]
    ΩU = MaternKernel(ΩU)
    ΩV = MaternKernel(ΩV)

    ΣU = [copy(ΩU) for i in 1:k]
    ΣV = [copy(ΩV) for i in 1:k]
    
    MaternData(Y, n, m, k, ΣU, ΣV)
end


Base.show(io::IO, data::Data) =
  print(io, "Data\n",
    " ├─── n: ", data.n, '\n',
    " ├─── m: ", data.m, '\n',
    " ├─── k: ", data.k, '\n',
    " ├─── ΩU: ", typeof(data.ΩU), '\n',
    " └─── ΩV: ", typeof(data.ΩV), '\n')
#
