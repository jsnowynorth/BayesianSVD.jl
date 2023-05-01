

"""
    PON(n, k, Σ)

Create an n by k orthonormal matrix with covariance matrix Sigma.

# Arguments
- n: Number of locations 
- k: Number of basis functions
- Σ: covariance matrix

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())

k = 5
Φ = PON(n, k, ΣU.K)
Ψ = PON(n, k, ΣV.K)
``` 
"""
function PON(n, k, Σ)

    Z = Array{Float64}(undef, n, k)
    X = Array{Float64}(undef, n, k)
    A = ones(n)

    for i in 1:k
      z = rand(MvNormal(zeros(n), Σ.K))
      Z[:,i] = z - Σ.K * A * inv(A' * Σ.K * A) * (A'*z)
      # Z[:,i] = rand(MvNormal(zeros(n), Σ.K))
    end
  
    X[:,1] = Z[:,1]
    for i in 2:k
      Xtmp = X[:,1:(i-1)]
      P = Xtmp*inv(Xtmp'*Xtmp)*Xtmp'
      X[:,i] = (diagm(ones(n)) - P) * Z[:,i]
    end
    X = X ./ [norm(X[:,i]) for i in 1:k]'
    return X
end

function PON(n, k, Σ::Vector{T}) where T <: Correlation

  Z = Array{Float64}(undef, n, k)
  X = Array{Float64}(undef, n, k)
  A = ones(n)

  for i in 1:k
    # z = rand(MvNormal(zeros(n), Σ[i].K))
    # Z[:,i] = z - Σ[i].K * A * inv(A' * Σ[i].K * A) * (A'*z)
    Z[:,i] = rand(MvNormal(zeros(n), Σ[i].K))
  end

  X[:,1] = Z[:,1]
  for i in 2:k
    Xtmp = X[:,1:(i-1)]
    P = Xtmp*inv(Xtmp'*Xtmp)*Xtmp'
    X[:,i] = (diagm(ones(n)) - P) * Z[:,i]
  end
  X = X ./ [norm(X[:,i]) for i in 1:k]'

  return X
end


"""
  GenerateData(ΣU::Correlation, ΣV::Correlation, D, k, ϵ; SNR = false)
  GenerateData(ΣU::Vector{T}, ΣV::Vector{T}, D, k, ϵ; SNR = false) where T <: Correlation

Generate random basis functions and data given a correlation structure for ``U`` and ``V``.

# Arguments
- ΣU::Correlation
- ΣU::Correlation
- D: vector of length k
- k: number of basis functions
- ϵ: standard devaiation of the noise, if SNR = true then ϵ is the signal to noise ratio

# Optional Arguments
- SNR = false: Boolean for if ϵ is standard deviation (false) or signal to noise ratio value (true)

# Returns
- U: U basis functions
- V: V basis functions
- Y: True smooth surface
- Z: Noisy "observed" surface

# Examples
```
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 0.1) # standard deviation
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true) # signal to noise
```
"""
function GenerateData(ΣU::Correlation, ΣV::Correlation, D, k, ϵ; SNR = false)

  n = size(ΣU.K,1)
  m = size(ΣV.K,1)

  U = PON(n, k, ΣU)
  V = PON(m, k, ΣV)
  Y = U * diagm(D) * V'

  if SNR

    η = rand(Normal(), n, m)
    # A = ones(n*m)
    # η = reshape(reshape(η, :) - A * inv(A' * A) * (A'*reshape(η, :)), n, m)
  
    σ = sqrt.(var(Y) ./ (ϵ * var(η))) # set the standard deviation
    Z = Y + σ .* η

  else

    η = rand(Normal(0, sqrt(ϵ)), n, m)
    # A = ones(n*m)
    # η = reshape(reshape(η, :) - ϵ * A * inv(A' * ϵ * A) * (A'*reshape(η, :)), n, m)
  
    Z = Y + η

  end

  return U, V, Y, Z
  
end

function GenerateData(ΣU::Vector{T}, ΣV::Vector{T}, D, k, ϵ; SNR = false) where T <: Correlation

  n = size(ΣU[1].K,1)
  m = size(ΣV[1].K,1)

  U = PON(n, k, ΣU)
  V = PON(m, k, ΣV)
  Y = U * diagm(D) * V'

  if SNR

    η = rand(Normal(), n, m)
    # A = ones(n*m)
    # η = reshape(reshape(η, :) - A * inv(A' * A) * (A'*reshape(η, :)), n, m)
  
    σ = sqrt.(var(Y) ./ (ϵ * var(η))) # set the standard deviation
    Z = Y + σ .* η

  else

    η = rand(Normal(0, sqrt(ϵ)), n, m)
    # A = ones(n*m)
    # η = reshape(reshape(η, :) - ϵ * A * inv(A' * ϵ * A) * (A'*reshape(η, :)), n, m)
  
    Z = Y + η

  end

  return U, V, Y, Z
  
end
