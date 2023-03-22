

"""
    PON(n, k, Sigma)

Create an n by k orthonormal matrix with covariance matrix Sigma.

# Arguments
- n: Number of locations 
- k: Number of basis functions
- Sigma: covariance matrix

# Examples
```@example
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())

k = 5
Φ = PON(n, k, ΣU.K)
Ψ = PON(n, k, ΣV.K)
``` 
"""
function PON(n, k, Sigma)

  Z = Array{Float64}(undef, n, k)
  X = Array{Float64}(undef, n, k)
  A = ones(n)

  for i in 1:k
    z = rand(MvNormal(zeros(n), Sigma))
    Z[:,i] = z - Sigma * A * inv(A' * Sigma * A) * (A'*z)
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

    GenerateData(ΣU::KernelFunction, ΣV::KernelFunction, D, k, ϵ; SNR = false)

Generate data 

# Arguments
- ΣU::KernelFunction
- ΣU::KernelFunction
- D: vector of length K
- k: number of basis functions
- ϵ: standard devaiation of the noise
- SNR = false: should ϵ be considered the desired signal-to-noise ratio instead of the noise standard deviation

# Returns
- U: U basis functions
- V: V basis functions
- Y: True smooth surface
- Z: Noisy "observed" surface

# Examples
```@example
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40 ,20 ,10 ,5 ,2] # sqrt of eigenvalues
k = 5 # number of basis functions 
ϵ = 0.01 # noise

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
``` 

"""
function GenerateData(ΣU::KernelFunction, ΣV::KernelFunction, D, k, ϵ; SNR = false)

  n = size(ΣU.K,1)
  m = size(ΣV.K,1)

  U = PON(n, k, ΣU.K)
  V = PON(m, k, ΣV.K)
  Y = U * diagm(D) * V'

  if SNR

    η = rand(Normal(), n, m)
    A = ones(n*m)
    η = reshape(reshape(η, :) - A * inv(A' * A) * (A'*reshape(η, :)), n, m)
  
    σ = sqrt.(var(Y) ./ (ϵ * var(η))) # set the standard deviation
    Z = Y + σ .* η

  else

    η = rand(Normal(0, sqrt(ϵ)), n, m)
    A = ones(n*m)
    η = reshape(reshape(η, :) - ϵ * A * inv(A' * ϵ * A) * (A'*reshape(η, :)), n, m)
  
    Z = Y + η

  end

  return U, V, Y, Z
  
end