

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
  
    for i in 1:k
      Z[:,i] = rand(MvNormal(zeros(n), Sigma))
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