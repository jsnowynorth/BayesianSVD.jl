######################################################################
#### Helper Functions
######################################################################

"""
    hpd(x; p=0.95)

Computes the highest posterior density interval of `x`.

# Arguments
- x: vector of data.

# Optional Arguments
- p: value between 0 and 1 for the probability level.

# Examples
```@example
x = [1, 3, 2, 5, .2, 1,9]
hpd(x)
``` 
"""
function hpd(x; p=0.95)

  if isempty(x)
    return [0,0]
  else
    x_sorted = sort(x)
    n = length(x)
    num_to_keep = Int(ceil(p * n))
    num_to_drop = n - num_to_keep

    possible_starts = range(1, num_to_drop + 1, step=1)
    # Just count down from the other end
    possible_ends = range(n - num_to_drop, n, step=1)

    # Find smallest interval
    span = x_sorted[possible_ends] - x_sorted[possible_starts]
    edge = argmin(span)
    edges = [possible_starts[edge], possible_ends[edge]]

    return x_sorted[edges]
  end
    
end

function hpd(d::Array{Array{Float64, 2}, 1}, p::Real)

  dat = reshape(reduce(hcat, d), size(d[1], 1), size(d[1], 2), size(d, 1))
  return permutedims(reshape(hcat([hpd(dat[i,j,:], p = p) for i in axes(dat, 1), j in axes(dat, 2)]...), 2,  size(d[1], 1), size(d[1], 2)), (2, 3, 1))

end

function hpd(d::Array{Float64, 3}, p::Real)

  return permutedims(reshape(hcat([hpd(d[i,j,:], p = p) for i in axes(d, 1), j in axes(d, 2)]...), 2,  size(d, 1), size(d, 2)), (2, 3, 1))

end


function Statistics.quantile(d::Array{Float64, 2}, p::Real)

  return [Statistics.quantile(d[i,:], p) for i in axes(d, 1)]

end

function Statistics.quantile(d::Array{Array{Float64, 2}, 1}, p::Real)

  dat = reshape(reduce(hcat, d), size(d[1], 1), size(d[1], 2), size(d, 1))
  return [Statistics.quantile(dat[i,j,:], p) for i in axes(dat, 1), j in axes(dat, 2)] 

end

function Statistics.quantile(d::Array{Float64, 3}, p::Real)

  return [Statistics.quantile(d[i,j,:], p) for i in axes(d, 1), j in axes(d, 2)]

end


function posteriorCoverage(c::Vector{Float64}, d::Array{Float64, 2}, p::Real; returnData = false)

  upper = round(p + (1-p)/2, digits = 5)
  lower = round((1-p)/2, digits = 5)

  upperCI = quantile(d, upper)
  lowerCI = quantile(d, lower)

  if !returnData
    return sum((lowerCI .< c) .& (c .< upperCI))/length(c)
  else
    return (lowerCI .< c) .& (c .< upperCI)
  end
  
end


function posteriorCoverage(c::Matrix{Float64}, d::Array{Float64, 3}, p::Real; returnData = false)

  upper = round(p + (1-p)/2, digits = 5)
  lower = round((1-p)/2, digits = 5)

  upperCI = quantile(d, upper)
  lowerCI = quantile(d, lower)

  if !returnData
    return sum((lowerCI .< c) .& (c .< upperCI))/length(c)
  else
    return (lowerCI .< c) .& (c .< upperCI)
  end
  
end

function posteriorCoverage(c::Matrix{Float64}, d::Array{Array{Float64, 2}, 1}, p::Real; returnData = false)

  upper = round(p + (1-p)/2, digits = 5)
  lower = round((1-p)/2, digits = 5)

  upperCI = quantile(d, upper)
  lowerCI = quantile(d, lower)

  if !returnData
    return sum((lowerCI .< c) .& (c .< upperCI))/length(c)
  else
    return (lowerCI .< c) .& (c .< upperCI)
  end
  
end





"""
    CreateDesignMatrix(n::Int, m::Int, Xs, Xt, Xst; intercept = true)

Creates a design matrix.

See also [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- n::Int: Number of spatial locations
- m::Int: Number of temporal locations
- Xs: Spatial only covariates of dimension n x pₙ
- Xt: Time only covariates of dimension m x pₘ
- Xst: Space-time only covariates of dimension mn x pₘₙ
- intercept = true: should a global intercept be included

# Returns
- X: Design matrix of dimension mn x (pₘ + pₙ + pₘₙ) or mn x (1 + pₘ + pₙ + pₘₙ)
- Ps: Projection onto the null space spanned by the spatial covariates of dimension pₙ x pₙ
- Pt: Projection onto the null space spanned by the temporal covariates of dimension pₘ x pₘ

# Examples
```
m = 75
n = 50

#### Spatial
Xs = hcat(sin.((2*pi .* x) ./ 10), cos.((2*pi .* x) ./ 10))

#### Temporal
Xt = hcat(Vector(t))

#### Spatio-temporal
Xst = rand(Normal(), n*m, 2)

#### Create design matrix
CreateDesignMatrix(n, m, Xs, Xt, Xst, intercept = true)
``` 
"""
function CreateDesignMatrix(n::Int, m::Int, Xs, Xt, Xst; intercept = true)
    
  if (typeof(Xs) == typeof(Xt) == typeof(Xst) <: Nothing) & !intercept
      throw(error("You have no covariates!! You don't need this!!"))
  end

  Ps = diagm(ones(n))
  Pt = diagm(ones(m))

  X = ones(n*m)
  if typeof(Xs) <: Array
      X = hcat(X, repeat(Xs, outer = (m,1)))
      Ps = diagm(ones(n)) - Xs * inv(Xs' * Xs) * Xs'
  end
  if typeof(Xt) <: Array
      X = hcat(X, repeat(Xt, inner = (n,1)))
      Pt = diagm(ones(m)) - Xt * inv(Xt' * Xt) * Xt'
  end
  if typeof(Xst) <: Array
      X = hcat(X, Xst)
  end

  if intercept
      return X, Ps, Pt
  else
      return X[:,2:end], Ps, Pt
  end

end



"""
    NS(A)

Compute the nullspace of the matrix A.

"""
function NS(A)
  k = size(A, 2)
  svd(A; full = true).U[:,(k+1):(end)]
end