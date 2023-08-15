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
    CreateDesignMatrix(n::Int, m::Int, Xt, Xs, Xst; intercept = true)

Creates a design matrix.

See also [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

# Arguments
- n::Int: Number of spatial locations
- m::Int: Number of temporal locations
- Xt: Time only covariates of dimension m x pₘ
- Xs: Spatial only covariates of dimension n x pₙ
- Xst: Space-time only covariates of dimension mn x pₘₙ
- intercept = true: should a global intercept be included

# Returns
- X: Design matrix of dimension mn x (pₘ + pₙ + pₘₙ) or mn x (1 + pₘ + pₙ + pₘₙ)



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
CreateDesignMatrix(n, m, Xt, Xs, Xst, intercept = true)
``` 
"""
function CreateDesignMatrix(n::Int, m::Int, Xt, Xs, Xst; intercept = true)
    
    if intercept
        X = hcat(ones(n*m), repeat(Xs, outer = (m,1)), repeat(Xt, inner = (n,1)), Xst)
    else
        X = hcat(repeat(Xs, outer = (m,1)), repeat(Xt, inner = (n,1)), Xst)
        # X = hcat(repeat(Xs, inner = (m,1)), repeat(Xt, outer = (n,1)), Xst)
    end

    return X
end
