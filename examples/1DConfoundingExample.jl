
########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 10-May-2024
#### Description: Code used for Synthetic example #3: covariates
########################################################################


using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra, Statistics


######################################################################
#### Generate Some Data
######################################################################
#region

m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5
ϵ = 0.01

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Z))
Plots.contourf(x, t, Z', clim = extrema(Z))


########################################################################
#### IID covariates M1
########################################################################
#region

βs = vcat(-2, 0.6)
βt = 1.2
βst = -0.9
β = [βs; βt; βst]

X = rand(Normal(0, 0.2), n*m, length(β))

M = reshape(X * β, n, m)
Z = Z + M

#endregion


########################################################################
#### Space-Time covariates M2
########################################################################
#region

βs = vcat(-2, 0.6)
βt = 1.2
βst = -0.9
β = [βs; βt; βst]

#### Spatial
Xs = rand(MvNormal(zeros(n), ΣU.K), length(βs))
Ws = Xs * βs

Plots.plot(x, Ws)

#### Temporal
Xt = rand(MvNormal(zeros(m), ΣV.K), length(βt))
Wt = Xt * βt

Plots.plot(t, Wt)

#### Spatio-temporal
Ω = MaternCorrelation(x, t, ρ = 1, ν = 3.5, metric = Euclidean())
Xst = rand(MvNormal(zeros(n*m), Ω.K), length(βst))

Wst = reshape(Xst * βst, n, m)

Plots.contourf(t, x, Wst)



#### full covariates

X = CreateDesignMatrix(n, m, Xt, Xs, Xst, intercept = false)
β = vcat(βs, βt, βst)

M = reshape(X * β, n, m)
Z = Z + M

#endregion

########################################################################
#### Space-Time covariates M3
########################################################################
#region

βs = vcat(-2, 0.6)
βt = 1.2
βst = -0.9
β = [βs; βt; βst]

#### Spatial

ΣU2 = MaternCorrelation(x, ρ = 0.3, ν = 3.5, metric = Euclidean())
Xs = rand(MvNormal(zeros(n), ΣU2.K), length(βs))

Ws = Xs * βs

Plots.plot(x, Ws)

#### Temporal

ΣV2 = MaternCorrelation(t, ρ = 0.3, ν = 3.5, metric = Euclidean())
Xt = rand(MvNormal(zeros(m), ΣV2.K), length(βt))

Wt = Xt * βt

Plots.plot(t, Wt)

#### Spatio-temporal

Ω2 = MaternCorrelation(x, t, ρ = 1, ν = 3.5, metric = Euclidean())
Xst = rand(MvNormal(zeros(n*m), Ω2.K), length(βst))

Wst = reshape(Xst * βst, n, m)

Plots.contourf(t, x, Wst)

Plots.heatmap(t, x, reshape(Xst, n, m), c = :balance)

#### full covariates

X = CreateDesignMatrix(n, m, Xt, Xs, Xst, intercept = false)
β = vcat(βs, βt, βst)

M = reshape(X * β, n, m)
Z = Z + M

#endregion



######################################################################
#### Sample
######################################################################
#region

k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, X, x, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)

hcat(round.(posterior.β_hat, digits = 3),
    round.(posterior.β_lower, digits = 3),
    round.(posterior.β_upper, digits = 3))'

trsfrm = ones(k)
for l in 1:k
    if posteriorCoverage(U[:,l], posterior.U[:,l,:], 0.95) > posteriorCoverage(-U[:,l], posterior.U[:,l,:], 0.95)
        continue
    else
        trsfrm[l] = -1.0
    end
end

posteriorCoverage(Matrix((U[:,1:k]' .* trsfrm)'), posterior.U, 0.95)
posteriorCoverage(Matrix((V[:,1:k]' .* trsfrm)'), posterior.V, 0.95)

#endregion