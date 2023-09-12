

using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra, Statistics
using Kronecker

# using CairoMakie
# using DataFrames, DataFramesMeta, Chain, CSV
# using LaTeXStrings


######################################################################
#### Generate Some Data
######################################################################
#region

m = 75
n = 50
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5
ϵ = 0.01

Random.seed!(2)
# U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Z))
Plots.contourf(x, t, Z', clim = extrema(Z))


########################################################################
#### Space-Time covariates
########################################################################
#region


#### Spatial

# βs = vcat(0.2, -0.3)
# Xs = hcat(sin.((2*pi .* x) ./ 10), cos.((2*pi .* x) ./ 10))
βs = vcat(-1, 2)
# Xs = hcat(sin.((2*pi .* x) ./ 10))
Xs = hcat(exp.(-(x .- 1).^2), sin.((2*pi .* x) ./ 5))
Ws = Xs * βs

Plots.plot(x, Ws)

#### Temporal

βt = vcat(0.5)
# Xt = hcat(sin.((2*pi .* t) ./ 10) + cos.((2*pi .* t) ./ 10))
Xt = hcat(Vector(t))
Wt = Xt * βt

Plots.plot(t, Wt)

#### Spatio-temporal

Ω = MaternCorrelation(x, t, ρ = 1, ν = 3.5, metric = Euclidean())
βst = vcat(-0.6, 0.8)
Xst = rand(MvNormal(zeros(n*m), Ω.K), length(βst))
# Xst = rand(Normal(), n*m, length(βst))
Wst = reshape(Xst * βst, n, m)

Plots.contourf(t, x, Wst)




#### Full data set

X = kronecker(Xt, Xs)
β = vcat(-1, 2)
Z = Z + reshape(X * β, n, m)

# Z = Z + reshape(repeat(Ws, m), n, m) + copy(reshape(repeat(Wt, n), m, n)') + Wst
# Z = Z + Wst

Z = Z + reshape(repeat(Ws, m), n, m)
# Z = Z + reshape(repeat(Wt, n), m, n)'
# Z = Z + Wst
# Z = Z + reshape(repeat(Ws, m), n, m) + copy(reshape(repeat(Wt, n), m, n)')
# Z = Z + reshape(repeat(Ws, m), n, m) + copy(reshape(repeat(Wt, n), m, n)') + Wst
Plots.contourf(x, t, Z' - Y')

# Plots.contourf(x, t, (reshape(repeat(Ws, m), n, m) + copy(reshape(repeat(Wt, n), m, n)') + Wst)')

#endregion


######################################################################
#### Sample
######################################################################
#region


# X, Ps, Pt = CreateDesignMatrix(n, m, Xs, Xt, Xst, intercept = false) # all data
X, Ps, Pt = CreateDesignMatrix(n, m, Xs, Xt, nothing, intercept = false) # no space-time
# X, Ps, Pt = CreateDesignMatrix(n, m, Xs, nothing, nothing, intercept = false) # no time or space-time
# X, Ps, Pt = CreateDesignMatrix(n, m, nothing, Xt, nothing, intercept = false) # no space or space-time
β = vcat(βs)
# β = vcat(βt)
# β = vcat(βs, βt)
# β = vcat(βs, βt, βst)

X = kronecker(Xt, Xs)
X = convert(Matrix, X)
Ps = diagm(ones(n))
Pt = diagm(ones(m))

k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, X, Ps, Pt, x, t, k)
pars = Pars(data, ΩU, ΩV)

# pars.β = β
# pars.U = U
# pars.V = V
# pars.D = D


posterior, pars = SampleSVD(data, pars; nits = 5000, burnin = 2500)

posterior.β_hat'
posterior.β_lower'
posterior.β_upper'

posterior.D_hat
posterior.D_lower
posterior.D_upper
posterior.σ_hat
posterior.σU_hat
posterior.σV_hat



P = inv(X' * X) * X'
Pss = inv(Xs' * Xs) * Xs'
Ptt = inv(Xt' * Xt) * Xt'

EY = [P * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]',:) for i in axes(posterior.β, 2)]
Eβ = [posterior.β[:,i] - P * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]',:) for i in axes(posterior.β, 2)]
Vβ = [P * var(reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]',:)) * P' for i in axes(posterior.β, 2)]

δ = reduce(hcat, [rand(MvNormal(Eβ[i], Hermitian(Vβ[i]))) for i in axes(posterior.β, 2)])

Plots.plot(δ', label = false, c = [:blue :red :magenta :orange :green])
Plots.hline!(β, label = false, c = [:blue :red :magenta :orange :green])

Plots.plot(posterior.β', label = false, c = [:blue :red :magenta :orange :green])
Plots.hline!(β, label = false, c = [:blue :red :magenta :orange :green])


mean(δ, dims = 2)

[quantile(δ[i,:], 0.025) for i in axes(δ, 1)]'
[quantile(δ[i,:], 0.975) for i in axes(δ, 1)]'


Plots.plot(reduce(hcat, EY)')



Pss = inv(Xs' * Xs) * Xs'
Ptt = inv(Xt' * Xt) * Xt'
P = inv(X' * X) * X'


delta = reduce(hcat, [posterior.β[:,i] .- P * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :) for i in axes(posterior.β,2)])

# delta1 = [rand(MvNormal(posterior.β[1:2,i], Hermitian(Pss * posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.U[:,:,i]'  * Pss'))) for i in axes(posterior.U, 3)]
# delta2 = [rand(Normal(posterior.β[3,i], (Ptt * posterior.V[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' * Ptt'))) for i in axes(posterior.U, 3)]

# mean(delta1)
# mean(delta2)

β'




delta = reduce(hcat, [posterior.β[:,i] .- P * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :) for i in axes(posterior.β,2)])


SigmaEst = [P * ((reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :)) * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :)' .+ diagm(posterior.σ[i] * ones(n*m))) * P' for i in axes(posterior.U, 3)]
delta = reduce(hcat, [rand(MvNormal(posterior.β[:,i], Hermitian(SigmaEst[i]))) for i in axes(posterior.U, 3)])

# delta = reduce(hcat, [rand(MvNormal(posterior.β[:,i] .- P * reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :), Hermitian(SigmaEst[i]))) for i in axes(posterior.U, 3)])



mean(delta, dims = 2)

[quantile(delta[i,:], 0.025) for i in axes(delta, 1)]'
[quantile(delta[i,:], 0.975) for i in axes(delta, 1)]'
β'

Plots.plot(delta', label = false, c = [:blue :red :magenta :orange :green])
Plots.hline!(β, label = false, c = [:blue :red :magenta :orange :green])

Plots.plot(posterior.β', label = false)
Plots.hline!(β, label = false)

Plots.plot(posterior.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)


trsfrm = ones(k)
for l in 1:k
    if posteriorCoverage(U[:,l], posterior.U[:,l,:], 0.95) > posteriorCoverage(-U[:,l], posterior.U[:,l,:], 0.95)
        continue
    else
        trsfrm[l] = -1.0
    end
end


g = Plots.plot(posterior, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (U' .* trsfrm)', label = false, color = "black", linewidth = 2)
g = Plots.plot!(x, svd(Z).U[:,1:data.k], label = false, linestyle = :dash, linewidth = 1.5, c = [:blue :red :magenta :orange :green])
# save("/Users/JSNorth/Desktop/UplotEst.svg", g)
# save("/Users/JSNorth/Desktop/UplotEst.png", g)

g = Plots.plot(posterior, t, size = (800, 400), basis = 'V',  linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "V")
g = Plots.plot!(t, (V' .* trsfrm)', label = false, color = "black", linewidth = 2)
g = Plots.plot!(t, svd(Z).V[:,1:data.k], c = [:blue :red :magenta :orange :green], linestyle = :dash, label = false, linewidth = 1.5)
# save("/Users/JSNorth/Desktop/VplotEst.svg", g)
# save("/Users/JSNorth/Desktop/VplotEst.png", g)

Up = [Ps * posterior.U[:,:,i] for i in axes(posterior.U, 3)]
Up = reshape(reduce(hcat, Up), 50, 5, 2500)

Plots.plot(Up[50,1,:], title = "U")
Plots.plot!(posterior.U[50,1,:], title = "U")

Plots.plot(posterior.σU', title = "σU", size = (1000, 600))
Plots.plot(posterior.σV', title = "σV", size = (1000, 600))
Plots.plot(posterior.σ, title = "σ")
Plots.plot(posterior.D', title = "D")
Plots.plot(posterior.U[50,:,:]', title = "U")
Plots.hline!(U[50,:] .* trsfrm, c = :black, label = false)
Plots.plot(posterior.V[45,:,:]', title = "V")
Plots.hline!(V[45,:] .* trsfrm, c = :black, label = false)

Plots.histogram(posterior.D')

posteriorCoverage(reshape(U[:,1]' .* trsfrm[1], :), posterior.U[:,1,:], 0.95)
posteriorCoverage(reshape(U[:,2]' .* trsfrm[2], :), posterior.U[:,2,:], 0.95)
posteriorCoverage(reshape(U[:,3]' .* trsfrm[3], :), posterior.U[:,3,:], 0.95)
posteriorCoverage(reshape(U[:,4]' .* trsfrm[4], :), posterior.U[:,4,:], 0.95)
posteriorCoverage(reshape(U[:,5]' .* trsfrm[5], :), posterior.U[:,5,:], 0.95)

posteriorCoverage(Matrix((U[:,1:k]' .* trsfrm)'), posterior.U, 0.95)
posteriorCoverage(Matrix((V[:,1:k]' .* trsfrm)'), posterior.V, 0.95)




svdY = svd(Z - reshape(repeat(Ws, m), n, m)).U[:,1:k] * diagm(svd(Z - reshape(repeat(Ws, m), n, m)).S[1:k]) * svd(Z - reshape(repeat(Ws, m), n, m)).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
# Y_hat = mean(Yest)

Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-1.5, 1.5)


l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Y_hat, title = "Y Hat", c = :balance, clim = (-3, 3))
p2 = Plots.contourf(t, x, Y, title = "Truth", c = :balance, clim = (-3, 3))
p3 = Plots.contourf(t, x, svdY, title = "Algorithm", c = :balance, clim = (-3, 3))
p4 = Plots.contourf(t, x, Y_hat_diff, title = "Y Hat - Truth", c = :balance, clim = (-0.15, 0.15))
p5 = Plots.contourf(t, x, Y, title = "Observed", c = :balance, clim = (-3, 3))
p6 = Plots.contourf(t, x, svdY .- Y, title = "Algorithm - Truth", c = :balance, clim = (-1, 1))
Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))


quantile(reshape(Y_hat_diff, :), Vector(range(0, 1, step = 0.1)))'
quantile(reshape(svdY .- Y, :), Vector(range(0, 1, step = 0.1)))'


# l = Plots.@layout [a b c; d e f]
# p1 = Plots.heatmap(t, x, Y_hat, title = "Y Hat", c = :balance, clim = (-3, 3))
# p2 = Plots.heatmap(t, x, Φ * D * Ψ', title = "Truth", c = :balance, clim = (-3, 3))
# p3 = Plots.heatmap(t, x, svdY, title = "Algorithm", c = :balance, clim = (-3, 3))
# p4 = Plots.heatmap(t, x, Y_hat_diff, title = "Y Hat - Truth", c = :balance, clim = (-0.4, 0.4))
# p5 = Plots.heatmap(t, x, Y, title = "Observed", c = :balance, clim = (-3, 3))
# p6 = Plots.heatmap(t, x, svdY .- (Φ * D * Ψ'), title = "Algorithm - Truth", c = :balance, clim = (-0.4, 0.4))
# Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))

# savefig("./results/oneSpatialDimension/recoveredplots.png")

#endregion


######################################################################
#### Plot Results
######################################################################
#region


#### Plot U basis functions

ubounds = maximum(abs.(extrema(reduce(hcat, (reshape(posterior.U_hat, :), reshape(posterior.U_lower, :), reshape(posterior.U_upper, :))))))
ubounds = (-1.05, 1.05) .* ubounds

vbounds = maximum(abs.(extrema(reduce(hcat, (reshape(posterior.V_hat, :), reshape(posterior.V_lower, :), reshape(posterior.V_upper, :))))))
vbounds = (-1.05, 1.05) .* vbounds

colorlist = [:blue, :red, :magenta, :orange, :green]
LW = 2
transparencyValue = 0.2

#### 5 x 2 basis function plot
g = CairoMakie.Figure(resolution = (1000, 1200))
ax1 = [CairoMakie.Axis(g[i, 1], limits = ((-5,5), ubounds)) for i in 1:5]
ax2 = [CairoMakie.Axis(g[i, 2], limits = ((0,10), vbounds)) for i in 1:5]

ax1[5].xlabel = "Space"
ax2[5].xlabel = "Time"

Ulabels = [L"\textbf{u}_{%$i}" for i in 1:5]
Vlabels = [L"\textbf{v}_{%$i}" for i in 1:5]

for (i, ax) in enumerate(ax1)
    # ax.title = "U$i"
    # CairoMakie.lines!(ax, x, posterior.U_hat[:,i], labels = false, color = colorlist[i], linewidth = LW)
    # CairoMakie.band!(ax, x, posterior.U_lower[:,i], posterior.U_upper[:,i], color = (colorlist[i], transparencyValue))
    # CairoMakie.lines!(ax, x, U[:,i] .* trsfrm[i], labels = false, color = :black, linewidth = LW)
    # CairoMakie.lines!(ax, x, svd(Z).U[:,i], color = colorlist[i], linestyle = :dash, linewidth = LW)
    # CairoMakie.text!(ax, 4, 0.22, text = Ulabels[i], fontsize = 30)
    CairoMakie.lines!(ax, x, posterior.U_hat[:,i], labels = false, color = :blue, linewidth = LW)
    CairoMakie.band!(ax, x, posterior.U_lower[:,i], posterior.U_upper[:,i], color = (:blue, transparencyValue))
    CairoMakie.lines!(ax, x, U[:,i] .* trsfrm[i], labels = false, color = :black, linewidth = LW)
    CairoMakie.lines!(ax, x, svd(Z).U[:,i], color = :red, linestyle = :dash, linewidth = LW)
    CairoMakie.text!(ax, 4, 0.22, text = Ulabels[i], fontsize = 30)
end

for (i, ax) in enumerate(ax2)
    # ax.title = "V$i"
    CairoMakie.lines!(ax, t, posterior.V_hat[:,i], labels = false, color = :blue, linewidth = LW)
    CairoMakie.band!(ax, t, posterior.V_lower[:,i], posterior.V_upper[:,i], color = (:blue, transparencyValue))
    CairoMakie.lines!(ax, t, V[:,i] .* trsfrm[i], labels = false, color = :black, linewidth = LW)
    CairoMakie.lines!(ax, t, svd(Z).V[:,i], color = :red, linestyle = :dash, linewidth = LW)
    CairoMakie.text!(ax, 9, 0.22, text = Vlabels[i], fontsize = 30)
end

hidexdecorations!.(ax1[1:4])
hidexdecorations!.(ax2[1:4])
g





#### Plot recovered surface


svdY = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-1.05, 1.05).*maximum(abs, Z)
nsteps = 20
nticks = 7

elims = (-1.05, 1.05).*maximum(abs, vcat(reshape(svdY .- Y, :), reshape(Y_hat .- Y, :)))
ensteps = 20
enticks = 7


# set up plot
g = CairoMakie.Figure(resolution = (1000, 700))
ax11 = Axis(g[1,1], ylabel = "Space", limits = ((0,10), (-5, 5)), title = "Bayes Estimate")
ax12 = Axis(g[1,2], limits = ((0,10), (-5, 5)), title = "Truth")
ax21 = Axis(g[2,1], ylabel = "Space", xlabel = "Time", limits = ((0,10), (-5, 5)), title = "Bayes - Truth")
ax13 = Axis(g[1,3], limits = ((0,10), (-5, 5)), title = "Algorithm")
ax23 = Axis(g[2,3], xlabel = "Time", limits = ((0,10), (-5, 5)), title = "Algorithm - Truth")
# linkyaxes!(ax11, ax12, ax13)
# linkxaxes!(ax22, ax23)
g

# plot results
p11 = CairoMakie.heatmap!(ax12, t, x, Y',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
p12 = CairoMakie.heatmap!(ax11, t, x, Y_hat',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
p13 = CairoMakie.heatmap!(ax13, t, x, svdY',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
CairoMakie.Colorbar(g[1,4], p11, ticks = round.(range(lims[1], lims[2], length = nticks), digits = 2))

# p21 = CairoMakie.heatmap!(ax21, t, x, Z',
#     colormap = :balance, colorrange = lims,
#     levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))

p21 = CairoMakie.heatmap!(ax21, t, x, (Y_hat .- Y)',
    colormap = :balance, colorrange = elims,
    levels = range(elims[1], elims[2], step = (elims[2]-elims[1])/ensteps))
#
p23 = CairoMakie.heatmap!(ax23, t, x, (svdY .- Y)',
    colormap = :balance, colorrange = elims,
    levels = range(elims[1], elims[2], step = (elims[2]-elims[1])/ensteps))
#
CairoMakie.Colorbar(g[2,4], p23, ticks = round.(range(elims[1], elims[2], length = enticks), digits = 2))


CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax13, grid = false)
CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax13, grid = false)
# CairoMakie.hideydecorations!(ax22, grid = false)
CairoMakie.hideydecorations!(ax23, grid = false)
g

# save("./results/oneSpatialDimension/generatedData1D.png", g)


#endregion