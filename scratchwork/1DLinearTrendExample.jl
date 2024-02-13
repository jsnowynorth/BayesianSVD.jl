


using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra
using CairoMakie

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
ϵ = 0.1

Random.seed!(3)
# U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 1, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Z))
Plots.contourf(x, t, Z', clim = extrema(Z))

# var(Y)/(var(Z - Y))

Random.seed!(3)
β = [-2, 0.6, 1.2, -0.9]
X = rand(Normal(0, 0.2), n*m, length(β))
M = reshape(X*β, n, m)
Z = Z + M
Plots.contourf(x, t, Z')

Plots.contourf(x, t, M')

# η = rand(Normal(), n, m)
# σ = sqrt.(var(M + Y) ./ (1 * var(η))) # set the standard deviation
# Z = M + Y + σ .* η

# var(M + Y)/(var(Z - M - Y))

#endregion



######################################################################
#### Visualize Data
######################################################################
#region

colorlist = [:blue, :red, :magenta, :orange, :green]
LW = 4
nsteps = 20
nticks = 7


##################
#### basis function plot
##################
# set up plot
g = CairoMakie.Figure(;resolution = (1000, 700), linewidth = 5)
ax11 = Axis(g[1,1], limits = ((0, 5), (0, 5)), xgridvisible = false, ygridvisible = false)
ax12 = Axis(g[1,2], yticks = [-0.3, -0.15, 0, 0.15, 0.3], limits = ((0, 10),(-0.36, 0.36)), xlabel = "Time", xlabelsize = 20, xlabelfont = :bold, xaxisposition = :top, yaxisposition = :right)
ax22 = Axis(g[2,2], limits = ((0,10), (0, 10)), yaxisposition = :right)
ax21 = Axis(g[2,1], xticks = [-0.35, -0.17, 0, 0.17, 0.35], xticklabelrotation = -pi/5, limits = ((-0.22, 0.22), (-5, 5)), ylabel = "Space", ylabelsize = 20, ylabelfont = :bold)
linkyaxes!(ax22, ax21)
linkxaxes!(ax22, ax12)
g


# plot the D Matrix
Dlabels = [L"\textbf{d}_{%$i, %$i} = %$(D[i])" for i in 1:5]

for i in 1:5
    CairoMakie.text!(ax11, 0.65*(i-1), 5-i, text = Dlabels[i], fontsize = 28, color = colorlist[i])
end
g



# plot time basis functions
CairoMakie.series!(ax12, t, V', labels = Dlabels, color = colorlist, linewidth = LW)
CairoMakie.xlims!(ax12, low = 0, high = 10)
g


# plot spatial basis functions
for i in axes(U, 2)
    CairoMakie.lines!(ax21, U[:,i], x, label = Dlabels[i], color = colorlist[i], linewidth = LW)
end
CairoMakie.ylims!(ax21, low = -5, high = 5)
g


# contour plot
crange = (-1.05, 1.05).*maximum(abs, Z)
hm = CairoMakie.contourf!(ax22, t, x, Z', 
    colormap = :balance, colorrange = crange,
    levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

#
CairoMakie.Colorbar(g[1:2,3], hm, ticks = round.(range(crange[1], crange[2], length = nticks), digits = 3))
g


CairoMakie.hideydecorations!(ax11, grid = false)
CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax21, grid = false)
CairoMakie.hidexdecorations!(ax22, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
g


# size the columns and rows
colsize!(g.layout, 1, Fixed(200))
rowsize!(g.layout, 1, Fixed(175))
g

# spacing the columns and rows
colgap!(g.layout, 1, 15)
rowgap!(g.layout, 1, 15)
colgap!(g.layout, 2, 15)
g

# label the subplots
CairoMakie.text!(ax12, 0.1, -0.35, text = L"\textbf{V}", fontsize = 30)
CairoMakie.text!(ax21, 0.15, 4.1, text = L"\textbf{U}", fontsize = 30)
CairoMakie.text!(ax22, 0.1, 4.1, text = L"\textbf{Z}", fontsize = 30)
g


# save("./results/oneSpatialDimension/generatedData1D.svg", g)
# save("./results/oneSpatialDimension/generatedData1D.png", g)




#endregion


######################################################################
#### Sample
######################################################################
#region

# X = reshape(ones(n*m), :, 1)

k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, X, x, t, k)
pars = Pars(data, ΩU, ΩV)


posterior, pars = SampleSVD(data, pars; nits = 10000, burnin = 5000)


posterior.β_hat
posterior.β_lower
posterior.β_upper

[std(posterior.β[i,:]) for i in 1:4]

posterior.σ_hat .* inv(data.X' * data.X)

Plots.plot(posterior.β', label = false, size = (900, 600))
Plots.hline!([β], label = false)

posterior.D_hat
posterior.D_lower
posterior.D_upper
posterior.σ_hat
posterior.σU_hat
posterior.σV_hat

[hpd(posterior.ρU[i,:]) for i in axes(posterior.ρU, 1)]
[hpd(posterior.ρV[i,:]) for i in axes(posterior.ρV, 1)]

Plots.plot(posterior.ρU')
Plots.plot(posterior.ρV')

var(U, dims = 1)
var(V, dims = 1)

# Plots.plot((posterior.σU ./ (posterior.D.^2))')
# Plots.plot((posterior.σV ./ (posterior.D.^2))')

mean((posterior.σU ./ (posterior.D.^2)), dims = 2)
mean((posterior.σV ./ (posterior.D.^2)), dims = 2)

[hpd((posterior.σU ./ (posterior.D.^2))[:,i]) for i in axes(posterior.D, 1)]
[hpd((posterior.σV ./ (posterior.D.^2))[:,i]) for i in axes(posterior.D, 1)]

D.^2 ./ (n-1)
[hpd(posterior.σU[i,:]) for i in axes(posterior.σU, 1)]

D.^2 ./ (m-1)
[hpd(posterior.σV[i,:]) for i in axes(posterior.σU, 1)]

posterior.σU_hat ./ ((posterior.D_hat.^2))
posterior.σV_hat ./ ((posterior.D_hat.^2))



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




svdY = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
# Y_hat = mean(Yest)

Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-1.5, 1.5)


l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Y_hat, title = "Y Hat", c = :balance, clim = (-3, 3))
p2 = Plots.contourf(t, x, Y, title = "Truth", c = :balance, clim = (-3, 3))
p3 = Plots.contourf(t, x, svdY, title = "Algorithm", c = :balance, clim = (-3, 3))
p4 = Plots.contourf(t, x, Y_hat .- Y, title = "Y Hat - Truth", c = :balance, clim = (-0.4, 0.4))
p5 = Plots.contourf(t, x, Y, title = "Observed", c = :balance, clim = (-3, 3))
p6 = Plots.contourf(t, x, svdY .- Y, title = "Algorithm - Truth", c = :balance, clim = (-0.4, 0.4))
Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))



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


########################################################################
#### Y coverage
########################################################################
#region

Ypost = [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U,3)]

posteriorCoverage(Y, Ypost, 0.95)

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