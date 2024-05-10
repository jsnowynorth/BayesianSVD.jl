
########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 10-May-2024
#### Description: Code used to create Fig 4, Fig S.1, and Fig S.4 
########################################################################


using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

using CairoMakie
using DataFrames, DataFramesMeta, Chain, CSV
using LaTeXStrings


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
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 10, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Z), c = :balance)
Plots.contourf(x, t, Z', clim = extrema(Z), c = :balance)

# var(Y)/(var(Z - Y))

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
ax12 = Axis(g[1,2], yticks = [-0.3, -0.15, 0, 0.15, 0.3], limits = ((0, 10),(-0.31, 0.31)), xlabel = "Time", xlabelsize = 20, xlabelfont = :bold, xaxisposition = :top, yaxisposition = :right)
ax22 = Axis(g[2,2], limits = ((0,10), (0, 10)), yaxisposition = :right)
ax21 = Axis(g[2,1], xticks = [-0.35, -0.17, 0, 0.17, 0.35], xticklabelrotation = -pi/5, limits = ((-0.36, 0.36), (-5, 5)), ylabel = "Space", ylabelsize = 20, ylabelfont = :bold)
linkyaxes!(ax22, ax21)
linkxaxes!(ax22, ax12)
g


# plot the D Matrix
Dlabels = [L"\textbf{d}_{%$i, %$i}" for i in 1:5]

for i in 1:5
    CairoMakie.text!(ax11, i-1, 5-i, text = Dlabels[i], fontsize = 28, color = colorlist[i])
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
CairoMakie.text!(ax12, 0.1, -0.3, text = L"\textbf{V}", fontsize = 30)
CairoMakie.text!(ax21, 0.23, 4.1, text = L"\textbf{U}", fontsize = 30)
CairoMakie.text!(ax22, 0.1, 4.1, text = L"\textbf{Z}", fontsize = 30)
g

# save("figures/generatedData1D.svg", g)
# save("figures/generatedData1D.png", g)





#endregion

######################################################################
#### Sample
######################################################################
#region


k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)


posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)



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
# save("figures/Ubasis.png", g)

g = Plots.plot(posterior, t, size = (800, 400), basis = 'V',  linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "V")
g = Plots.plot!(t, (V' .* trsfrm)', label = false, color = "black", linewidth = 2)
# save("figures/Vbasis.png", g)




Plots.plot(posterior.σU', title = "σU", size = (1000, 600))
Plots.plot(posterior.σV', title = "σV", size = (1000, 600))
Plots.plot(posterior.σ, title = "σ")
Plots.plot(posterior.U[50,:,:]', title = "U")
Plots.hline!(U[50,:] .* trsfrm, c = :black, label = false)
Plots.plot(posterior.V[45,:,:]', title = "V")
Plots.hline!(V[45,:] .* trsfrm, c = :black, label = false)


posteriorCoverage(reshape(U[:,1]' .* trsfrm[1], :), posterior.U[:,1,:], 0.95)
posteriorCoverage(reshape(U[:,2]' .* trsfrm[2], :), posterior.U[:,2,:], 0.95)
posteriorCoverage(reshape(U[:,3]' .* trsfrm[3], :), posterior.U[:,3,:], 0.95)
posteriorCoverage(reshape(U[:,4]' .* trsfrm[4], :), posterior.U[:,4,:], 0.95)
posteriorCoverage(reshape(U[:,5]' .* trsfrm[5], :), posterior.U[:,5,:], 0.95)

posteriorCoverage(Matrix((U[:,1:k]' .* trsfrm)'), posterior.U, 0.95)
posteriorCoverage(Matrix((V[:,1:k]' .* trsfrm)'), posterior.V, 0.95)


#endregion


######################################################################
#### Plot Results
######################################################################
#region

#### Plot U basis functions

ubounds = maximum(abs.(extrema(reduce(hcat, (reshape(posterior.U_hat, :), reshape(posterior.U_lower, :), reshape(posterior.U_upper, :))))))
# ubounds = (-1.05, 1.05) .* ubounds
ubounds = (-1.0, 1.0) .* ubounds

vbounds = maximum(abs.(extrema(reduce(hcat, (reshape(posterior.V_hat, :), reshape(posterior.V_lower, :), reshape(posterior.V_upper, :))))))
# vbounds = (-1.05, 1.05) .* vbounds
vbounds = (-1.0, 1.0) .* vbounds

colorlist = [:blue, :red, :magenta, :orange, :green]
LW = 2
transparencyValue = 0.3


#### 5 x 2 basis function plot
g = CairoMakie.Figure(resolution = (1400, 600))
ax1 = [CairoMakie.Axis(g[1, i], limits = ((-5,5), ubounds)) for i in 1:5]
ax2 = [CairoMakie.Axis(g[2, i], limits = ((0,10), vbounds)) for i in 1:5]

# ax1[5].xlabel = "Space"
# ax2[5].xlabel = "Time"

Ulabels = [L"\textbf{u}_{%$i}" for i in 1:5]
Vlabels = [L"\textbf{v}_{%$i}" for i in 1:5]

for (i, ax) in enumerate(ax1)
    CairoMakie.band!(ax, x, posterior.U_lower[:,i], posterior.U_upper[:,i], color = (:blue, transparencyValue))
    CairoMakie.lines!(ax, x, svd(Z).U[:,i], color = :red, linewidth = LW)
    CairoMakie.lines!(ax, x, posterior.U_hat[:,i], labels = false, color = :blue, linewidth = LW)
    CairoMakie.lines!(ax, x, U[:,i] .* trsfrm[i], labels = false, color = :black, linewidth = LW)
    CairoMakie.text!(ax, 3, 0.22, text = Ulabels[i], fontsize = 30)
end

for (i, ax) in enumerate(ax2)
    CairoMakie.band!(ax, t, posterior.V_lower[:,i], posterior.V_upper[:,i], color = (:blue, transparencyValue))
    CairoMakie.lines!(ax, t, svd(Z).V[:,i], color = :red, linewidth = LW)
    CairoMakie.lines!(ax, t, posterior.V_hat[:,i], labels = false, color = :blue, linewidth = LW)
    CairoMakie.lines!(ax, t, V[:,i] .* trsfrm[i], labels = false, color = :black, linewidth = LW)
    CairoMakie.text!(ax, 8, 0.22, text = Vlabels[i], fontsize = 30)
end

g

# save("figures/estimatedBasis1D.png", g)





#### Plot recovered surface


svdY = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-1.05, 1.05).*maximum(abs, svdY)
nsteps = 40
nticks = 7

elims = (-1.05, 1.05).*maximum(abs, vcat(reshape(svdY .- Y, :), reshape(Y_hat .- Y, :)))
ensteps = 40
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
p11 = CairoMakie.contourf!(ax12, t, x, Y',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
p12 = CairoMakie.contourf!(ax11, t, x, Y_hat',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
p13 = CairoMakie.contourf!(ax13, t, x, svdY',
    colormap = :balance, colorrange = lims,
    levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))
#
CairoMakie.Colorbar(g[1,4], p11, ticks = round.(range(lims[1], lims[2], length = nticks), digits = 2))

# p21 = CairoMakie.contourf!(ax21, t, x, Z',
#     colormap = :balance, colorrange = lims,
#     levels = range(lims[1], lims[2], step = (lims[2]-lims[1])/nsteps))

p21 = CairoMakie.contourf!(ax21, t, x, (Y_hat .- Y)',
    colormap = :balance, colorrange = elims,
    levels = range(elims[1], elims[2], step = (elims[2]-elims[1])/ensteps))
#
p23 = CairoMakie.contourf!(ax23, t, x, (svdY .- Y)',
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

# save("figures/estimatedSurface1D.png", g)



#endregion



########################################################################
#### Recreate data visual but with UQ estimates
########################################################################
#region

colorlist = [:blue, :red, :magenta, :orange, :green]
transparencyValue = 0.3
colorlistT = [(:blue, transparencyValue), (:red, transparencyValue), (:magenta, transparencyValue), (:orange, transparencyValue), (:green, transparencyValue)]
nsteps = 20
nticks = 7
LW = 3


##################
#### basis function plot
##################
# set up plot
g = CairoMakie.Figure(;resolution = (1000, 700), linewidth = 5)
ax11 = Axis(g[1,1], limits = ((0, 5), (0, 5)), xgridvisible = false, ygridvisible = false)
ax12 = Axis(g[1,2], yticks = [-0.3, -0.15, 0, 0.15, 0.3], limits = ((0, 10),(-0.36, 0.42)), xgridvisible = false, ygridvisible = false)
ax22 = Axis(g[2,2], limits = ((0,10), (0, 10)), yaxisposition = :right, xgridvisible = false, ygridvisible = false)
ax21 = Axis(g[2,1], xticks = [-0.35, -0.17, 0, 0.17, 0.35], xticklabelrotation = -pi/5, limits = ((-0.28, 0.2), (-5, 5)), xgridvisible = false, ygridvisible = false)
linkyaxes!(ax22, ax21)
linkxaxes!(ax22, ax12)
g


# plot the D Matrix
Dlabels = [L"(\textbf{d}_{%$i, %$i}^L, \textbf{d}_{%$i, %$i}^U)" for i in 1:5]

for i in 1:5
    CairoMakie.text!(ax11, 0.75*(i-1), 5-i, text = Dlabels[i], fontsize = 28, color = colorlist[i])
end
g



# plot time basis functions
CairoMakie.series!(ax12, t, posterior.V_hat', labels = Dlabels, color = colorlist, linewidth = LW)
CairoMakie.xlims!(ax12, low = 0, high = 10)
for i in 1:5
    CairoMakie.band!(ax12, t, posterior.V_lower[:,i], posterior.V_upper[:,i], color = colorlistT[i])
end
g


# plot spatial basis functions
for i in axes(U, 2)
    CairoMakie.lines!(ax21, posterior.U_hat[:,i], x, label = Dlabels[i], color = colorlist[i], linewidth = LW)
    CairoMakie.band!(ax21, Point2f.(posterior.U_lower[:,i], x), Point2f.(posterior.U_upper[:,i], x), color = colorlistT[i])
end
CairoMakie.ylims!(ax21, low = -5, high = 5)
g



# contour plot
crange = (-1.05, 1.05).*maximum(abs, Z)
hm = CairoMakie.contourf!(ax22, t, x, Z', 
    colormap = :balance, colorrange = crange,
    levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

#
g


CairoMakie.hideydecorations!(ax11, grid = false)
CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax21, grid = false)
CairoMakie.hideydecorations!(ax21, grid = false)
CairoMakie.hidexdecorations!(ax22, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
g


# size the columns and rows
colsize!(g.layout, 1, Fixed(275))
rowsize!(g.layout, 1, Fixed(225))
g

# spacing the columns and rows
colgap!(g.layout, 1, 15)
rowgap!(g.layout, 1, 15)
g


# save("figures/logo.png", g)


#endregion