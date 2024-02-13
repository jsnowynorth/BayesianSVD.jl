


include("../src/BasisFunctions.jl")
using CairoMakie
using DataFrames, DataFramesMeta, Chain, CSV
using LaTeXStrings


######################################################################
#### Generate Some Data
######################################################################
#region

m = 50
nx = 50
ny = 50
x = range(-5, 5, nx)
y = range(-5, 5, ny)
t = range(0, 10, m)

ΣU = MaternCorrelation(x, y, ρ = 2, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 1.5, ν = 3.5, metric = Euclidean())


# sqrt of eigenvalues
D = [40, 20, 10]

k = 3
ϵ = 0.05

Random.seed!(2)
# U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 1, SNR = true)

# var(Y)/(var(Z - Y))


Plots.contourf(x, y, reshape(U[:,1], nx, ny))
Plots.contourf(x, y, reshape(U[:,2], nx, ny))
Plots.contourf(x, y, reshape(U[:,3], nx, ny))
Plots.plot(t, V)


Plots.contourf(x, t, reshape(Y[:,10], nx, ny), clim = extrema(Z))
Plots.contourf(x, y, reshape(Z[:,10], nx, ny), clim = extrema(Z))

#endregion

######################################################################
#### Gif of Data
######################################################################
#region

limsY = (-1.05, 1.05).*maximum(abs, Y)
limsZ = (-1.01, 1.01).*maximum(abs, Z)
nsteps = 40
nticks = 7
# Zticks = (round.(range(limsZ[1], limsZ[2], length = nticks), digits = 2), string.(round.(range(limsZ[1], limsZ[2], length = nticks), digits = 2)))


l = Plots.@layout [a b]
p1 = Plots.contourf(x, y, reshape(Y[:,2], nx, ny), c = :balance, clim = limsY, nlevels = nsteps, lw = 0.2, xlabel = "X", ylabel = "Y", title = "Truth")
p2 = Plots.contourf(x, y, reshape(Z[:,2], nx, ny), c = :balance, clim = limsZ, nlevels = nsteps, lw = 0.2, xlabel = "X", title = "Noisy", yaxis = false)
Plots.plot(p1, p2, layout = l, size = (1000, 500))


anim = @animate for i in 1:50 
    l = Plots.@layout [A{0.01h}; [a b]]
    title = Plots.plot(title = "Timestep $i", grid = false, showaxis = false, bottom_margin = -30Plots.px)
    p1 = Plots.contourf(x, y, reshape(Y[:,i], nx, ny), c = :balance, clim = limsY, nlevels = nsteps, lw = 0.2, title = "Truth")
    p2 = Plots.contourf(x, y, reshape(Z[:,i], nx, ny), c = :balance, clim = limsZ, nlevels = nsteps, lw = 0.2, title = "Noisy", yaxis = false)
    Plots.plot(title, p1, p2, layout = l, size = (1200, 500))
end
# gif(anim, "/Users/JSNorth/Documents/Presentations/LBLMLGroup/SpatialData.gif", fps = 4)

#endregion

######################################################################
#### Visualize Data
######################################################################
#region


#### basis function plot
# set up plot
g = CairoMakie.Figure(resolution = (1300, 700), linewidth = 5)
ax11 = Axis(g[2,1], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax12 = Axis(g[2,2], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax13 = Axis(g[2,3], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax22 = Axis(g[3,1:3], xlabel = "time", yticks = -0.3:0.1:0.3, xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
linkyaxes!(ax11, ax12, ax13)
linkxaxes!(ax11, ax12)

# plot time basis functions
# CairoMakie.lines!(ax22, t, V[:,1], color = :blue, label = "Basis 1")
# CairoMakie.lines!(ax22, t, V[:,2], color = :red, label = "Basis 2")
# CairoMakie.lines!(ax22, t, V[:,3], color = :green, label = "Basis 3")
# CairoMakie.xlims!(ax22, low = 0, high = 10)
CairoMakie.series!(ax22, t, V', labels = ["Basis Function $i" for i in axes(V,2)], color = [:blue, :red, :green], linewidth = 4)
CairoMakie.xlims!(ax22, low = 0, high = 10)
CairoMakie.axislegend(ax22, position = :rt)

# plot spatial basis functions
b1 = CairoMakie.contourf!(ax11, x, y, reshape(U[:,1], nx, ny), 
    colormap = :balance, colorrange = (-0.07, 0.07),
    levels = range(-0.07, 0.07, step = 0.005))
#
CairoMakie.Colorbar(g[1,1], b1, ticks = [-0.07, -0.03, 0, 0.03, 0.07], vertical = false)

b2 = CairoMakie.contourf!(ax12, x, y, reshape(U[:,2], nx, ny), 
    colormap = :balance, colorrange = (-0.07, 0.07),
    levels = range(-0.07, 0.07, step = 0.005))
#
CairoMakie.Colorbar(g[1,2], b2, ticks = [-0.07, -0.03, 0, 0.03, 0.07], vertical = false)

b3 = CairoMakie.contourf!(ax13, x, y, reshape(U[:,3], nx, ny), 
    colormap = :balance, colorrange = (-0.07, 0.07),
    levels = range(-0.07, 0.07, step = 0.005))
#
CairoMakie.Colorbar(g[1,3], b3, ticks = [-0.07, -0.03, 0, 0.03, 0.07], vertical = false)
g

rowsize!(g.layout, 1, Fixed(10))
rowsize!(g.layout, 2, Fixed(350))
rowsize!(g.layout, 3, Fixed(150))
g

# save("/Users/JSNorth/Documents/Presentations/LBLMLGroup/basisFunctions2D.png", g)







#### data plot
# set up plot
g = CairoMakie.Figure(resolution = (1000, 700), linewidth = 5)
ax11 = Axis(g[1,1], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax12 = Axis(g[1,2], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax21 = Axis(g[2,1], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax22 = Axis(g[2,2], ylabel = "Y", xlabel = "X", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
linkyaxes!(ax11, ax12)
linkyaxes!(ax21, ax22)
linkxaxes!(ax11, ax21)
linkxaxes!(ax12, ax22)
g

# plot spatial basis functions
p1 = CairoMakie.contourf!(ax11, x, y, reshape(Z[:,1], nx, ny), 
    colormap = :balance, colorrange = (-1.4, 1.4),
    levels = range(-1.4, 1.4, step = 0.01))
#
p2 = CairoMakie.contourf!(ax12, x, y, reshape(Z[:,10], nx, ny), 
    colormap = :balance, colorrange = (-1.4, 1.4),
    levels = range(-1.4, 1.4, step = 0.01))
#
p3 = CairoMakie.contourf!(ax21, x, y, reshape(Z[:,20], nx, ny), 
    colormap = :balance, colorrange = (-1.4, 1.4),
    levels = range(-1.4, 1.4, step = 0.01))
#
p4 = CairoMakie.contourf!(ax22, x, y, reshape(Z[:,30], nx, ny), 
    colormap = :balance, colorrange = (-1.4, 1.4),
    levels = range(-1.4, 1.4, step = 0.01))
#
CairoMakie.Colorbar(g[1:2,3], p1, ticks = -1.4:0.2:1.4, 
    labelsize = 50, ticksize = 10, ticklabelsize = 20)
#

CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
g

# Label(g[1, 1, TopLeft()], "t = 1", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# Label(g[2, 1, TopLeft()], "t = 10", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# Label(g[1, 2, TopLeft()], "t = 20", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# Label(g[2, 2, TopLeft()], "t = 30", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# g

# save("./results/twoSpatialDimensions/generatedData2D.png", g)

#endregion


######################################################################
#### Sample
######################################################################
#region

# need to fix to allow for x, y, t locations in data call   
# maybe locs input and then the structure is locs = [x, t, ...] splatt or whatever
ulocs = reduce(hcat,reshape([[x, y] for x = x, y = y], nx * ny))'

k = 3
ΩU = MaternCorrelation(x, y, ρ = 2, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 1.5, ν = 3.5, metric = Euclidean())
data = Data(Z, ulocs[:,1], t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)

# Plots.plot(posterior, x, y, size = (1000, 600))

# posterior.rU_hat
# posterior.rV_hat
posterior.σU_hat
posterior.σV_hat
posterior.σ_hat
posterior.D_hat
posterior.D_lower
posterior.D_upper



Uhat = reshape(posterior.U_hat, nx, ny, k)
Ulower = reshape(posterior.U_lower, nx, ny, k)
Uupper = reshape(posterior.U_upper, nx, ny, k)
UhatSVD = reshape(svd(Z).U[:,1:k], nx, ny, k)

l = Plots.@layout[a b c; d e f; g h i]
p1 = Plots.contourf(Uhat[:,:,1], title = "Recover 1")
p2 = Plots.contourf(-Uhat[:,:,2], title = "Recover 2")
p3 = Plots.contourf(-Uhat[:,:,3], title = "Recover 3")
p4 = Plots.contourf(x, y, reshape(U[:,1], nx, ny), title = "True 1")
p5 = Plots.contourf(x, y, reshape(U[:,2], nx, ny), title = "True 2")
p6 = Plots.contourf(x, y, reshape(U[:,3], nx, ny), title = "True 3")
p7 = Plots.contourf(UhatSVD[:,:,1], title = "SVD 1")
p8 = Plots.contourf(-UhatSVD[:,:,2], title = "SVD 2")
p9 = Plots.contourf(-UhatSVD[:,:,3], title = "SVD 3")
Plots.plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout = l, size = (1000, 600))
#


# t = range(1, 100, step = 1)
Plots.plot(posterior, t, basis = 'V', size = (800, 400), linewidth = 2, c = [:blue :red :green], tickfontsize = 14, label = false)
Plots.plot!(t, (V' .* [1, -1, -1])', color = "black", label = false)
Plots.plot!(t, svd(Z).V[:,1:data.k], linewidth = 1.5, c = [:blue :red :green], linestyle = :dash, label = false)
#


Plots.plot(posterior.σ, title = "σ")
Plots.plot(posterior.D', title = "D")
Plots.plot(posterior.U[1270,:,:]', title = "U")
Plots.plot(posterior.V[10,:,:]', title = "V")

histogram(posterior.D')


Yhat = reshape(mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U, 3)]), nx, ny, m)
svdZ = svd(Z)
Yalgo = reshape(svdZ.U[:,1:data.k] * diagm(svdZ.S[1:data.k]) * svdZ.V[:,1:data.k]', nx, ny, m)

l = Plots.@layout[a b c; d e f]
p1 = Plots.contourf(x, y, Yhat[:,:,10], title = "Post 10")
p2 = Plots.contourf(x, y, reshape(Y[:,10], nx, ny), title = "True 10")
p3 = Plots.contourf(x, y, Yalgo[:,:,10], title = "Algo 10")
p4 = Plots.contourf(x, y, Yhat[:,:,40], title = "Post 40")
p5 = Plots.contourf(x, y, reshape(Y[:,40], nx, ny), title = "True 40")
p6 = Plots.contourf(x, y, Yalgo[:,:,40], title = "Algo 40")
Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1200, 600))
#

#endregion


######################################################################
#### Plot results
######################################################################
#region

Uhat = -reshape(posterior.U_hat, nx, ny, k)
UhatSVD = -reshape(svd(Z).U[:,1:k], nx, ny, k)
Utrue = reshape(U, nx, ny, k)

lowerBound = -0.07
upperBound = 0.07
stepBound = (upperBound*2)/28
stepSize = (upperBound*2)/8
transparencyValue = 0.5


#### data plot
# set up plot
g = CairoMakie.Figure(resolution = (1000, 900), linewidth = 2)
ax11 = Axis(g[1,1], ylabel = "Estimate", limits = ((-5, 5), (-5, 5)), ylabelsize = 20, ylabelfont = :bold, title = "U₁")
ax12 = Axis(g[1,2], limits = ((-5, 5), (-5, 5)), title = "U₂")
ax13 = Axis(g[1,3], limits = ((-5, 5), (-5, 5)), title = "U₃")
ax21 = Axis(g[2,1], ylabel = "Truth", limits = ((-5, 5), (-5, 5)), ylabelsize = 20, ylabelfont = :bold)
ax22 = Axis(g[2,2], limits = ((-5, 5), (-5, 5)))
ax23 = Axis(g[2,3], limits = ((-5, 5), (-5, 5)))
ax31 = Axis(g[3,1], ylabel = "Algorithm", limits = ((-5, 5), (-5, 5)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax32 = Axis(g[3,2], limits = ((-5, 5), (-5, 5)), xlabelsize = 20, xlabelfont = :bold)
ax33 = Axis(g[3,3], limits = ((-5, 5), (-5, 5)), xlabelsize = 20, xlabelfont = :bold)
ax4 = Axis(g[4,1:3], xlabel = "time", yticks = -0.4:0.1:0.4, limits = ((0, 10), (-0.42, 0.42)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
linkyaxes!(ax11, ax12, ax13)
linkyaxes!(ax21, ax22, ax23)
linkyaxes!(ax31, ax32, ax33)
linkxaxes!(ax11, ax12, ax13)
linkxaxes!(ax21, ax22, ax23)
linkxaxes!(ax31, ax32, ax33)
g


# plot temporal basis functions
V_hat = -posterior.V_hat
V_lower = -posterior.V_lower
V_upper = -posterior.V_upper
CairoMakie.lines!(ax4, t, V_hat[:,1], color = :blue, label = "V1")
CairoMakie.band!(ax4, t, V_lower[:,1], V_upper[:,1], color = (:blue, transparencyValue))
CairoMakie.lines!(ax4, t, V_hat[:,2], color = :red, label = "V2")
CairoMakie.band!(ax4, t, V_lower[:,2], V_upper[:,2], color = (:red, transparencyValue))
CairoMakie.lines!(ax4, t, V_hat[:,3], color = :green, label = "V3")
CairoMakie.band!(ax4, t, V_lower[:,3], V_upper[:,3], color = (:green, transparencyValue))
CairoMakie.lines!(ax4, t, V[:,1], color = :black)
CairoMakie.lines!(ax4, t, V[:,2], color = :black)
CairoMakie.lines!(ax4, t, V[:,3], color = :black)
# CairoMakie.lines!(ax4, t, svd(Y).V[:,1], color = :blue, label = "Algorithm 1")
# CairoMakie.lines!(ax4, t, svd(Y).V[:,2], color = :red, label = "Algorithm 2")
# CairoMakie.lines!(ax4, t, svd(Y).V[:,3], color = :green, label = "Algorithm 3")
CairoMakie.xlims!(ax4, low = 0, high = 10)
# CairoMakie.axislegend(ax4, position = :rt)
leg = CairoMakie.Legend(g[4,4], ax4,
    markersize = 10, labelsize = 15)
# leg.tellheight = true
g



# plot spatial basis functions
p11 = CairoMakie.contourf!(ax11, x, y, -Uhat[:,:,1], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p12 = CairoMakie.contourf!(ax12, x, y, Uhat[:,:,2], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p13 = CairoMakie.contourf!(ax13, x, y, Uhat[:,:,3], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p21 = CairoMakie.contourf!(ax21, x, y, Utrue[:,:,1], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p22 = CairoMakie.contourf!(ax22, x, y, Utrue[:,:,2], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p23 = CairoMakie.contourf!(ax23, x, y, Utrue[:,:,3], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p31 = CairoMakie.contourf!(ax31, x, y, -UhatSVD[:,:,1], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p32 = CairoMakie.contourf!(ax32, x, y, UhatSVD[:,:,2], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
p33 = CairoMakie.contourf!(ax33, x, y, UhatSVD[:,:,3], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
CairoMakie.Colorbar(g[1:3,4], p11, ticks = lowerBound:stepSize:upperBound, 
    labelsize = 50, ticksize = 10, ticklabelsize = 20)
#

CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax13, grid = false)
CairoMakie.hidexdecorations!(ax21, grid = false)
CairoMakie.hidexdecorations!(ax22, grid = false)
CairoMakie.hidexdecorations!(ax23, grid = false)

CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax13, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
CairoMakie.hideydecorations!(ax23, grid = false)
CairoMakie.hideydecorations!(ax32, grid = false)
CairoMakie.hideydecorations!(ax33, grid = false)
g

colsize!(g.layout, 4, Fixed(30))
rowsize!(g.layout, 4, Fixed(100))
g

# save("./results/twoSpatialDimensions/estimatedBasisFunctions2D.png", g)
# save("/Users/JSNorth/Documents/Presentations/LBLMLGroup/estimatedBasisFunctions2D.png", g)


######################################################################
#### Uncertainty plot

# CI = reshape((posterior.U_lower .< -Φ) .& (posterior.U_upper .> -Φ), nx, ny, k)
CI = (posterior.U_lower .> -U) .| (posterior.U_upper .< -U)
CI[:,1] = .!CI[:,1]
SD = reshape(std(posterior.U, dims = 3)[:,:,1], nx, ny, k)


locs = reduce(hcat,reshape([[x, y] for x = x, y = y], nx * ny))'


lowerBound = -0.07
upperBound = 0.07
stepBound = 0.005
stepSize = 0.02
transparencyValue = 0.7

lowerBoundSD = 0
upperBoundSD = 0.006
stepBoundSD = 0.0002
stepSizeSD = 0.001



#### data plot
# set up plot
g = CairoMakie.Figure(resolution = (1000, 600), linewidth = 2)
ax11 = Axis(g[1,1], ylabel = "Mean", limits = ((-5, 5), (-5, 5)), ylabelsize = 20, ylabelfont = :bold, title = "U₁")
ax12 = Axis(g[1,2], limits = ((-5, 5), (-5, 5)), title = "U₂")
ax13 = Axis(g[1,3], limits = ((-5, 5), (-5, 5)), title = "U₃")
ax21 = Axis(g[2,1], ylabel = "Standard Deviation", limits = ((-5, 5), (-5, 5)), ylabelsize = 20, ylabelfont = :bold)
ax22 = Axis(g[2,2], limits = ((-5, 5), (-5, 5)))
ax23 = Axis(g[2,3], limits = ((-5, 5), (-5, 5)))
linkyaxes!(ax11, ax21)
linkyaxes!(ax12, ax22)
linkyaxes!(ax13, ax23)
linkxaxes!(ax11, ax12, ax13)
linkxaxes!(ax21, ax22, ax23)
g


# plot spatial basis functions
p11 = CairoMakie.contourf!(ax11, x, y, Uhat[:,:,1], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
CairoMakie.scatter!(ax11, locs[CI[:,1],1], locs[CI[:,1],2], color = (:black, transparencyValue), marker = 'x')
#
p12 = CairoMakie.contourf!(ax12, x, y, Uhat[:,:,2], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
CairoMakie.scatter!(ax12, locs[CI[:,2],1], locs[CI[:,2],2], color = (:black, transparencyValue), marker = 'x')
#
p13 = CairoMakie.contourf!(ax13, x, y, Uhat[:,:,3], 
    colormap = :balance, colorrange = (lowerBound, upperBound),
    levels = range(lowerBound, upperBound, step = stepBound))
#
CairoMakie.scatter!(ax13, locs[CI[:,3],1], locs[CI[:,3],2], color = (:black, transparencyValue), marker = 'x')
#
p21 = CairoMakie.contourf!(ax21, x, y, SD[:,:,1], 
    colormap = :matter, colorrange = (lowerBound, upperBound),
    levels = range(lowerBoundSD, upperBoundSD, step = stepBoundSD))
#
CairoMakie.scatter!(ax21, locs[CI[:,1],1], locs[CI[:,1],2], color = (:black, transparencyValue), marker = 'x')
#
p22 = CairoMakie.contourf!(ax22, x, y, SD[:,:,2], 
    colormap = :matter, colorrange = (lowerBound, upperBound),
    levels = range(lowerBoundSD, upperBoundSD, step = stepBoundSD))
#
CairoMakie.scatter!(ax22, locs[CI[:,2],1], locs[CI[:,2],2], color = (:black, transparencyValue), marker = 'x')
#
p23 = CairoMakie.contourf!(ax23, x, y, SD[:,:,3], 
    colormap = :matter, colorrange = (lowerBound, upperBound),
    levels = range(lowerBoundSD, upperBoundSD, step = stepBoundSD))
#
CairoMakie.scatter!(ax23, locs[CI[:,3],1], locs[CI[:,3],2], color = (:black, transparencyValue), marker = 'x')
#
CairoMakie.Colorbar(g[1,4], p11, ticks = lowerBound:stepSize:upperBound, 
    labelsize = 50, ticksize = 10, ticklabelsize = 20)
#
CairoMakie.Colorbar(g[2,4], p21, ticks = lowerBoundSD:stepSizeSD:upperBoundSD, 
    labelsize = 50, ticksize = 10, ticklabelsize = 20)
#


CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax13, grid = false)

CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax13, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
CairoMakie.hideydecorations!(ax23, grid = false)
g

# save("./results/twoSpatialDimensions/estimatedUQ2D.png", g)
# save("/Users/JSNorth/Documents/Presentations/LBLMLGroup/uncertaintyEstimateBasisFunctions2D.png", g)




#### Gif of smoothed surface

Ymean = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U, 3)])
Ydiff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' - Y for i in axes(posterior.U, 3)])

lims = (-1.05, 1.05).*maximum((maximum(abs, Y), maximum(abs, Ymean)))
limsDiff = (-1.05, 1.05).*maximum(abs, Ydiff)
nsteps = 40
nticks = 7


anim = @animate for i in 1:50 
    l = Plots.@layout [A{0.01h}; [a b]]
    title = Plots.plot(title = "Timestep $i", grid = false, showaxis = false, bottom_margin = -30Plots.px)
    p1 = Plots.contourf(x, y, reshape(Y[:,i], nx, ny), c = :balance, clim = lims, nlevels = nsteps, lw = 0.2, title = "Truth")
    p2 = Plots.contourf(x, y, reshape(Ymean[:,i], nx, ny), c = :balance, clim = lims, nlevels = nsteps, lw = 0.2, title = "Estimate", yaxis = false)
    # p3 = Plots.contourf(x, y, reshape(Ydiff[:,i], nx, ny), c = :balance, clim = limsDiff, nlevels = nsteps, lw = 0.2, title = "Difference", yaxis = false)
    Plots.plot(title, p1, p2, layout = l, size = (1200, 500))
end
# gif(anim, "/Users/JSNorth/Documents/Presentations/LBLMLGroup/SpatialEstimate.gif", fps = 4)



#endregion