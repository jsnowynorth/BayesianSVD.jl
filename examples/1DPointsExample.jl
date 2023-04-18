
include("../src/BasisFunctions.jl")

# using CairoMakie
using DataFrames, DataFramesMeta, Chain, CSV
# using LaTeXStrings



######################################################################
#### Generate Some Data
######################################################################
#region
m = 50
n = 30

#### need to sample such that the points are at least δ apart
# x = range(-5, 5, n)
x = sort(rand(Uniform(-5, 5), 100))
# t = range(0, 10, m)
t = sort(rand(Uniform(0, 10), 100))

lower = -5
upper = 5
δ = 0.05

# lower: lower axis value
# upper: upper axis value
# n: number of points
# δ: minimum distance between points
function sampleAxis(lower, upper, n, δ)
    if step(range(lower, upper, n)) < δ
        throw(DomainError(δ, "argument must be greater than " * string(step(range(lower, upper, n)))))
    else
        x = sort(rand(Uniform(lower, upper), n))
        d = x[2:end] - x[1:(end-1)]
        xKeep = x[vcat(BitArray(1), d .> δ)]
        
        while length(xKeep) < n
            
            xSamp = rand(Uniform(lower, upper), n-length(xKeep))
            x = sort(vcat(xKeep, xSamp))
            d = x[2:end] - x[1:(end-1)]
            xKeep = x[vcat(BitArray(1), d .> δ)]
        
        end
        return xKeep
    end
end

x = sampleAxis(-10, 10, n, 0.1)
t = sampleAxis(0, 20, m, 0.1)

ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5
ϵ = 0.1

Random.seed!(2)
# U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.scatter!(x, U, label = false)
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])
Plots.scatter!(t, V, label = false)

Plots.contourf(x, t, Y', clim = extrema(Z))
Plots.contourf(x, t, Z', clim = extrema(Z))


locs = reduce(hcat,reshape([[x, t] for x = x, t = t], n * m))'
Plots.scatter(locs[:,1], locs[:,2], zcolor = reshape(Y, :), label = false)
Plots.scatter(locs[:,1], locs[:,2], zcolor = reshape(Z, :), label = false)


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
# g = CairoMakie.Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1000, 700), linewidth = 5)
g = CairoMakie.Figure(;resolution = (1000, 700), linewidth = 5)
ax11 = Axis(g[1,1], yticks = [-0.3, -0.15, 0, 0.15, 0.3], limits = ((0, 10),(-0.3, 0.3)))
ax21 = Axis(g[2,1], ylabel = "Space", xlabel = "Time", limits = ((0,10), (0, 10)), xlabelsize = 20, ylabelsize = 20, xlabelfont = :bold, ylabelfont = :bold)
ax22 = Axis(g[2,2], xticks = [-0.24, -0.12, 0, 0.12, 0.24], xticklabelrotation = -pi/5, limits = ((-0.25, 0.25), (-5, 5)))
linkyaxes!(ax21, ax22)
linkxaxes!(ax21, ax11)



# plot time basis functions
CairoMakie.series!(ax11, t, V', labels = ["Basis Function $i" for i in axes(V,2)], color = colorlist, linewidth = LW)
CairoMakie.xlims!(ax11, low = 0, high = 10)
g


# plot spatial basis functions
for i in axes(U, 2)
    CairoMakie.lines!(ax22, U[:,i], x, label = "Basis Function $i", color = colorlist[i], linewidth = LW)
end
CairoMakie.ylims!(ax22, low = -5, high = 5)
g

# make legend
leg = CairoMakie.Legend(g[1,2], ax11, 
    patchsize = (40.0f0, 40.0f0),
    markersize = 10, labelsize = 15,
    height = 200, width = 200)
leg.tellheight = true
g


# contour plot
crange = (-1.05, 1.05).*maximum(abs, Z)
hm = CairoMakie.contourf!(ax21, t, x, Z', 
    colormap = :balance, colorrange = crange,
    levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

#
CairoMakie.Colorbar(g[1:2,3], hm, ticks = round.(range(crange[1], crange[2], length = nticks), digits = 3))

CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
g

colsize!(g.layout, 1, Fixed(600))
colsize!(g.layout, 2, Fixed(200))
rowsize!(g.layout, 1, Fixed(200))
rowsize!(g.layout, 2, Fixed(400))
g


# save("./results/oneSpatialDimension/generatedData1D.svg", g)
# save("./results/oneSpatialDimension/generatedData1D.png", g)


g = Plots.plot(x, U, c = [:blue :red :magenta :orange :green], linewidth = 2, legend = false, size = (700, 400))
g = Plots.plot!(x, (svd(Z).U[:,1:k]' .* [1, -1, 1, -1, -1])', c = [:blue :red :black :orange :green], linewidth = 2, linestyle = :dash, legend = false)
# save("/Users/JSNorth/Desktop/Uplot.svg", g)

g = Plots.plot(t, V, c = [:blue :red :magenta :orange :green], linewidth = 2, legend = false, size = (700, 400))
g = Plots.plot!(t, (svd(Z).V[:,1:k]' .* [1, -1, 1, -1, -1])', c = [:blue :red :black :orange :green], linewidth = 2, linestyle = :dash, legend = false)
# save("/Users/JSNorth/Desktop/Vplot.svg", g)

#endregion

######################################################################
#### Sample
######################################################################
#region

k = 5
ΩU = IdentityCorrelation(x)
ΩV = IdentityCorrelation(t)
data = Data(Z, rand(Normal(), n*m, 4), x, t, k)
pars = Pars(data, ΩU, ΩV)


k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)



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




svdY = svd(Y).U[:,1:k] * diagm(svd(Y).S[1:k]) * svd(Y).V[:,1:k]'
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

