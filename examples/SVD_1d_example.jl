
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

# using CairoMakie
# using DataFrames, DataFramesMeta, Chain, CSV


######################################################################
#### Generate Some Data
######################################################################
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())


Random.seed!(2)

k = 5
Φ = PON(n, k, ΣU.K)
Ψ = PON(m, k, ΣV.K)

Plots.plot(x, Φ, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, Ψ, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])


# sqrt of eigenvalues
d1 = 40
d2 = 20
d3 = 10
d4 = 5
d5 = 2
D = diagm([d1, d2, d3, d4, d5])

ϵ = rand(Normal(0, sqrt(0.01)), n, m)
Y = Φ * D * Ψ' + ϵ # n × m

mean((Φ * D * Ψ'))^2 / mean(Y)^2
mean((Φ * D * Ψ')).^2 / 0.01
mean(Φ * D * Ψ') / sqrt(0.01)

mean(Y)^2/var(Y)



Plots.contourf(x, t, (Φ * D * Ψ')', clim = extrema(Y))
Plots.contourf(x, t, Y', clim = extrema(Y))


######################################################################
#### Visualize Data
######################################################################


# set up plot
# g = CairoMakie.Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1000, 700), linewidth = 5)
g = CairoMakie.Figure(resolution = (1000, 700), linewidth = 5)
ax11 = Axis(g[1,1])
ax21 = Axis(g[2,1], ylabel = "Space", xlabel = "Time", limits = ((0,10), (0, 10)))
# ax22 = Axis(g[2,2], xticks = -0.35:0.15:0.35, xticklabelrotation = -pi/5, limits = ((-0.4, 0.4), (0, 10)))
ax22 = Axis(g[2,2], xticks = [-0.35, -0.15, 0, 0.15, 0.35], xticklabelrotation = -pi/5, limits = ((-0.4, 0.4), (-5, 5)))
linkyaxes!(ax21, ax22)
linkxaxes!(ax21, ax11)

# plot time basis functions
CairoMakie.lines!(ax11, t, Ψ[:,1], color = :blue, label = "Basis 1")
CairoMakie.lines!(ax11, t, Ψ[:,2], color = :red, label = "Basis 2")
CairoMakie.lines!(ax11, t, Ψ[:,3], color = :black, label = "Basis 3")
CairoMakie.lines!(ax11, t, Ψ[:,4], color = :orange, label = "Basis 4")
CairoMakie.lines!(ax11, t, Ψ[:,5], color = :green, label = "Basis 5")
CairoMakie.xlims!(ax11, low = 0, high = 10)
# CairoMakie.text!(ax11, 0.5, -0.1, text = "U", font = :bold, textsize = 26)
g


# plot spatial basis functions
CairoMakie.lines!(ax22, Φ[:,1], x, color = :blue, label = "Basis 1")
CairoMakie.lines!(ax22, Φ[:,2], x, color = :red, label = "Basis 2")
CairoMakie.lines!(ax22, Φ[:,3], x, color = :black, label = "Basis 3")
CairoMakie.lines!(ax22, Φ[:,4], x, color = :orange, label = "Basis 4")
CairoMakie.lines!(ax22, Φ[:,5], x, color = :green, label = "Basis 5")
CairoMakie.ylims!(ax22, low = -5, high = 5)
# CairoMakie.text!(ax22, 0, -4.9, text = "V", font = :bold, textsize = 26)
g
# ax22.xticks = -0.35:0.1:0.35

# make legend
leg = CairoMakie.Legend(g[1,2], ax11, 
    patchsize = (40.0f0, 40.0f0),
    markersize = 10, labelsize = 15,
    height = 200, width = 200)
leg.tellheight = true
g


# contour plot
hm = CairoMakie.contourf!(ax21, t, x, Y', 
    colormap = :balance, colorrange = (-ceil(maximum(abs.(extrema(Y))), digits = 1), ceil(maximum(abs.(extrema(Y))), digits = 1)),
    levels = range(-ceil(maximum(abs.(extrema(Y))), digits = 1), ceil(maximum(abs.(extrema(Y))), digits = 1), step = 0.1))

#
tmark = -ceil(maximum(abs.(extrema(Y))), digits = 1):(2*ceil(maximum(abs.(extrema(Y))), digits = 1)/8):ceil(maximum(abs.(extrema(Y))), digits = 1)
# clean up and add color bar
CairoMakie.hidexdecorations!(ax11, grid = false)
CairoMakie.hideydecorations!(ax22, grid = false)
# CairoMakie.colgap!(g, 20)
# CairoMakie.rowgap!(g, 20)
CairoMakie.Colorbar(g[1:2,3], hm, ticks = round.(tmark, digits = 2))
# CairoMakie.Colorbar(g[3,1], hm, ticks = -3:1:3, vertical = false, flipaxis = false)
g

colsize!(g.layout, 1, Fixed(600))
colsize!(g.layout, 2, Fixed(200))
rowsize!(g.layout, 1, Fixed(200))
rowsize!(g.layout, 2, Fixed(400))
g

# Label(g[1, 1, TopLeft()], "A", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# Label(g[2, 1, TopLeft()], "B", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# Label(g[2, 2, TopLeft()], "C", textsize = 26, font = :bold, padding = (0, 5, 5, 0), halign = :right)
# g


# save("./results/oneSpatialDimension/generatedData1D.png", g)



######################################################################
#### Sample
######################################################################

k = 5
ΩU = IdentityKernel(x)
ΩV = IdentityKernel(t)
data = Data(Y, ΩU, ΩV, k)
pars = Pars(data)

# k=3
# ΩU = ExponentialKernel(x, ρ = 0.6)
# ΩV = ExponentialKernel(t, ρ = 0.3)
# data = Data(Y, ΩU, ΩV, k)
# pars = Pars(data)

# k=3
# ΩU = GaussianKernel(x, ρ = 0.6)
# ΩV = GaussianKernel(t, ρ = 0.3)
# data = Data(Y, ΩU, ΩV, k)
# pars = Pars(data)


k = 5
ΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Euclidean())
ΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Euclidean())
data = Data(Y, ΩU, ΩV, k)
pars = Pars(data)

# pars.U = copy(Φ)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)

# posterior.U[:,:,1]' * posterior.U[:,:,1]
# posterior.V[:,:,1]' * posterior.V[:,:,1]

# posterior.U_hat' * posterior.U_hat
# posterior.V_hat' * posterior.V_hat


Plots.plot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple :blue :orange])
# Plots.plot!(x, Φ, label = false, color = "black", linewidth = 2)
Plots.plot!(x, (Φ' .* [1, 1, 1, -1, -1])', label = false, color = "black", linewidth = 2)
Plots.plot!(x, svd(Y).U[:,1:data.k] .* [1, 1, 1, 1, 1]', label = false, linestyle = :dash, linewidth = 2, c = [:red :green :purple :blue :orange])


Plots.plot(posterior, t, size = (900, 500), basis = 'V', c = [:red :green :purple :blue :orange])
Plots.plot!(t, (Ψ' .* [1, 1, 1, -1, -1])', label = false, color = "black", linewidth = 2)
Plots.plot!(t, (svd(Y).V[:,1:data.k]' .* [1, 1, 1, 1, 1])', c = [:red :green :purple :blue :orange], linestyle = :dash, label = false, linewidth = 2)
#
# savefig("./results/oneSpatialDimension/Vplots.png")



#### Some plots
Plots.plot(posterior, x, size = (900, 600), basis = 'U', c = [:purple :green :red], label = ["U1" "U2" "U3"], linewidth = 2)
Plots.plot!(x, (Φ' .* [-1, -1, 1])', label = false, color = "black", linewidth = 2)
Plots.plot!(x, svd(Y).U[:,1:data.k] .* [1, 1, 1]', label = false, c = [:purple :green :red],linestyle = :dash, linewidth = 2)
#
# savefig("./results/oneSpatialDimension/Uplots.png")

Plots.plot(posterior, t, size = (900, 500), basis = 'V', c = [:purple :green :red], label = ["V1" "V2" "V3"], linewidth = 2)
Plots.plot!(t, (Ψ' .* [1, 1, 1, -1, 1])', label = false, color = "black", linewidth = 2)
Plots.plot!(t, (svd(Y).V[:,1:data.k]' .* [1, 1, 1])', c = [:purple :green :red], linestyle = :dash, label = false, linewidth = 2)
#
# savefig("./results/oneSpatialDimension/Vplots.png")


# Plots.plot(x, (Φ' .* [1, 1, 1])', label = false, c = [:blue :red :black], linewidth = 2, size = (900, 500))
# Plots.plot!(x, svd(Y).U[:,1:data.k] .* [-1, -1, 1]', label = false, c = [:blue :red :black], linestyle = :dash, linewidth = 2)
# #
# savefig("./results/oneSpatialDimension/Uplots_ASVD.png")

# Plots.plot(t, (Ψ' .* [1, 1, 1])', label = false, c = [:blue :red :black], linewidth = 2, size = (900, 500))
# Plots.plot!(t, (svd(Y).V[:,1:data.k]' .* [-1, -1, 1])', c = [:blue :red :black], linestyle = :dash, label = false, linewidth = 2)
# #
# savefig("./results/oneSpatialDimension/Vplots_ASVD.png")



Plots.plot(posterior, x, size = (900, 600), basis = 'U', c = [:purple :green :red])
Plots.plot!(x, (Φ' .* [-1, -1, 1])', label = ["U₁ True" "U₂ True" "U₃ True"], color = "black")
Plots.plot!(x, svd(Y).U[:,1:data.k] .* [1, 1, 1]', label = ["U" * string(i) * " Trad" for i in (1:k)'], c = [:purple :green :red],linestyle = :dash)
#

Plots.plot(posterior, t, size = (900, 500), basis = 'V', c = [:purple :green :red])
Plots.plot!(t, (Ψ' .* [-1, -1, 1])', label = ["U₁ True" "U₂ True" "U₃ True"], color = "black")
Plots.plot!(t, (svd(Y).V[:,1:data.k]' .* [1, 1, 1])', label = ["V" * string(i) * " Trad" for i in (1:k)'], color = "red", linestyle = :dash)
#



Plots.plot(posterior.σU', title = "σU")
Plots.plot(posterior.σV', title = "σV")
Plots.plot(posterior.σ, title = "σ")
Plots.plot(posterior.D', title = "D")
Plots.plot(posterior.U[30,:,:]', title = "U")
Plots.plot(posterior.V[30,:,:]', title = "V")

Plots.histogram(posterior.D')

# sum((posterior.D[1,2:end] .- ShiftedArrays.lag(posterior.D[1,:])[2:end]) .!= 0.0)/5000
# sum((posterior.D[2,2:end] .- ShiftedArrays.lag(posterior.D[2,:])[2:end]) .!= 0.0)/5000
# sum((posterior.D[3,2:end] .- ShiftedArrays.lag(posterior.D[3,:])[2:end]) .!= 0.0)/5000


# mean([posterior.U[:,:,i]' * posterior.U[:,:,i] for i in axes(posterior.U, 3)])
# mean([posterior.V[:,:,i]' * posterior.V[:,:,i] for i in axes(posterior.U, 3)])

# pars.U'*pars.U
# pars.V'*pars.V
# pars.rU
# pars.rV
# pars.D


svdY = svd(Y).U[:,1:k] * diagm(svd(Y).S[1:k]) * svd(Y).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
# Y_hat = mean(Yest)

Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- (Φ * D * Ψ') for i in axes(posterior.U,3)])

lims = (-1.5, 1.5)


l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Y_hat, title = "Y Hat", c = :balance, clim = (-3, 3))
p2 = Plots.contourf(t, x, Φ * D * Ψ', title = "Truth", c = :balance, clim = (-3, 3))
p3 = Plots.contourf(t, x, svdY, title = "Algorithm", c = :balance, clim = (-3, 3))
p4 = Plots.contourf(t, x, Y_hat .- (Φ * D * Ψ'), title = "Y Hat - Truth", c = :balance, clim = (-0.4, 0.4))
p5 = Plots.contourf(t, x, Y, title = "Observed", c = :balance, clim = (-3, 3))
p6 = Plots.contourf(t, x, svdY .- (Φ * D * Ψ'), title = "Algorithm - Truth", c = :balance, clim = (-0.4, 0.4))
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

######################################################################
#### Plot Results
######################################################################

#### Plot basis functions

## U plot
# set up plot
g = CairoMakie.Figure(resolution = (1000, 700), linewidth = 2)
ax1 = Axis(g[1,1], xlabel = "Space", limits = ((-5,5), (-0.4, 0.4)))
# linkxaxes!(ax1, ax2)
# g


# plot spatial basis functions
transparencyValue = 0.2
CairoMakie.lines!(ax1, x, posterior.U_hat[:,1], color = :blue, label = "Estimate 1")
CairoMakie.band!(ax1, x, posterior.U_lower[:,1], posterior.U_upper[:,1], color = (:blue, transparencyValue))
CairoMakie.lines!(ax1, x, posterior.U_hat[:,2], color = :red, label = "Estimate 2")
CairoMakie.band!(ax1, x, posterior.U_lower[:,2], posterior.U_upper[:,2], color = (:red, transparencyValue))
CairoMakie.lines!(ax1, x, posterior.U_hat[:,3], color = :green, label = "Estimate 3")
CairoMakie.band!(ax1, x, posterior.U_lower[:,3], posterior.U_upper[:,3], color = (:green, transparencyValue))
CairoMakie.lines!(ax1, x, -Φ[:,1], color = :black, label = "True 1")
CairoMakie.lines!(ax1, x, -Φ[:,2], color = :black, label = "True 2")
CairoMakie.lines!(ax1, x, Φ[:,3], color = :black, label = "True 3")
CairoMakie.lines!(ax1, x, svd(Y).U[:,1], color = :blue, label = "Algorithm 1", linestyle = :dot)
CairoMakie.lines!(ax1, x, svd(Y).U[:,2], color = :red, label = "Algorithm 2", linestyle = :dot)
CairoMakie.lines!(ax1, x, svd(Y).U[:,3], color = :green, label = "Algorithm 3", linestyle = :dot)
g


# make legend patchsize = (40.0f0, 40.0f0), height = 150, width = 150
leg = CairoMakie.Legend(g[1,2], ax1, 
    markersize = 20, labelsize = 25)
# leg.tellheight = true
g




# set up plot
g = CairoMakie.Figure(resolution = (1500, 500), linewidth = 2)
ax1 = Axis(g[1,1], xlabel = "Space", limits = ((-5,5), (-0.4, 0.4)))
ax2 = Axis(g[1,2], xlabel = "Time", limits = ((0,10), (-0.25, 0.25)))
# linkxaxes!(ax1, ax2)
g


# plot spatial basis functions
transparencyValue = 0.2
CairoMakie.lines!(ax1, x, posterior.U_hat[:,1], color = :blue, label = "Estimate 1")
CairoMakie.band!(ax1, x, posterior.U_lower[:,1], posterior.U_upper[:,1], color = (:blue, transparencyValue))
CairoMakie.lines!(ax1, x, posterior.U_hat[:,2], color = :red, label = "Estimate 2")
CairoMakie.band!(ax1, x, posterior.U_lower[:,2], posterior.U_upper[:,2], color = (:red, transparencyValue))
CairoMakie.lines!(ax1, x, posterior.U_hat[:,3], color = :green, label = "Estimate 3")
CairoMakie.band!(ax1, x, posterior.U_lower[:,3], posterior.U_upper[:,3], color = (:green, transparencyValue))
CairoMakie.lines!(ax1, x, -Φ[:,1], color = :black, label = "True 1")
CairoMakie.lines!(ax1, x, -Φ[:,2], color = :black, label = "True 2")
CairoMakie.lines!(ax1, x, Φ[:,3], color = :black, label = "True 3")
CairoMakie.lines!(ax1, x, svd(Y).U[:,1], color = :blue, label = "Algorithm 1")
CairoMakie.lines!(ax1, x, svd(Y).U[:,2], color = :red, label = "Algorithm 2")
CairoMakie.lines!(ax1, x, svd(Y).U[:,3], color = :green, label = "Algorithm 3")
g

# plot temporal basis functions
CairoMakie.lines!(ax2, t, posterior.V_hat[:,1], color = :blue, label = "Estimate 1")
CairoMakie.band!(ax2, t, posterior.V_lower[:,1], posterior.V_upper[:,1], color = (:blue, transparencyValue))
CairoMakie.lines!(ax2, t, posterior.V_hat[:,2], color = :red, label = "Estimate 2")
CairoMakie.band!(ax2, t, posterior.V_lower[:,2], posterior.V_upper[:,2], color = (:red, transparencyValue))
CairoMakie.lines!(ax2, t, posterior.V_hat[:,3], color = :green, label = "Estimate 3")
CairoMakie.band!(ax2, t, posterior.V_lower[:,3], posterior.V_upper[:,3], color = (:green, transparencyValue))
CairoMakie.lines!(ax2, t, -Ψ[:,1], color = :black, label = "True 1")
CairoMakie.lines!(ax2, t, -Ψ[:,2], color = :black, label = "True 2")
CairoMakie.lines!(ax2, t, Ψ[:,3], color = :black, label = "True 3")
CairoMakie.lines!(ax2, t, svd(Y).V[:,1], color = :blue, label = "Algorithm 1")
CairoMakie.lines!(ax2, t, svd(Y).V[:,2], color = :red, label = "Algorithm 2")
CairoMakie.lines!(ax2, t, svd(Y).V[:,3], color = :green, label = "Algorithm 3")
g

# make legend patchsize = (40.0f0, 40.0f0), height = 150, width = 150
leg = CairoMakie.Legend(g[1,3], ax1, 
    markersize = 20, labelsize = 25)
leg.tellheight = true
g





#### Plot recovered surface

svdY = svd(Y).U[:,1:k] * diagm(svd(Y).S[1:k]) * svd(Y).V[:,1:k]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'


# set up plot
g = CairoMakie.Figure(resolution = (1000, 700))
ax11 = Axis(g[1,1], xlabel = "Time", ylabel = "Space", limits = ((0,10), (-5, 5)), title = "Truth")
# ax21 = Axis(g[2,1], ylabel = "Space", xlabel = "Time", limits = ((0,10), (-5, 5)), title = "Observed")
ax12 = Axis(g[1,2], limits = ((0,10), (-5, 5)), title = "Estimate")
ax22 = Axis(g[2,2], ylabel = "Space", xlabel = "Time", limits = ((0,10), (-5, 5)), title = "Difference")
ax13 = Axis(g[1,3], limits = ((0,10), (-5, 5)), title = "Algorithm")
ax23 = Axis(g[2,3], xlabel = "Time", limits = ((0,10), (-5, 5)), title = "Difference")
linkyaxes!(ax11, ax12, ax13)
linkxaxes!(ax22, ax23)
g

# plot results
p11 = CairoMakie.contourf!(ax11, t, x, (Φ * D * Ψ')',
    colormap = :balance, colorrange = (-2.5, 2.5),
    levels = range(-3, 3, step = 0.2), )
#
p12 = CairoMakie.contourf!(ax12, t, x, Y_hat',
    colormap = :balance, colorrange = (-2.5, 2.5),
    levels = range(-3, 3, step = 0.2))
#
p13 = CairoMakie.contourf!(ax13, t, x, svdY',
    colormap = :balance, colorrange = (-2.5, 2.5),
    levels = range(-3, 3, step = 0.2))
#
CairoMakie.Colorbar(g[1,4], p11, ticks = -3:1:3)

# p21 = CairoMakie.contourf!(ax21, t, x, Y',
#     colormap = :balance, colorrange = (-2.5, 2.5),
#     levels = range(-3, 3, step = 0.2))
#
p22 = CairoMakie.contourf!(ax22, t, x, (Y_hat .- (Φ * D * Ψ'))',
    colormap = :balance, colorrange = (-0.4, 0.4),
    levels = range(-0.4, 0.4, step = 0.02))
#
p23 = CairoMakie.contourf!(ax23, t, x, (svdY .- (Φ * D * Ψ'))',
    colormap = :balance, colorrange = (-0.4, 0.4),
    levels = range(-0.4, 0.4, step = 0.02))
#
CairoMakie.Colorbar(g[2,4], p23, ticks = -0.4:0.1:0.4)


CairoMakie.hidexdecorations!(ax12, grid = false)
CairoMakie.hidexdecorations!(ax13, grid = false)
CairoMakie.hideydecorations!(ax12, grid = false)
CairoMakie.hideydecorations!(ax13, grid = false)
CairoMakie.hideydecorations!(ax23, grid = false)
g

# colsize!(g.layout, 1, Fixed(650))
# colsize!(g.layout, 2, Fixed(150))
# rowsize!(g.layout, 1, Fixed(150))
# rowsize!(g.layout, 2, Fixed(450))
# g



# save("./results/oneSpatialDimension/generatedData1D.png", g)








######################################################################
#### Simulation Study
######################################################################

σ = sqrt.([0.01, 0.1, 1]) # set the standard deviation
k = [1, 2, 3, 4, 5]
ρ = [1, 2, 3, 4, 5]

# σ = sqrt.([0.01, 0.1, 1]) # set the standard deviation
# k = [1, 3, 5]
# ρ = [1, 3, 5]


ncombos = copy(reduce(hcat, reshape([[σ, k, ρ] for σ=σ, k=k, ρ=ρ], length(σ)*length(k)*length(ρ)))')
nreps = 4
df = DataFrame(RMSEData = zeros(nreps), RMSEU = zeros(nreps), RMSEV = zeros(nreps), coverData = zeros(nreps), coverU = zeros(nreps), coverV = zeros(nreps))

sim = Dict()
for i in axes(ncombos, 1)
    sim[i] = Dict("df" => DataFrame(RMSEData = zeros(nreps), RMSEU = zeros(nreps), RMSEV = zeros(nreps), coverData = zeros(nreps), coverU = zeros(nreps), coverV = zeros(nreps)), 
                    'σ' => ncombos[i,1], 'k' => Int(ncombos[i,2]), 'ρ' => ncombos[i,3])
end



# sim[i]["df"]
# sim[i]['σ']
# sim[i]['k']
# sim[i]['ρ']

Φtmp = hcat(Φ, zeros(n, 2))
Ψtmp = hcat(Ψ, zeros(m, 2))
Dtmp = diagm(vcat(diag(D), 0, 0))

p = Progress(size(ncombos,1), desc = "Simulating..."; showspeed = true)
for i in axes(ncombos, 1)

    Random.seed!(10)
    ϵ = rand(Normal(0, sim[i]['σ']), n, m)
    Y = Φ * D * Ψ' + ϵ # n × m

    Threads.@threads for j in 1:nreps

        # k = sim[i]['k']
        # ΩU = IdentityKernel(x)
        # ΩV = IdentityKernel(t)
        # data = Data(Y, ΩU, ΩV, k)
        # pars = Pars(data)

        k = sim[i]['k']
        ΩU = MaternKernel(x, ρ = sim[i]['ρ'], ν = 4, metric = Euclidean())
        ΩV = MaternKernel(t, ρ = 1, ν = 4, metric = Euclidean())
        data = Data(Y, ΩU, ΩV, k)
        pars = Pars(data)

        posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = false)

        Yest = [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U, 3)]

        trsfrm = ones(k)
        for j in 1:k
            if posteriorCoverage(Φtmp[:,j], posterior.U[:,j,:], 0.95) > posteriorCoverage(-Φtmp[:,j], posterior.U[:,j,:], 0.95)
                continue
            else
                trsfrm[j] = -1.0
            end
        end

        sim[i]["df"][!,:RMSEData][j] = sqrt(mean(((Φ * D * Ψ') .- mean(Yest)).^2))
        sim[i]["df"][!,:RMSEU][j] = sqrt(mean((Φtmp[:,1:k] .- posterior.U_hat).^2))
        sim[i]["df"][!,:RMSEV][j] = sqrt(mean((Ψtmp[:,1:k] .- posterior.V_hat).^2))
        sim[i]["df"][!,:coverData][j] = posteriorCoverage(Φ * D * Ψ', Yest, 0.95)
        sim[i]["df"][!,:coverU][j] = posteriorCoverage(Matrix((Φtmp[:,1:k]' .* trsfrm)'), posterior.U, 0.95)
        sim[i]["df"][!,:coverV][j] = posteriorCoverage(Matrix((Ψtmp[:,1:k]' .* trsfrm)'), posterior.V, 0.95)
        
    end

    ProgressMeter.next!(p)
end



#### write to csv

dfstore = sim[1]["df"]
dfstore[!,:sigma] = fill(sim[1]['σ'], nreps)
dfstore[!,:k] = fill(sim[1]['k'], nreps)
dfstore[!,:rho] = fill(sim[1]['ρ'], nreps)

for i in 2:length(sim)
    dftmp = sim[i]["df"]
    dftmp[!,:sigma] = fill(sim[i]['σ'], nreps)
    dftmp[!,:k] = fill(sim[i]['k'], nreps)
    dftmp[!,:rho] = fill(sim[i]['ρ'], nreps)
    dfstore = vcat(dfstore, dftmp)
end

# CSV.write("/Users/JSNorth/Desktop/tmpsim.csv", dfstore)



# df = DataFrame(copy(reduce(hcat, reshape([[σ, k, ρ] for σ=σ, k=k, ρ=ρ], length(σ)*length(k)*length(ρ)))'), :auto)
# rename!(df,:x1 => :standardDeviation)
# rename!(df,:x2 => :numberBasis)
# rename!(df,:x3 => :effectiveDistance)
# df[!,:RMSEData] = zeros(size(df,1))
# df[!,:RMSEU] = zeros(size(df,1))
# df[!,:RMSEV] = zeros(size(df,1))
# df[!,:coverData] = zeros(size(df,1))
# df[!,:coverU] = zeros(size(df,1))
# df[!,:coverV] = zeros(size(df,1))

# df[!,:numberBasis] = convert.(Int, df[!,:numberBasis])

# df

# p = Progress(size(df, 1), desc = "Simulating..."; showspeed = true)
# # for i in axes(df, 1)
# for i in axes(df, 1)

#     Random.seed!(10)
#     ϵ = rand(Normal(0, df[:,:standardDeviation][i]), n, m)
#     Y = Φ * D * Ψ' + ϵ # n × m

#     # k = df[:,:numberBasis][i]
#     # ΩU = IdentityKernel(x)
#     # ΩV = IdentityKernel(t)
#     # data = Data(Y, ΩU, ΩV, k)
#     # pars = Pars(data)

#     k = df[:,:numberBasis][i]
#     ΩU = MaternKernel(x, ρ = df[:,:effectiveDistance][i], ν = 4, metric = Euclidean())
#     ΩV = MaternKernel(t, ρ = 1, ν = 4, metric = Euclidean())
#     data = Data(Y, ΩU, ΩV, k)
#     pars = Pars(data)

#     posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = false)

#     Yest = [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U, 3)]

#     trsfrm = ones(minimum([df[:,:numberBasis][i], 3]))
#     for j in 1:minimum([df[:,:numberBasis][i], 3])
#         if posteriorCoverage(Φ[:,j], posterior.U[:,j,:], 0.95) > posteriorCoverage(-Φ[:,j], posterior.U[:,j,:], 0.95)
#             continue
#         else
#             trsfrm[j] = -1.0
#         end
#     end

#     nbf = minimum([df[:,:numberBasis][i], 3])

#     df[!,:RMSEData][i] = sqrt(mean(((Φ * D * Ψ') .- mean(Yest)).^2))
#     df[!,:RMSEU][i] = sqrt(mean((Φ[:,1:nbf] .- posterior.U_hat[:,1:nbf]).^2))
#     df[!,:RMSEV][i] = sqrt(mean((Ψ[:,1:nbf] .- posterior.V_hat[:,1:nbf]).^2))
#     df[!,:coverData][i] = posteriorCoverage(Φ * D * Ψ', Yest, 0.95)
#     df[!,:coverU][i] = posteriorCoverage(Matrix((Φ[:,1:nbf]' .* trsfrm)'), posterior.U[:,1:nbf,:], 0.95)
#     df[!,:coverV][i] = posteriorCoverage(Matrix((Ψ[:,1:nbf]' .* trsfrm)'), posterior.V[:,1:nbf,:], 0.95)

#     ProgressMeter.next!(p)

# end




# ρcolor = [:purple, :green, :red]
# σshape = [:circle, :utriangle, :rect]

# g = CairoMakie.Figure(resolution = (1200, 700), linewidth = 2)
# ax11 = Axis(g[1,1], xlabel = "k", ylabel = "RMSE", title = "RMSE Data", xticks = 1:2:5)
# ax12 = Axis(g[1,2], xlabel = "k", ylabel = "RMSE", title = "RMSE U", xticks = 1:2:5)
# ax13 = Axis(g[1,3], xlabel = "k", ylabel = "RMSE", title = "RMSE V", xticks = 1:2:5)
# ax21 = Axis(g[2,1], xlabel = "k", ylabel = "coverage", title = "Coverage Data", xticks = 1:2:5)
# ax22 = Axis(g[2,2], xlabel = "k", ylabel = "coverage", title = "Coverage U", xticks = 1:2:5)
# ax23 = Axis(g[2,3], xlabel = "k", ylabel = "coverage", title = "Coverage V", xticks = 1:2:5)
# g

# linkyaxes!(ax11, ax12, ax13)
# linkyaxes!(ax21, ax22, ax23)

# linkxaxes!(ax11, ax12, ax13, ax21, ax22, ax23)
# g

# for i in 1:3
#     for j in 1:3

#         dfsub = @chain df begin
#             @subset(:standardDeviation .== σ[i])
#             @subset(:effectiveDistance .== ρ[j])
#         end

#         # RMSE Data
#         CairoMakie.lines!(ax11, dfsub[:,:numberBasis], dfsub[:,:RMSEData], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax11, dfsub[:,:numberBasis], dfsub[:,:RMSEData], color = ρcolor[j], marker = σshape[i])

#         # RMSE U
#         CairoMakie.lines!(ax12, dfsub[:,:numberBasis], dfsub[:,:RMSEU], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax12, dfsub[:,:numberBasis], dfsub[:,:RMSEU], color = ρcolor[j], marker = σshape[i])

#         # RMSE V
#         CairoMakie.lines!(ax13, dfsub[:,:numberBasis], dfsub[:,:RMSEV], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax13, dfsub[:,:numberBasis], dfsub[:,:RMSEV], color = ρcolor[j], marker = σshape[i])

#         # Coverage Data
#         CairoMakie.lines!(ax21, dfsub[:,:numberBasis], dfsub[:,:coverData], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax21, dfsub[:,:numberBasis], dfsub[:,:coverData], color = ρcolor[j], marker = σshape[i])

#         # Coverage U
#         CairoMakie.lines!(ax22, dfsub[:,:numberBasis], dfsub[:,:coverU], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax22, dfsub[:,:numberBasis], dfsub[:,:coverU], color = ρcolor[j], marker = σshape[i])

#         # Coverage V
#         CairoMakie.lines!(ax23, dfsub[:,:numberBasis], dfsub[:,:coverV], color = ρcolor[j], marker = σshape[i])
#         CairoMakie.scatter!(ax23, dfsub[:,:numberBasis], dfsub[:,:coverV], color = ρcolor[j], marker = σshape[i])

#     end
# end
# g



# elem_1 = [LineElement(color = ρcolor[1], linestyle = nothing, linewidth = 5)]
# elem_2 = [LineElement(color = ρcolor[2], linestyle = nothing, linewidth = 5)]
# elem_3 = [LineElement(color = ρcolor[3], linestyle = nothing, linewidth = 5)]
# elem_4 = [MarkerElement(color = :black, marker = σshape[1], markersize = 15)]
# elem_5 = [MarkerElement(color = :black, marker = σshape[2], markersize = 15)]
# elem_6 = [MarkerElement(color = :black, marker = σshape[3], markersize = 15)]


# Legend(g[1:2,4],
#     [elem_1, elem_2, elem_3, elem_4, elem_5, elem_6],
#     [string("ρ = ", ρ[1]), string("ρ = ", ρ[2]), string("ρ = ", ρ[3]),
#     string("σ = ", σ[1]), string("σ = ", round(σ[2], digits = 2)), string("σ = ", σ[3])],
#     patchsize = (35, 35), rowgap = 10, labelsize = 20)
# #
# g

