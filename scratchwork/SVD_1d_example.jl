


using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra


######################################################################
#### Generate Some Data
######################################################################
m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 20, 10, 5, 2] # sqrt of eigenvalues
k = 5 # number of basis functions 
ϵ = 0.01 # noise

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)
# U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true)

Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Z))
Plots.contourf(x, t, Z', clim = extrema(Z))


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
ΩU = MaternKernel(x, ρ = 1, ν = 3.5, metric = Euclidean())
ΩV = MaternKernel(t, ρ = 1, ν = 3.5, metric = Euclidean())
data = Data(Z, ΩU, ΩV, k)
pars = Pars(data)


posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)


######################################################################
#### Plot some results
######################################################################

Plots.plot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple :blue :orange])
Plots.plot!(x, (U' .* [1, -1, 1, -1, -1])', label = false, color = "black", linewidth = 2)
Plots.plot!(x, svd(Z).U[:,1:data.k], label = false, linestyle = :dash, linewidth = 2, c = [:red :green :purple :blue :orange])


Plots.plot(posterior, t, size = (900, 500), basis = 'V', c = [:red :green :purple :blue :orange])
Plots.plot!(t, (V' .* [1, -1, 1, -1, 1])', label = false, color = "black", linewidth = 2)
Plots.plot!(t, svd(Z).V[:,1:data.k], c = [:red :green :purple :blue :orange], linestyle = :dash, label = false, linewidth = 2)


#### posterior chains
Plots.plot(posterior.σU', title = "σU")
Plots.plot(posterior.σV', title = "σV")
Plots.plot(posterior.σ, title = "σ")
Plots.plot(posterior.D', title = "D")
Plots.plot(posterior.U[30,:,:]', title = "U")
Plots.plot(posterior.V[30,:,:]', title = "V")

Plots.histogram(posterior.D')



svdZ = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'
Z_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
# Y_hat = mean(Yest)

Z_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])



lims = (-1.05, 1.05).*maximum(abs, Z)


l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Z_hat, title = "Y Hat", c = :balance, clim = lims)
p2 = Plots.contourf(t, x, Y, title = "Truth", c = :balance, clim = lims)
p3 = Plots.contourf(t, x, svdZ, title = "Algorithm", c = :balance, clim = lims)
p4 = Plots.contourf(t, x, Z_hat .- Y, title = "Y Hat - Truth", c = :balance, clim = (-0.2, 0.2))
p5 = Plots.contourf(t, x, Z, title = "Observed", c = :balance, clim = lims)
p6 = Plots.contourf(t, x, svdZ .- Y, title = "Algorithm - Truth", c = :balance, clim = (-0.2, 0.2))
Plots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))


