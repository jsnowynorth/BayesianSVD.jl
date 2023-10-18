

using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra, Statistics
using Kronecker


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
# D = [100, 70, 50, 20, 10]
k = 5
ϵ = 0.01

Random.seed!(1)

# spatial process
Xs = hcat(exp.(-(x .- 1).^2), sin.((2*pi .* x) ./ 5))
# Xs = Xs .+ rand(MvNormal(zeros(2), 0.1*diagm(reshape(var(Xs, dims = 1),:))), 50)'

Xs = rand(MvNormal(zeros(n), MaternCorrelation(x, ρ = 1, ν = 3.5, metric = Euclidean()).K), 2)


# temporal process
# Xt = hcat(Vector(t))
Xt = cos.((2*pi .* t) ./ 10)

Xt = rand(MvNormal(zeros(m), MaternCorrelation(t, ρ = 0.5, ν = 3.5, metric = Euclidean()).K), 1)

# space-time process
Ω = MaternCorrelation(x, t, ρ = 1, ν = 3.5, metric = Euclidean())
βst = vcat(-0.6, 0.8)
Xst = rand(MvNormal(zeros(n*m), Ω.K), length(βst))

# covariates and parameters
X = kronecker(Xt, Xs)
X = convert(Matrix, X)
β = vcat(-1, 2)

X = [X Xst]
β = [β; βst]

Ps = diagm(ones(n)) - Xs * inv(Xs' * Xs) * Xs'
Pt = diagm(ones(m)) - Xt * inv(Xt' * Xt) * Xt'

Random.seed!(2)
# U, V, Y, Z = GenerateCorrelatedData(ΣU, ΣV, D, k, X, β, Ps, Pt, 2; SNR = true)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2; SNR = true)

Z = Z .+ reshape(X * β, n, m)


Plots.plot(x, U, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(t, V, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Y', clim = extrema(Y))
Plots.contourf(x, t, Z', clim = extrema(Z))



######################################################################
#### Sample
######################################################################
#region

Ps = diagm(ones(n))
Pt = diagm(ones(m))

k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, X, Ps, Pt, x, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 10000, burnin = 5000)

posterior.β_hat'
posterior.β_lower'
posterior.β_upper'

posterior.D_hat
posterior.D_lower
posterior.D_upper
posterior.σ_hat
posterior.σU_hat
posterior.σV_hat


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
Up = reshape(reduce(hcat, Up), 50, 5, :)

i=1
Plots.plot(Up[25,i,:], title = "U")
Plots.plot!(posterior.U[25,i,:], title = "U")


i=2
Plots.plot(U[:,i], title = "U", c = :black)
Plots.plot!(Up[:,i,Vector(range(1,size(Up, 3), step = 100))], title = "U", label = false, c = :gray, linealpha = 0.4)
Plots.plot!(posterior.U[:,i,Vector(range(1,size(Up, 3), step = 100))], title = "U", label = false, c = :blue, linealpha = 0.2)


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




svdZ = svd(Z - reshape(X * β, n, m))
svdY = svdZ.U[:,1:5] * diagm(svdZ.S[1:5]) * svdZ.V[:,1:5]'
Y_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'
# Y_hat = mean(Yest)

Y_hat = mean([Ps * posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' * Pt' for i in axes(posterior.U,3)])
Y_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])

lims = (-5, 5)


l = Plots.@layout [a b c; d e f]
p1 = Plots.contourf(t, x, Y_hat, title = "Y Hat", c = :balance)
p2 = Plots.contourf(t, x, Y, title = "Truth", c = :balance,)
p3 = Plots.contourf(t, x, svdY, title = "Algorithm", c = :balance)
p4 = Plots.contourf(t, x, Y_hat_diff, title = "Y Hat - Truth", c = :balance)
p5 = Plots.contourf(t, x, Y, title = "Observed", c = :balance)
p6 = Plots.contourf(t, x, svdY .- Y, title = "Algorithm - Truth", c = :balance)
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

########################################################################
#### look at covariance between X and posterior Y = UDV'
########################################################################
#region

Y_hat = reshape(reduce(hcat, [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U,3)]), n, m, :)

Xmat = reshape(X, n, m, :)

cov(Xmat[:,:,1], Y_hat[:,:,1])

heatmap(Xmat[:,:,1])
heatmap(Xmat[:,:,2])

C1 = [cov(Xmat[:,:,1], posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]') for i in axes(posterior.U,3)]
C2 = [cov(Xmat[:,:,2], posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]') for i in axes(posterior.U,3)]

C = [cov(X[:,j], reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', :)) for i in axes(posterior.U,3), j in axes(X, 2)]

l = Plots.@layout [a b; c d]
p1 = Plots.plot(C[:,1])
p2 = Plots.plot(C[:,2])
p3 = Plots.plot(C[:,3])
p4 = Plots.plot(C[:,4])
Plots.plot(p1, p2, p3, p4, layout = l, size = (1400, 600))




#endregion