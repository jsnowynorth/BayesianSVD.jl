
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra


######################################################################
#### Generate Some Data
######################################################################
#region

Random.seed!(1)

m = 100
n = 100
x = range(-50, 50, n)
t = range(0, 100, m)

# D = [20, 10]
# k = 2
ϵ = 5


# ρu = sort(rand(truncated(Exponential(3), lower = 0.5, upper = 4.5), k), rev = true)
# ρv = sort(rand(truncated(Exponential(3), lower = 0.5, upper = 4.5), k), rev = true)

# ρu = sort(rand(Uniform(1, 4), k), rev = true)
# ρv = sort(rand(Uniform(1, 4), k), rev = true)

# ρu = [4.2, 3, 2, 1.5, 1]
# ρv = [4.2, 3, 2, 1.5, 1]



# ρu = [40, 2]
# ρv = [40, 2]


D = [40, 30, 20, 10, 7]
k=5
# ρu = [40, 5, 4, 3, 5]
# ρv = [40, 5, 4, 3, 5]

ρu = [40, 30, 5, 5, 5]
ρv = [40, 30, 5, 5, 5]



# Plots.plot(1:k, ρu)

ΣUvariable = [MaternCorrelation(x, ρ = ρu[i], ν = 3.5, metric = Euclidean()) for i in 1:k]
ΣVvariable = [MaternCorrelation(t, ρ = ρv[i], ν = 3.5, metric = Euclidean()) for i in 1:k]

ΣUstatic = MaternCorrelation(x, ρ = 15, ν = 3.5, metric = Euclidean())
ΣVstatic = MaternCorrelation(t, ρ = 15, ν = 3.5, metric = Euclidean())


# Plots.plot(ΣUvariable[1].K[:,30], label = "K1")
# Plots.plot!(ΣUvariable[2].K[:,30], label = "K2")
# Plots.plot!(MaternCorrelation(x, ρ = 15, ν = 3.5, metric = Euclidean()).K[:,30], label = "K")


Random.seed!(3)
Uvariable, Vvariable, Yvariable, Zvariable = GenerateData(ΣUvariable, ΣVvariable, D, k, ϵ, SNR = true)

Random.seed!(3)
Ustatic, Vstatic, Ystatic, Zstatic = GenerateData(ΣUstatic, ΣVstatic, D, k, ϵ, SNR = true)

Plots.plot(x, Uvariable, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])
Plots.plot(x, Ustatic, xlabel = "Space", ylabel = "Value", label = ["U" * string(i) for i in (1:k)'])

Plots.plot(t, Vvariable, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])
Plots.plot(t, Vstatic, xlabel = "Time", ylabel = "Value", label = ["V" * string(i) for i in (1:k)'])

Plots.contourf(x, t, Yvariable', clim = extrema(Zvariable), c = :balance)
Plots.contourf(x, t, Ystatic', clim = extrema(Zstatic), c = :balance)

Plots.contourf(x, t, Zvariable', clim = extrema(Zvariable), c = :balance)
Plots.contourf(x, t, Zstatic', clim = extrema(Zstatic), c = :balance)

# var(Y)/(var(Z - Y))

#endregion

######################################################################
#### Sample
######################################################################
#region


# initialize model parameters
k = 5
ΩU = MaternCorrelation(x, ρ = 10, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 10, ν = 3.5, metric = Euclidean())

# create data structures
dataVariable = Data(Zvariable, x, t, k)
dataStatic = Data(Zstatic, x, t, k)


nsamp = 10000
nburn = 5000

# run models
# variable model variable data
parsVV = Pars(dataVariable, ΩU, ΩV; ρUMax = [50, 40, 10, 10, 10], ρVMax = [50, 40, 10, 10, 10])
posteriorVV, parsVV = SampleSVD(dataVariable, parsVV; nits = nsamp, burnin = nburn)

# # variable model static data data
# parsVS = Pars(dataStatic, ΩU, ΩV)
# posteriorVS, parsVS = SampleSVD(dataStatic, parsVS; nits = nsamp, burnin = nburn)

# # static model static data
# parsSS = Pars(dataStatic, ΩU, ΩV)
# posteriorSS, parsSS = SampleSVDGrouped(dataStatic, parsSS; nits = nsamp, burnin = nburn)

# static model variable data
parsSV = Pars(dataVariable, ΩU, ΩV)
posteriorSV, parsSV = SampleSVDGrouped(dataVariable, parsSV; nits = nsamp, burnin = nburn)



Plots.plot(posteriorVV.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)

# Plots.plot(posteriorVS.D', label = false, size = (900, 600))
# Plots.hline!([D], label = false)

# Plots.plot(posteriorSS.D', label = false, size = (900, 600))
# Plots.hline!([D], label = false)

Plots.plot(posteriorSV.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)


Plots.plot(posteriorVV.ρU', label = false, size = (900, 600))
# Plots.plot(posteriorVS.ρU', label = false, size = (900, 600))
# Plots.plot(posteriorSS.ρU', label = false, size = (900, 600))
Plots.plot(posteriorSV.ρU', label = false, size = (900, 600))


Plots.plot(posteriorVV.ρV', label = false, size = (900, 600))
# Plots.plot(posteriorVS.ρV', label = false, size = (900, 600))
# Plots.plot(posteriorSS.ρV', label = false, size = (900, 600))
Plots.plot(posteriorSV.ρV', label = false, size = (900, 600))


trsfrmVV = ones(k)
for l in 1:k
    if posteriorCoverage(Uvariable[:,l], posteriorVV.U[:,l,:], 0.95) > posteriorCoverage(-Uvariable[:,l], posteriorVV.U[:,l,:], 0.95)
        continue
    else
        trsfrmVV[l] = -1.0
    end
end

# trsfrmVS = ones(k)
# for l in 1:k
#     if posteriorCoverage(Ustatic[:,l], posteriorVS.U[:,l,:], 0.95) > posteriorCoverage(-Ustatic[:,l], posteriorVS.U[:,l,:], 0.95)
#         continue
#     else
#         trsfrmVS[l] = -1.0
#     end
# end

# trsfrmSS = ones(k)
# for l in 1:k
#     if posteriorCoverage(Ustatic[:,l], posteriorSS.U[:,l,:], 0.95) > posteriorCoverage(-Ustatic[:,l], posteriorSS.U[:,l,:], 0.95)
#         continue
#     else
#         trsfrmSS[l] = -1.0
#     end
# end


trsfrmSV = ones(k)
for l in 1:k
    if posteriorCoverage(Uvariable[:,l], posteriorSV.U[:,l,:], 0.95) > posteriorCoverage(-Uvariable[:,l], posteriorSV.U[:,l,:], 0.95)
        continue
    else
        trsfrmSV[l] = -1.0
    end
end



g = Plots.plot(posteriorVV, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (Uvariable' .* trsfrmVV)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSV, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (Uvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2)

# g = Plots.plot(posteriorVS, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
# g = Plots.plot!(x, (Ustatic' .* trsfrmVS)', label = false, color = "black", linewidth = 2)

# g = Plots.plot(posteriorSS, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
# g = Plots.plot!(x, (Ustatic' .* trsfrmSS)', label = false, color = "black", linewidth = 2)


g = Plots.plot(posteriorVV, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vvariable' .* trsfrmVV)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSV, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2)

# g = Plots.plot(posteriorVS, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
# g = Plots.plot!(t, (Vstatic' .* trsfrmVS)', label = false, color = "black", linewidth = 2)

# g = Plots.plot(posteriorSS, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
# g = Plots.plot!(t, (Vstatic' .* trsfrmSS)', label = false, color = "black", linewidth = 2)



posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.U, 0.95) # 0.882
# posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmVS)'), posteriorVS.U, 0.95) # 0.916
# posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmSS)'), posteriorSS.U, 0.95) # 0.944
posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.U, 0.95) # 0.748

posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.V, 0.95) # 0.934
# posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmVS)'), posteriorVS.V, 0.95) # 0.856
# posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmSS)'), posteriorSS.V, 0.95) # 0.87
posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.V, 0.95) # 0.782



dx = x[2] - x[1]
mean((posteriorVV.U_upper .- posteriorVV.U_lower) .* dx, dims = 1)
# mean((posteriorSV.U_upper .- posteriorSV.U_lower) .* dx, dims = 1)
# mean((posteriorSS.U_upper .- posteriorSS.U_lower) .* dx, dims = 1)
mean((posteriorSV.U_upper .- posteriorSV.U_lower) .* dx, dims = 1)


mean((posteriorVV.V_upper .- posteriorVV.V_lower) .* dx, dims = 1)
# mean((posteriorSV.V_upper .- posteriorSV.V_lower) .* dx, dims = 1)
# mean((posteriorSS.V_upper .- posteriorSS.V_lower) .* dx, dims = 1)
mean((posteriorSV.V_upper .- posteriorSV.V_lower) .* dx, dims = 1)


Plots.plot((posteriorVV.U_upper .- posteriorVV.U_lower) .* dx)
Plots.plot((posteriorSV.U_upper .- posteriorSV.U_lower) .* dx)

Plots.plot((posteriorVV.V_upper .- posteriorVV.V_lower) .* dx)
Plots.plot((posteriorSV.V_upper .- posteriorSV.V_lower) .* dx)



mean([sqrt.(mean((posteriorVV.U[:,:,i] .- (Uvariable[:,1:k]' .* trsfrmVV)') .^2, dims = 1)) for i in axes(posteriorVV.U, 3)])
mean([sqrt.(mean((posteriorSV.U[:,:,i] .- (Uvariable[:,1:k]' .* trsfrmSV)') .^2, dims = 1)) for i in axes(posteriorSV.U, 3)])
mean([sqrt.(mean((posteriorSS.U[:,:,i] .- (Ustatic[:,1:k]' .* trsfrmSS)') .^2, dims = 1)) for i in axes(posteriorSS.U, 3)])
mean([sqrt.(mean((posteriorVS.U[:,:,i] .- (Ustatic[:,1:k]' .* trsfrmVS)') .^2, dims = 1)) for i in axes(posteriorVS.U, 3)])


mean([sqrt.(mean((posteriorVV.V[:,:,i] .- (Vvariable[:,1:k]' .* trsfrmVV)') .^2, dims = 1)) for i in axes(posteriorVV.V, 3)])
mean([sqrt.(mean((posteriorSV.V[:,:,i] .- (Vvariable[:,1:k]' .* trsfrmSV)') .^2, dims = 1)) for i in axes(posteriorSV.V, 3)])
mean([sqrt.(mean((posteriorSS.V[:,:,i] .- (Vstatic[:,1:k]' .* trsfrmSS)') .^2, dims = 1)) for i in axes(posteriorSS.V, 3)])
mean([sqrt.(mean((posteriorVS.V[:,:,i] .- (Vstatic[:,1:k]' .* trsfrmVS)') .^2, dims = 1)) for i in axes(posteriorVS.V, 3)])


Plots.plot(x, posteriorVV.U_hat .- Uvariable)
Plots.plot(x, posteriorSV.U_hat .- Uvariable)
Plots.plot(x, posteriorVS.U_hat .- Ustatic)
Plots.plot(x, posteriorSS.U_hat .- Ustatic)





g = Plots.plot(posteriorSV, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot(x, posteriorSV.U_hat[:,5], label = false, color = "black", linewidth = 2)
g = Plots.plot!(x, Uvariable[:,5], label = false, color = "black", linewidth = 2)


p1 = Plots.plot(x, (Uvariable' .* trsfrmVV)', c = :black, markersize = 3, markerstrokewidth = 0, label = false)
for i in 1:100:5000
    p1 = Plots.plot!(x, posteriorVV.U[:,:,i], label = false, c = :gray, alpha = 0.5)
end
# p1

# p1 = Plots.plot(x, (Uvariable' .* trsfrmSV)', c = :black, markersize = 3, markerstrokewidth = 0, label = false)
for i in 1:100:5000
    p1 = Plots.plot!(x, posteriorSV.U[:,:,i], label = false, c = :blue, alpha = 0.5)
end
p1 = Plots.plot!(x, (Uvariable' .* trsfrmVV)', c = :black, markersize = 3, markerstrokewidth = 0, label = false)
p1


p1 = Plots.plot(x, (Uvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2, size  = (1200, 600))
p1 = Plots.plot!(x, posteriorVV.U_upper, label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorVV.U_lower, label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.U_upper, label = false, color = "red", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.U_lower, label = false, color = "red", linewidth = 1)


p1 = Plots.plot(x, (Uvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2, size  = (1200, 600))
p1 = Plots.plot!(x, maximum(posteriorVV.U, dims = 3)[:,:,1], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, minimum(posteriorVV.U, dims = 3)[:,:,1], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, maximum(posteriorSV.U, dims = 3)[:,:,1], label = false, color = "red", linewidth = 1)
p1 = Plots.plot!(x, minimum(posteriorSV.U, dims = 3)[:,:,1], label = false, color = "red", linewidth = 1)



b = 5
p1 = Plots.plot(x, (Uvariable' .* trsfrmSV)'[:,b], label = false, color = "black", linewidth = 2, size  = (1200, 600))
p1 = Plots.plot!(x, posteriorVV.U_upper[:,b], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorVV.U_lower[:,b], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.U_upper[:,b], label = false, color = "red", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.U_lower[:,b], label = false, color = "red", linewidth = 1)


b = 5
p1 = Plots.plot(x, (Vvariable' .* trsfrmSV)'[:,b], label = false, color = "black", linewidth = 2, size  = (1200, 600))
p1 = Plots.plot!(x, posteriorVV.V_upper[:,b], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorVV.V_lower[:,b], label = false, color = "blue", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.V_upper[:,b], label = false, color = "red", linewidth = 1)
p1 = Plots.plot!(x, posteriorSV.V_lower[:,b], label = false, color = "red", linewidth = 1)









# posteriorCoverage(ρu, posteriorVV.ρU, 0.95)
# posteriorCoverage(ρv, posteriorVV.ρV, 0.95)

# posteriorCoverage(fill(3.0, k), posteriorVS.ρU, 0.95)
# posteriorCoverage(fill(3.0, k), posteriorVS.ρV, 0.95)



Plots.plot(posteriorVV.σU', title = "σU", size = (1000, 600))
Plots.plot(posteriorVV.σV', title = "σV", size = (1000, 600))
Plots.plot(posteriorVV.σ, title = "σ")
Plots.plot(posteriorVV.U[50,:,:]', title = "U")
Plots.hline!(Uvariable[50,:] .* trsfrmVV, c = :black, label = false)
Plots.plot(posteriorVV.V[45,:,:]', title = "V")
Plots.hline!(Vvariable[45,:] .* trsfrmVV, c = :black, label = false)



#endregion



# data = Data(Zstatic, x, t, k)
# pars = Pars(data, ΩU, ΩV)


# propsd = pars.propSU[1]
# ρprop = rand(truncated(Normal(pars.ΩU[1].ρ, propsd), 0, pars.ρUMax[1]))
# Cprop = copy(pars.ΩU[1])
# Cprop.ρ = ρprop
# Cprop = updateCorrelation(Cprop)
# ρprior = Uniform(0, pars.ρUMax[1])


# rat = 0
# for i in 1:data.k
    
#     Σprop =  pars.σU[i] * Hermitian(pars.NU[:,:,i]' * Cprop.K * pars.NU[:,:,i])
#     Σcurr = pars.σU[i] * pars.NΩUN[:,:,i]

#     rat += ρLike(pars.UZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(truncated(Normal(ρprop, propsd), 0, Inf), pars.ΩU[i].ρ) - 
#             (ρLike(pars.UZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩU[i].ρ) + logpdf(truncated(Normal(pars.ΩU[i].ρ, propsd), 0, Inf), ρprop))

# end

# if log(rand(Uniform())) < rat
#     for i in 1:data.k
#         pars.ΩU[i] = Cprop
#         pars.Uaccept[i] = 1
#     end
# end








