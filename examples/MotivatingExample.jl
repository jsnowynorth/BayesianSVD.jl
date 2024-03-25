
using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

using CairoMakie
using DataFrames, DataFramesMeta, Chain, CSV
using LaTeXStrings


######################################################################
#### Generate Some Data
######################################################################
#region

Random.seed!(3659)

m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

D = [40, 30, 20, 10, 5]
k = 5
ϵ = 0.1

ρu = sort(rand(Uniform(1, 4), k), rev = true)
ρv = sort(rand(Uniform(1, 4), k), rev = true)


ΣUvariable = [MaternCorrelation(x, ρ = ρu[i], ν = 3.5, metric = Euclidean()) for i in 1:k]
ΣVvariable = [MaternCorrelation(t, ρ = ρv[i], ν = 3.5, metric = Euclidean()) for i in 1:k]

ΣUstatic = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣVstatic = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())

Random.seed!(3)
Uvariable, Vvariable, Yvariable, Zvariable = GenerateData(ΣUvariable, ΣVvariable, D, k, 2, SNR = true)

Random.seed!(3)
Ustatic, Vstatic, Ystatic, Zstatic = GenerateData(ΣUstatic, ΣVstatic, D, k, 2, SNR = true)

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
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())

# create data structures
dataVariable = Data(Zvariable, x, t, k)
dataStatic = Data(Zstatic, x, t, k)


# run models
# variable model variable data
parsVV = Pars(dataVariable, ΩU, ΩV)
posteriorVV, parsVV = SampleSVD(dataVariable, parsVV; nits = 10000, burnin = 5000)

# variable model static data data
parsVS = Pars(dataVariable, ΩU, ΩV)
posteriorVS, parsVS = SampleSVD(dataStatic, parsVS; nits = 10000, burnin = 5000)

# static model static data
parsSS = Pars(dataStatic, ΩU, ΩV)
posteriorSS, parsSS = SampleSVDstatic(dataStatic, parsSS; nits = 10000, burnin = 5000)

# static model variable data
parsSV = Pars(dataStatic, ΩU, ΩV)
posteriorSV, parsSV = SampleSVDstatic(dataVariable, parsSV; nits = 10000, burnin = 5000)



Plots.plot(posteriorVV.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)

Plots.plot(posteriorVS.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)

Plots.plot(posteriorSS.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)

Plots.plot(posteriorSV.D', label = false, size = (900, 600))
Plots.hline!([D], label = false)


trsfrmVV = ones(k)
for l in 1:k
    if posteriorCoverage(Uvariable[:,l], posteriorVV.U[:,l,:], 0.95) > posteriorCoverage(-Uvariable[:,l], posteriorVV.U[:,l,:], 0.95)
        continue
    else
        trsfrmVV[l] = -1.0
    end
end

trsfrmVS = ones(k)
for l in 1:k
    if posteriorCoverage(Ustatic[:,l], posteriorVS.U[:,l,:], 0.95) > posteriorCoverage(-Ustatic[:,l], posteriorVS.U[:,l,:], 0.95)
        continue
    else
        trsfrmVS[l] = -1.0
    end
end

trsfrmSS = ones(k)
for l in 1:k
    if posteriorCoverage(Ustatic[:,l], posteriorSS.U[:,l,:], 0.95) > posteriorCoverage(-Ustatic[:,l], posteriorSS.U[:,l,:], 0.95)
        continue
    else
        trsfrmSS[l] = -1.0
    end
end


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

g = Plots.plot(posteriorVS, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (Ustatic' .* trsfrmVS)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSS, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (Ustatic' .* trsfrmSS)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSV, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(x, (Uvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2)



g = Plots.plot(posteriorVV, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vvariable' .* trsfrmVV)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorVS, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vstatic' .* trsfrmVS)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSS, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vstatic' .* trsfrmSS)', label = false, color = "black", linewidth = 2)

g = Plots.plot(posteriorSV, t, size = (800, 400), basis = 'V', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = "U")
g = Plots.plot!(t, (Vvariable' .* trsfrmSV)', label = false, color = "black", linewidth = 2)


posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.U, 0.95) # 0.882
posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmVS)'), posteriorVS.U, 0.95) # 0.916
posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmSS)'), posteriorSS.U, 0.95) # 0.944
posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.U, 0.95) # 0.748

posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.V, 0.95) # 0.934
posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmVS)'), posteriorVS.V, 0.95) # 0.856
posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmSS)'), posteriorSS.V, 0.95) # 0.87
posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.V, 0.95) # 0.782



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

