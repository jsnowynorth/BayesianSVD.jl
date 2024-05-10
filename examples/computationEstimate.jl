########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 10-May-2024
#### Description: Code used for Fig S.2
########################################################################


using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra

using CairoMakie
using DataFrames, DataFramesMeta, Chain, CSV
using LaTeXStrings

# using BenchmarkTools
# using TimerOutputs


########################################################################
#### function to update all parameters
########################################################################
#region

function updateAll(data, pars)
    BayesianSVD.update_D(data, pars)
    BayesianSVD.update_U(data, pars)
    BayesianSVD.update_V(data, pars)
    BayesianSVD.update_σ(data, pars)
    BayesianSVD.update_σU(data, pars)
    BayesianSVD.update_σV(data, pars)
    BayesianSVD.update_ρU(data, pars, pars.ΩU[1])
    BayesianSVD.update_ρV(data, pars, pars.ΩV[1])
end

#endregion

######################################################################
#### Search parameters
######################################################################
#region

nValues = [20, 50, 100, 500, 1000]
mValues = [20, 50, 100, 500, 1000]
simulationParameters = reduce(hcat, reshape([[i, j] for i = nValues, j = mValues], :))'

df = DataFrame(n = simulationParameters[:,1], m = simulationParameters[:,2], t = zeros(size(simulationParameters, 1)))

D = [40, 30, 20, 10, 5]
k = 5
ϵ = 2
stepSize = 0.2


for i in axes(simulationParameters, 1)

    x = range(stepSize, stepSize*simulationParameters[i,1], simulationParameters[i,1])
    t = range(stepSize, stepSize*simulationParameters[i,2], simulationParameters[i,2])

    ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())

    Random.seed!(3)
    U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)

    data = Data(Z, x, t, k)
    pars = Pars(data, ΣU, ΣV)

    # b = @benchmark updateAll(data, pars) samples=100 seconds=30
    # df[i,:t] = median(b.times)

    df[i,:t] = median([@elapsed updateAll(data, pars) for j in 1:10])

    println(i)

end


df
CSV.write("/Users/JSNorth/Desktop/comsim.csv", df)

