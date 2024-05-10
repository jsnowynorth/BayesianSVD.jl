
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 20-April-2023
#### Description: Simulation study code
########################################################################

# find . -name 'julia-*' -delete

# println(Vector(range(30, 3000, step = 30)) ./ 32)
# nodes, nReplicates, ntasks
# hcat(Vector(range(30, 3000, step = 30)) ./ 32, 1:100, Vector(1:100) .* 30)


using ClusterManagers
using Distributed
addprocs(4) # change in Perlmutter not on local computer!!!!
# nprocs()

# addprocs(SlurmManager(100)) # change in Perlmutter not on local computer!!!!
print(nprocs())
myid()

######################################################################
#### Sample
######################################################################

# simulationParameters[:,1] replicate ID
# simulationParameters[:,2] k value
# simulationParameters[:,3] SNR value
# simulationParameters[:,4] seed value


@everywhere using BayesianSVD
@everywhere using Distances, Random, Distributions, LinearAlgebra
@everywhere using DataFrames, DataFramesMeta, CSV


@time @sync @distributed for i in 1:(nprocs()-1)


    Random.seed!(Int(i))
    # simulated model parameters
    m = 100
    n = 100
    x = range(-5, 5, n)
    t = range(0, 10, m)
    k = 5

    D = [40, 20, 10, 5, 2] # sqrt of eigenvalues
    ϵ = 2 # noise

    ρu = sort(rand(Uniform(0.5, 4.5), k), rev = true)
    ρv = sort(rand(Uniform(0.5, 4.5), k), rev = true)

    
    ΣUvariable = [MaternCorrelation(x, ρ = ρu[i], ν = 3.5, metric = Euclidean()) for i in 1:k]
    ΣVvariable = [MaternCorrelation(t, ρ = ρv[i], ν = 3.5, metric = Euclidean()) for i in 1:k]

    ΣUstatic = MaternCorrelation(x, ρ = 0.5, ν = 3.5, metric = Euclidean())
    ΣVstatic = MaternCorrelation(t, ρ = 0.5, ν = 3.5, metric = Euclidean())

    Uvariable, Vvariable, Yvariable, Zvariable = GenerateData(ΣUvariable, ΣVvariable, D, k, ϵ, SNR = true)

    Ustatic, Vstatic, Ystatic, Zstatic = GenerateData(ΣUstatic, ΣVstatic, D, k, ϵ, SNR = true)

    
    # initialize model parameters
    k = 5
    ΩU = MaternCorrelation(x, ρ = 0.5, ν = 3.5, metric = Euclidean())
    ΩV = MaternCorrelation(t, ρ = 0.5, ν = 3.5, metric = Euclidean())

    # create data structures
    dataVariable = Data(Zvariable, x, t, k)
    dataStatic = Data(Zstatic, x, t, k)

    
    # nits = 10000
    # nburn = 5000
    nits = 2000
    nburn = 1000
    # run models
    # variable model variable data
    parsVV = Pars(dataVariable, ΩU, ΩV)
    posteriorVV, parsVV = SampleSVD(dataVariable, parsVV; nits = nits, burnin = nburn)

    # variable model static data data
    parsVS = Pars(dataStatic, ΩU, ΩV)
    posteriorVS, parsVS = SampleSVD(dataStatic, parsVS; nits = nits, burnin = nburn)

    # static model static data
    parsSS = Pars(dataStatic, ΩU, ΩV)
    posteriorSS, parsSS = SampleSVDstatic(dataStatic, parsSS; nits = nits, burnin = nburn)

    # static model variable data
    parsSV = Pars(dataVariable, ΩU, ΩV)
    posteriorSV, parsSV = SampleSVDstatic(dataVariable, parsSV; nits = nits, burnin = nburn)


    # align basis functions
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


    U_VV = posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.U, 0.95)
    U_VS = posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmVS)'), posteriorVS.U, 0.95)
    U_SS = posteriorCoverage(Matrix((Ustatic[:,1:k]' .* trsfrmSS)'), posteriorSS.U, 0.95)
    U_SV = posteriorCoverage(Matrix((Uvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.U, 0.95)

    V_VV = posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmVV)'), posteriorVV.V, 0.95)
    V_VS = posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmVS)'), posteriorVS.V, 0.95)
    V_SS = posteriorCoverage(Matrix((Vstatic[:,1:k]' .* trsfrmSS)'), posteriorSS.V, 0.95)
    V_SV = posteriorCoverage(Matrix((Vvariable[:,1:k]' .* trsfrmSV)'), posteriorSV.V, 0.95)


    Y_VV = [posteriorVV.U[:,:,j] * diagm(posteriorVV.D[:,j]) * posteriorVV.V[:,:,j]' for j in axes(posteriorVV.U, 3)]
    Y_VS = [posteriorVS.U[:,:,j] * diagm(posteriorVS.D[:,j]) * posteriorVS.V[:,:,j]' for j in axes(posteriorVS.U, 3)]
    Y_SS = [posteriorSS.U[:,:,j] * diagm(posteriorSS.D[:,j]) * posteriorSS.V[:,:,j]' for j in axes(posteriorSS.U, 3)]
    Y_SV = [posteriorSV.U[:,:,j] * diagm(posteriorSV.D[:,j]) * posteriorSV.V[:,:,j]' for j in axes(posteriorSV.U, 3)]

    RMSE_VV = sqrt(mean(reduce(vcat, [(Y_VV[j] .- Yvariable) .^2 for j in axes(Y_VV, 1)])))
    RMSE_VS = sqrt(mean(reduce(vcat, [(Y_VS[j] .- Ystatic) .^2 for j in axes(Y_VV, 1)])))
    RMSE_SS = sqrt(mean(reduce(vcat, [(Y_SS[j] .- Ystatic) .^2 for j in axes(Y_VV, 1)])))
    RMSE_SV = sqrt(mean(reduce(vcat, [(Y_SV[j] .- Yvariable) .^2 for j in axes(Y_VV, 1)])))

    coverData_VV = posteriorCoverage(Yvariable, Y_VV, 0.95)
    coverData_VS = posteriorCoverage(Ystatic, Y_VS, 0.95)
    coverData_SS = posteriorCoverage(Ystatic, Y_SS, 0.95)
    coverData_SV = posteriorCoverage(Yvariable, Y_SV, 0.95)

    

    df = DataFrame(replicate = i,
                    U_VV = U_VV,
                    U_VS = U_VS,
                    U_SS = U_SS,
                    U_SV = U_SV,
                    V_VV = V_VV,
                    V_VS = V_VS,
                    V_SS = V_SS,
                    V_SV = V_SV,
                    coverData_VV = coverData_VV,
                    coverData_VS = coverData_VS,
                    coverData_SS = coverData_SS,
                    coverData_SV = coverData_SV)
    #
    
    CSV.write("/Users/JSNorth/Desktop/VariableSimulation/run" * string(i) * ".csv", df)
    # CSV.write("../results/simulationResults/run" * string(i) * ".csv", df)
end
