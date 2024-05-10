
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 20-April-2023
#### Description: Variable length simulation study code
########################################################################

# find . -name 'julia-*' -delete

# println(Vector(range(30, 3000, step = 30)) ./ 32)
# nodes, nReplicates, ntasks
# hcat(Vector(range(6, 600, step = 6)) ./ 32, 1:100, Vector(1:100) .* 6)


using ClusterManagers
using Distributed
# addprocs(4) # change in Perlmutter not on local computer!!!!
# nprocs()

# addprocs(SlurmManager(600)) # change in Perlmutter not on local computer!!!!
print(nprocs())
myid()

######################################################################
#### Sample
######################################################################

# simulationParameters[:,1] replicate ID
# simulationParameters[:,2] SNR value
# simulationParameters[:,3] seed value


@everywhere using BayesianSVD
@everywhere using Distances, Random, Distributions, LinearAlgebra
@everywhere using DataFrames, DataFramesMeta, CSV


@time @sync @distributed for i in 1:(nprocs()-1)

    nReplicates = 100 # change in Perlmutter not on local computer!!!!
    SNRValues = [0.1, 0.5, 1, 2, 5, 10]

    replicateVector = Vector(range(1, nReplicates))
    simulationParameters = reduce(hcat, reshape([[i, k] for i = replicateVector, k = SNRValues], :))'
    nSimulations = size(simulationParameters, 1)
    simulationParameters = hcat(simulationParameters, Vector(1:nSimulations)) # replicate ID, SNR value, seed value


    # Random.seed!(Int(i))
    Random.seed!(Int(simulationParameters[i, 3]))
    # simulated model parameters
    m = 100
    n = 100
    x = range(-5, 5, n)
    t = range(0, 10, m)
    k = 4

    D = [40, 30, 20, 10] # sqrt of eigenvalues
    # ϵ = 0.5 # noise
    ϵ = simulationParameters[i, 2] # noise

    ρu = [3.5, 1, 0.5, 0.25]
    ρv = [3.5, 1, 0.5, 0.25]

    
    ΣU = [MaternCorrelation(x, ρ = ρu[i], ν = 3.5, metric = Euclidean()) for i in 1:k]
    ΣV = [MaternCorrelation(t, ρ = ρv[i], ν = 3.5, metric = Euclidean()) for i in 1:k]


    U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)

    

    # initialize model parameters
    ΩU = MaternCorrelation(x, ρ = 1, ν = 3.5, metric = Euclidean())
    ΩV = MaternCorrelation(t, ρ = 1, ν = 3.5, metric = Euclidean())

    # create data structures
    data = Data(Z, x, t, k)

    
    nits = 10000
    nburn = 5000
    # nits = 200
    # nburn = 100
    # run models
    # variable model variable data
    parsVar = Pars(data, ΩU, ΩV)
    posteriorVar, parsVar = SampleSVD(data, parsVar; nits = nits, burnin = nburn)

    # static model variable data
    parsGroup = Pars(data, ΩU, ΩV)
    posteriorGroup, parsGroup = SampleSVDGrouped(data, parsGroup; nits = nits, burnin = nburn)


    # align basis functions
    trsfrmVar = ones(k)
    for l in 1:k
        if posteriorCoverage(U[:,l], posteriorVar.U[:,l,:], 0.95) > posteriorCoverage(-U[:,l], posteriorVar.U[:,l,:], 0.95)
            continue
        else
            trsfrmVar[l] = -1.0
        end
    end

    trsfrmGroup = ones(k)
    for l in 1:k
        if posteriorCoverage(U[:,l], posteriorGroup.U[:,l,:], 0.95) > posteriorCoverage(-U[:,l], posteriorGroup.U[:,l,:], 0.95)
            continue
        else
            trsfrmGroup[l] = -1.0
        end
    end


    cover_U_Var = posteriorCoverage(Matrix((U[:,1:k]' .* trsfrmVar)'), posteriorVar.U, 0.95)
    cover_U_Group = posteriorCoverage(Matrix((U[:,1:k]' .* trsfrmGroup)'), posteriorGroup.U, 0.95)

    cover_V_Var = posteriorCoverage(Matrix((V[:,1:k]' .* trsfrmVar)'), posteriorVar.V, 0.95)
    cover_V_Group = posteriorCoverage(Matrix((V[:,1:k]' .* trsfrmGroup)'), posteriorGroup.V, 0.95)

    Y_Var = [posteriorVar.U[:,:,j] * diagm(posteriorVar.D[:,j]) * posteriorVar.V[:,:,j]' for j in axes(posteriorVar.U, 3)]
    Y_Group = [posteriorGroup.U[:,:,j] * diagm(posteriorGroup.D[:,j]) * posteriorGroup.V[:,:,j]' for j in axes(posteriorGroup.U, 3)]

    cover_Y_Var = posteriorCoverage(Y, Y_Var, 0.95)
    cover_Y_Group = posteriorCoverage(Y, Y_Group, 0.95)


    RMSE_U_Var = sqrt.(mean(((U' .* trsfrmVar)' .- posteriorVar.U_hat) .^2, dims = 1))
    RMSE_U_Group = sqrt.(mean(((U' .* trsfrmGroup)' .- posteriorGroup.U_hat) .^2, dims = 1))

    RMSE_V_Var = sqrt.(mean(((V' .* trsfrmVar)' .- posteriorVar.V_hat) .^2, dims = 1))
    RMSE_V_Group = sqrt.(mean(((V' .* trsfrmGroup)' .- posteriorGroup.V_hat) .^2, dims = 1))

    RMSE_Y_Var = sqrt(mean((mean(Y_Var) - Y) .^2))
    RMSE_Y_Group = sqrt(mean((mean(Y_Group) - Y) .^2))
    
    
    dfRMSE = DataFrame(replicate = fill(simulationParameters[i, 1], k),
                        SNR = fill(simulationParameters[i, 2], k),
                        seed = fill(simulationParameters[i, 3], k),
                        basis = 1:4, 
                        RMSE_U_Var = RMSE_U_Var[1,:], 
                        RMSE_U_Group = RMSE_U_Group[1,:], 
                        RMSE_V_Var = RMSE_V_Var[1,:], 
                        RMSE_V_Group = RMSE_V_Group[1,:])

    df = DataFrame(replicate = simulationParameters[i, 1],
                    SNR = simulationParameters[i, 2],
                    seed = simulationParameters[i, 3],
                    cover_U_Var = cover_U_Var,
                    cover_U_Group = cover_U_Group,
                    cover_V_Var = cover_V_Var,
                    cover_V_Group = cover_V_Group,
                    cover_Y_Var = cover_Y_Var,
                    cover_Y_Group = cover_Y_Group,
                    RMSE_Y_Var = RMSE_Y_Var,
                    RMSE_Y_Group = RMSE_Y_Group)
    #
    
    # CSV.write("/Users/JSNorth/Desktop/VariableSimulation/run" * string(i) * ".csv", df)
    # CSV.write("/Users/JSNorth/Desktop/VariableSimulation/runRMSE" * string(i) * ".csv", dfRMSE)
    CSV.write("../results/run" * string(i) * ".csv", df)
    CSV.write("../results/runRMSE" * string(i) * ".csv", dfRMSE)
end
