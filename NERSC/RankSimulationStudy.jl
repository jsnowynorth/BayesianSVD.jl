
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 20-April-2023
#### Description: Rank simulation study code
########################################################################

# find . -name 'julia-*' -delete

# println(Vector(range(30, 3000, step = 30)) ./ 32)
# nodes, nReplicates, ntasks
# hcat(Vector(range(30, 3000, step = 30)) ./ 32, 1:100, Vector(1:100) .* 30)


using ClusterManagers
using Distributed
# addprocs(6) # change in Perlmutter not on local computer!!!!
# nprocs()

# addprocs(SlurmManager(3000)) # change in Perlmutter not on local computer!!!!
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


@sync @distributed for i in 1:(nprocs()-1)


    nReplicates = 100 # change in Perlmutter not on local computer!!!!
    kValues = [3, 4, 5, 6, 7]
    SNRValues = [0.1, 0.5, 1, 2, 5, 10]

    replicateVector = Vector(range(1, nReplicates))
    simulationParameters = reduce(hcat, reshape([[i, j, k] for i = replicateVector, j = kValues, k = SNRValues], :))'
    nSimulations = size(simulationParameters, 1)
    simulationParameters = hcat(simulationParameters, Vector(1:nSimulations)) # replicate ID, k value, SNR value, seed value


    Random.seed!(Int(simulationParameters[i, 4]))
    # simulated model parameters
    m = 100
    n = 100
    x = range(-5, 5, n)
    t = range(0, 10, m)

    ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())

    D = [40, 20, 10, 5, 2] # sqrt of eigenvalues
    # k = 5 # number of basis functions 
    ϵ = simulationParameters[i, 3] # noise

    # simulate data
    U, V, Y, Z = GenerateData(ΣU, ΣV, D, 5, ϵ, SNR = true)

    # padded data for calculating RMSE and coverage
    Utmp = hcat(U, zeros(n, 2))
    Vtmp = hcat(V, zeros(m, 2))
    Dtmp = vcat(D, 0, 0)

    # set up model
    kChoice = Int(simulationParameters[i, 2])
    ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
    data = Data(Z, x, t, kChoice)
    pars = Pars(data, ΩU, ΩV)

    # run model
    posterior, pars = SampleSVD(data, pars, nits = 10000, burnin = 5000, show_progress = false)


    Yest = [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' for i in axes(posterior.U, 3)]

    trsfrm = ones(kChoice)
    for l in 1:kChoice
        if posteriorCoverage(Utmp[:,l], posterior.U[:,l,:], 0.95) > posteriorCoverage(-Utmp[:,l], posterior.U[:,l,:], 0.95) # make sure the estimated and true have the same sign
            continue
        else
            trsfrm[l] = -1.0
        end
    end


    RMSEData = sqrt(mean(reduce(vcat, [((Y) .- Yest[i]).^2 for i in axes(Yest,1)])))
    RMSEU = sqrt(mean(((Utmp[:,1:data.k]' .* trsfrm)' .- posterior.U_hat).^2))
    RMSEV = sqrt(mean(((Vtmp[:,1:data.k]' .* trsfrm)' .- posterior.V_hat).^2))
    coverData = posteriorCoverage(Y, Yest, 0.95)
    coverU = posteriorCoverage(Matrix((Utmp[:,1:data.k]' .* trsfrm)'), posterior.U, 0.95)
    coverV = posteriorCoverage(Matrix((Vtmp[:,1:data.k]' .* trsfrm)'), posterior.V, 0.95)

    svdY = svd(Z)
    RMSEDataA = sqrt(mean((Y .- svdY.U[:,1:data.k] * diagm(svdY.S[1:data.k]) * svdY.V[:,1:data.k]').^2))
    RMSEUA = sqrt(mean(((Utmp[:,1:data.k]' .* trsfrm)' .- svdY.U[:,1:data.k]).^2))
    RMSEVA = sqrt(mean(((Vtmp[:,1:data.k]' .* trsfrm)' .- svdY.V[:,1:data.k]).^2))


    df = DataFrame(replicate = simulationParameters[i, 1],
                    k = simulationParameters[i, 2],
                    SNR = simulationParameters[i, 3],
                    seed = simulationParameters[i, 4],
                    RMSEData = RMSEData,
                    RMSEU = RMSEU,
                    RMSEV = RMSEV,
                    coverData = coverData,
                    coverU = coverU,
                    coverV = coverV,
                    RMSEDataA = RMSEDataA,
                    RMSEUA = RMSEUA,
                    RMSEVA = RMSEVA)
    #
    
    CSV.write("../results/simulationResults/run" * string(i) * ".csv", df)

end
