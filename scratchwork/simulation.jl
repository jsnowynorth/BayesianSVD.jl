


# using BayesianSVD
# using Distances, Plots, Random, Distributions, LinearAlgebra

using Distributed
addprocs(6)

nprocs()


######################################################################
#### Sample
######################################################################

nreps = 2

ρsim = [1, 1.5, 2, 2.5, 3, 3.5]
ϵsim = [0.5, 1, 5]
rep = Vector(range(1, nreps))

simpars = reduce(vcat, [(i, j, k) for i = rep, j = ρsim, k = ϵsim])

@everywhere using BayesianSVD
@everywhere using Distances, Plots, Random, Distributions, LinearAlgebra

@time begin
    @sync @distributed (append!) for i in 1:6

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

        k = 5
        ΩU = MaternKernel(x, ρ = ρsim[i], ν = 3.5, metric = Euclidean())
        ΩV = MaternKernel(t, ρ = ρsim[i], ν = 3.5, metric = Euclidean())
        data = Data(Z, ΩU, ΩV, k)
        pars = Pars(data)

        posterior, pars = SampleSVD(data, pars, nits = 1000, burnin = 500, show_progress = false)
        # println(sqrt(mean(posterior.U_hat - U).^2), ", Process ID - ", myid(), ", ρ = ", ρsim[i])
        res = sqrt(mean(posterior.U_hat - U).^2)
        [res]

    end
end

@time begin
    for i in 1:6

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
    
        k = 5
        ΩU = MaternKernel(x, ρ = ρsim[i], ν = 3.5, metric = Euclidean())
        ΩV = MaternKernel(t, ρ = ρsim[i], ν = 3.5, metric = Euclidean())
        data = Data(Z, ΩU, ΩV, k)
        pars = Pars(data)
    
        posterior, pars = SampleSVD(data, pars, nits = 1000, burnin = 500, show_progress = false)
        # println(sqrt(mean(posterior.U_hat - U).^2), ", Process ID - ", myid(), ", ρ = ", ρsim[i])
        res = sqrt(mean(posterior.U_hat - U).^2)
    
    end
end


nsims = length(simpars)

result = @sync @distributed (append!) for i in 1:nsims

    m = 100
    n = 100
    x = range(-5, 5, n)
    t = range(0, 10, m)

    ΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())


    D = [40, 20, 10, 5, 2] # sqrt of eigenvalues
    k = 5 # number of basis functions 
    ϵ = simpars[i][3] # noise

    # Random.seed!(2)
    U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)

    k = 5
    ΩU = MaternKernel(x, ρ = simpars[i][2], ν = 3.5, metric = Euclidean())
    ΩV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())
    data = Data(Z, ΩU, ΩV, k)
    pars = Pars(data)

    posterior, pars = SampleSVD(data, pars, nits = 1000, burnin = 500, show_progress = false)
    # println(sqrt(mean(posterior.U_hat - U).^2), ", Process ID - ", myid(), ", ρ = ", ρsim[i])
    res = sqrt(mean(posterior.U_hat - U).^2)
    [res]

end


using DataFrames, DataFramesMeta

df = DataFrame(rep = ones(length(simpars)), rho = ones(length(simpars)), epsilon = ones(length(simpars)), RMSE = ones(length(simpars)))

for i in axes(simpars,1)
    df[i,:rep] = simpars[i][1]
    df[i,:rho] = simpars[i][2]
    df[i,:epsilon] = simpars[i][3]
    df[i,:RMSE] = result[i]
end

df