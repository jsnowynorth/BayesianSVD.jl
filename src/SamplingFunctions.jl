######################################################################
#### Sampling Functions
######################################################################


#### Update U
function update_U(data::Data, pars::Pars)

    U = pars.U
    UZ = pars.UZ
    # Σ = pars.σ*diagm(ones(data.n))
    # Σinv = inv(Σ)
    NONinv = data.ΩU.Kinv

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(U[:,inds]')
        E = data.Y - U[:,inds] * diagm(pars.D[inds]) * pars.V[:,inds]'
        
        m = pars.D[i] * (1/pars.σ) * N' * E * pars.V[:,i]
        S = pars.D[i]^2 * (1/pars.σU[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.n - data.k + 1))

        G = cholesky(Hermitian(S))
        b = rand(Normal(0, 1), data.n - data.k + 1)
        UZ[:,i] = (G.U \ ((G.L \ m) + b))
        UZ[:,i] = UZ[:,i] / norm(UZ[:,i])
        U[:,i] = N * UZ[:,i]
    end

    pars.U = U
    pars.UZ = UZ

    return pars
    
end


#### Update V
function update_V(data::Data, pars::Pars)

    V = pars.V
    VZ = pars.VZ
    # Σ = pars.σ*diagm(ones(data.n))
    # Σinv = inv(Σ)
    NONinv = data.ΩV.Kinv

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(V[:,inds]')
        E = data.Y - pars.U[:,inds] * diagm(pars.D[inds]) * V[:,inds]'
        
        m = pars.D[i] * (1/pars.σ) * N' * E' * pars.U[:,i]
        S = pars.D[i]^2 * (1/pars.σV[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.m - data.k + 1))

        G = cholesky(Hermitian(S))
        b = rand(Normal(0, 1), data.m - data.k + 1)
        VZ[:,i] = (G.U \ ((G.L \ m) + b))
        VZ[:,i] = VZ[:,i] / norm(VZ[:,i])
        V[:,i] = N * VZ[:,i]
    end

    pars.V = V
    pars.VZ = VZ

    return pars
    
end


#### Update D
function likeD(d, i, data::Data, pars::Pars)

    Dtest = copy(diagm(pars.D))
    Dtest[i,i] = d
    
    ll = (-((data.n-data.k+1)/2) * log(2*pi*pars.σU[i]) - (1/2) * data.ΩU.logdet - (1/2)*(d .* pars.UZ[:,i])'*data.ΩU.Kinv*(d .* pars.UZ[:,i]) + (data.n-data.k-1) * log(d) ) +
    (-((data.m-data.k+1)/2) * log(2*pi*pars.σV[i]) - (1/2) * data.ΩV.logdet - (1/2)*(d .* pars.VZ[:,i])'*data.ΩV.Kinv*(d .* pars.VZ[:,i]) + (data.m-data.k-1) * log(d) ) +
    (-((data.m*data.n)/2) * log(2*pi*pars.σ) -(1/2) * (1/pars.σ) * tr((data.Y .- pars.U * Dtest * pars.V')'*(data.Y .- pars.U * Dtest * pars.V')))
    return ll

    # return(logpdf(MvNormal(zeros(size(pars.UZ, 1)), Hermitian(Symmetric(pars.σU[i] .* data.ΩU.K))), d .* pars.UZ[:,i]) + (data.n-data.k-1) * log(d) +
    #        logpdf(MvNormal(zeros(size(pars.VZ, 1)), Hermitian(Symmetric(pars.σV[i] .* data.ΩV.K))), d .* pars.VZ[:,i]) + (data.m-data.k-1) * log(d) +
    #        logpdf(MatrixNormal(pars.U * Dtest * pars.V', pars.σ * I(data.n), I(data.m)), data.Y))

end

function update_D(data::Data, pars::Pars)

    D = copy(pars.D)
    propSD = pars.propSD
    Daccept = pars.Daccept

    for i in 1:data.k

        # propSD = 0.2
        # lbnd = 0.1
        # ubnd = 100
        # lbnd = 0
        # ubnd = Inf
        propD = rand(TruncatedNormal(D[i], propSD[i], 0, Inf))
        
        rat = likeD(propD, i, data, pars) - likeD(D[i], i, data, pars)

        if log(rand(Uniform())) < rat
            D[i] = propD
            Daccept[i] = 1
        end
        
    end

    pars.D = D
    pars.Daccept = Daccept

    return pars
    
end




#### Update σ
function update_σ(data::Data, pars::Pars)

    M = reshape(pars.U * diagm(pars.D) * pars.V',:)
    y = reshape(data.Y, :)
    
    a_hat = 2 + data.n*data.m/2
    b_hat = 2 + 0.5*(y-M)' * (y-M)

    σ = rand(InverseGamma(a_hat, b_hat))

    pars.σ = σ

    return pars
    
end


#### Update σU
function update_σU(data::Data, pars::Pars)

    σU = pars.σU
    a_hat = data.n/2 + 2

    for i in 1:data.k
        b_hat = 2 + 0.5 * (pars.D[i]^2 * pars.UZ[:,i]' * (data.ΩU.Kinv) * pars.UZ[:,i])
        σU[i] = rand(InverseGamma(a_hat, b_hat))
    end

    pars.σU = σU

    return pars
    
end


#### Update σV
function update_σV(data::Data, pars::Pars)

    σV = pars.σV
    a_hat = data.m/2 + 2

    for i in 1:data.k
        b_hat = 2 + 0.5 * (pars.D[i]^2 * pars.VZ[:,i]' * (data.ΩV.Kinv) * pars.VZ[:,i])
        σV[i] = rand(InverseGamma(a_hat, b_hat))
    end

    pars.σV = σV

    return pars
    
end


#### Update ρ
function update_ρ(data::Data, pars::Pars)

    D = copy(pars.D)
    propSD = pars.propSD
    Daccept = pars.Daccept

    for i in 1:data.k

        # propSD = 0.2
        # lbnd = 0.1
        # ubnd = 100
        lbnd = 0
        ubnd = Inf
        propD = rand(TruncatedNormal(D[i], propSD[i], lbnd, ubnd))
        
        rat = likeD(propD, i, data, pars) - likeD(D[i], i, data, pars)

        if log(rand(Uniform())) < rat
            D[i] = propD
            Daccept[i] = 1
        end
        
    end

    pars.D = D
    pars.Daccept = Daccept

    return pars
    
end

# IdentityData
# ExponentialData
# MaternData
# SplineData

# IdentityPars
# ExponentialPars
# MaternPars
# SplinePars

#### Sampling Function
"""
    SampleSVD(data::Data, pars::Pars; nits = 10000, burnin = 5000, show_progress = true)

Runs the MCMC sampler for the Bayesian SVD model.

See also [`Pars`](@ref), [`Data`](@ref), and [`Posterior`](@ref).

# Arguments
- data: Data structure of type Identity, Exponential, Gaussian, or Matern
- pars: Parameter structure of type Identity, Exponential, Gaussian, or Matern

# Optional Arguments
- nits = 10000: Total number of posterior samples to compute
- burnin = 5000: Number of samples discarded as burnin
- show_progress = true: Indicator on whether to show a progress bar (true) or not (false).


# Examples
```@example

Random.seed!(2)

m = 100
n = 100
x = range(-5, 5, n)
t = range(0, 10, m)

ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())
ΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())

k = 5
Φ = PON(n, k, ΣU.K)
Ψ = PON(n, k, ΣV.K)

D = diagm([40, 20, 10, 5, 2])

ϵ = rand(Normal(0, sqrt(0.01)), n, m)
Y = Φ * D * Ψ' + ϵ # n × m


ΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())
ΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Distances.Euclidean())
data = Data(Y, ΩU, ΩV, k)
pars = Pars(data)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = true)
``` 
"""
function SampleSVD(data::IdentityData, pars::IdentityPars; nits = 10000, burnin = 5000, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    UZ_post = Array{Float64}(undef, data.n - data.k + 1, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    VZ_post = Array{Float64}(undef, data.m - data.k + 1, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)

        Daccept_post[:,i] = pars.Daccept
        pars.Daccept = zeros(data.k)

        if mod(i, 10) == 0
            acceptRate = mean(Daccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSD = pars.propSD * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD = pars.propSD / 0.7
                else
                    pars.propSD = pars.propSD
                end
            end
        end

        propSD_post[:,i] = pars.propSD

        ProgressMeter.next!(p)

    end

    p = Progress(keep_samps, desc = "Sampling..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
    
        U_post[:,:,i] = pars.U
        UZ_post[:,:,i] = pars.UZ
        V_post[:,:,i] = pars.V
        VZ_post[:,:,i] = pars.VZ
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post)
    
    return posterior, pars

end

function SampleSVD(data::ExponentialData, pars::ExponentialPars; nits = 10000, burnin = 5000, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    UZ_post = Array{Float64}(undef, data.n - data.k + 1, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    VZ_post = Array{Float64}(undef, data.m - data.k + 1, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρ_post = Array{Float64}(undef, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)

        Daccept_post[:,i] = pars.Daccept
        pars.Daccept = zeros(data.k)

        if mod(i, 10) == 0
            acceptRate = mean(Daccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSD = pars.propSD * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD = pars.propSD / 0.7
                else
                    pars.propSD = pars.propSD
                end
            end
        end

        propSD_post[:,i] = pars.propSD

        ProgressMeter.next!(p)

    end
    
    p = Progress(keep_samps, desc = "Sampling..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)
    
        U_post[:,:,i] = pars.U
        UZ_post[:,:,i] = pars.UZ
        V_post[:,:,i] = pars.V
        VZ_post[:,:,i] = pars.VZ
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρ_post[i] = pars.ρ

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρ_post)
    
    return posterior, pars

end

function SampleSVD(data::GaussianData, pars::GaussianPars; nits = 10000, burnin = 5000, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    UZ_post = Array{Float64}(undef, data.n - data.k + 1, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    VZ_post = Array{Float64}(undef, data.m - data.k + 1, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρ_post = Array{Float64}(undef, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)

        Daccept_post[:,i] = pars.Daccept
        pars.Daccept = zeros(data.k)

        if mod(i, 10) == 0
            acceptRate = mean(Daccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSD = pars.propSD * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD = pars.propSD / 0.7
                else
                    pars.propSD = pars.propSD
                end
            end
        end

        propSD_post[:,i] = pars.propSD

        ProgressMeter.next!(p)

    end
    
    p = Progress(keep_samps, desc = "Sampling..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)
    
        U_post[:,:,i] = pars.U
        UZ_post[:,:,i] = pars.UZ
        V_post[:,:,i] = pars.V
        VZ_post[:,:,i] = pars.VZ
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρ_post[i] = pars.ρ

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρ_post)
    
    return posterior, pars

end

function SampleSVD(data::MaternData, pars::MaternPars; nits = 10000, burnin = 5000, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    UZ_post = Array{Float64}(undef, data.n - data.k + 1, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    VZ_post = Array{Float64}(undef, data.m - data.k + 1, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρ_post = Array{Float64}(undef, keep_samps)
    ν_post = Array{Float64}(undef, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)
        # pars = update_ν(data, pars)

        Daccept_post[:,i] = pars.Daccept
        pars.Daccept = zeros(data.k)

        if mod(i, 10) == 0
            acceptRate = mean(Daccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSD = pars.propSD * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD = pars.propSD / 0.7
                else
                    pars.propSD = pars.propSD
                end
            end
        end

        propSD_post[:,i] = pars.propSD

        ProgressMeter.next!(p)

    end
    
    p = Progress(keep_samps, desc = "Sampling..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)
        # pars = update_ν(data, pars)
    
        U_post[:,:,i] = pars.U
        UZ_post[:,:,i] = pars.UZ
        V_post[:,:,i] = pars.V
        VZ_post[:,:,i] = pars.VZ
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρ_post[i] = pars.ρ
        ν_post[i] = pars.ν

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρ_post, ν_post)
    
    return posterior, pars

end