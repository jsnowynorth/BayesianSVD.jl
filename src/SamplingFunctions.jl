######################################################################
#### Sampling Functions
######################################################################


#### Update U
function update_U(data::Data, pars::Pars)

    U = pars.U
    UZ = pars.UZ
    # Σ = pars.σ*diagm(ones(data.n))
    # Σinv = inv(Σ)
    # NONinv = data.ΩU.Kinv

    for i in 1:data.k
        NONinv = data.ΩU[i].Kinv
        inds = i .∉ Vector(1:data.k)
        N = nullspace(U[:,inds]')
        E = data.Y - U[:,inds] * diagm(pars.D[inds]) * pars.V[:,inds]'
        
        m = pars.D[i] * (1/pars.σ) * N' * E * pars.V[:,i]
        S = pars.D[i]^2 * (1/pars.σU[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.n - data.k + 1))

        UZ[:,i] = rand(MvNormalCanon(m, S))
        # G = cholesky(Hermitian(S))
        # b = rand(Normal(0, 1), data.n - data.k + 1)
        # UZ[:,i] = (G.U \ ((G.L \ m) + b))
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
    # NONinv = data.ΩV.Kinv

    for i in 1:data.k
        NONinv = data.ΩV[i].Kinv
        inds = i .∉ Vector(1:data.k)
        N = nullspace(V[:,inds]')
        E = data.Y - pars.U[:,inds] * diagm(pars.D[inds]) * V[:,inds]'
        
        m = pars.D[i] * (1/pars.σ) * N' * E' * pars.U[:,i]
        S = pars.D[i]^2 * (1/pars.σV[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.m - data.k + 1))

        VZ[:,i] = rand(MvNormalCanon(m, S))
        # G = cholesky(Hermitian(S))
        # b = rand(Normal(0, 1), data.m - data.k + 1)
        # VZ[:,i] = (G.U \ ((G.L \ m) + b))
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
    
    ll = (-((data.n-data.k+1)/2) * log(2*pi*pars.σU[i]) - (1/2) * data.ΩU[i].logdet - (1/2)*(d .* pars.UZ[:,i])' * (data.ΩU[i].Kinv./pars.σU[i]) * (d .* pars.UZ[:,i]) + (data.n-data.k-1) * log(d) ) +
    (-((data.m-data.k+1)/2) * log(2*pi*pars.σV[i]) - (1/2) * data.ΩV[i].logdet - (1/2)*(d .* pars.VZ[:,i])' * (data.ΩV[i].Kinv/pars.σV[i]) * (d .* pars.VZ[:,i]) + (data.m-data.k-1) * log(d) ) +
    (-((data.m*data.n)/2) * log(2*pi*pars.σ) -(1/2) * (1/pars.σ) * tr((data.Y .- pars.U * Dtest * pars.V')'*(data.Y .- pars.U * Dtest * pars.V')))
    return ll

end

function update_D(data::Data, pars::Pars)

    D = copy(pars.D)
    propSD = pars.propSD
    Daccept = pars.Daccept

    for i in 1:data.k

        propD = rand(TruncatedNormal(D[i], propSD[i], 0, Inf))
        
        rat = likeD(propD, i, data, pars) + logpdf(TruncatedNormal(propD, propSD[i], 0, Inf), D[i]) - likeD(D[i], i, data, pars) - logpdf(TruncatedNormal(D[i], propSD[i], 0, Inf), propD)

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

    ν = 2
    A = 1e6
    for i in 1:data.k
        a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σU[i]))
        σU[i] = rand(InverseGamma((data.n+ν)/2, ν/a + 0.5 * (pars.D[i]^2 * pars.UZ[:,i]' * data.ΩU[i].Kinv * pars.UZ[:,i])))
    end

    pars.σU = σU

    return pars
    
end



#### Update σV
function update_σV(data::Data, pars::Pars)

    σV = pars.σV
    
    ν = 2
    A = 1e6
    for i in 1:data.k
        a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σV[i]))
        σV[i] = rand(InverseGamma((data.m+ν)/2, ν/a + 0.5 * (pars.D[i]^2 * pars.VZ[:,i]' * data.ΩV[i].Kinv * pars.VZ[:,i])))
    end

    pars.σV = σV

    return pars
    
end


#### Update ρ
function ρLike(u, C)
    return sum(logpdf(MvNormal(zeros(size(u, 1)), C.K), u))
end

function update_ρ(data::Data, pars::Pars)

    for i in 1:data.k
        propsd = pars.propSU[i]
        ρprop = rand(TruncatedNormal(data.ΩU[i].ρ, propsd, 0, 5))
        Cprop = copy(data.ΩU[i])
        Cprop.ρ = ρprop
        Cprop = MaternKernel(Cprop)
        ρprior = InverseGamma(2, 2)

        rat = ρLike(pars.UZ[:,i], Cprop) + logpdf(ρprior, ρprop) + logpdf(TruncatedNormal(ρprop, propsd, 0, Inf), data.ΩU[i].ρ) - 
                (ρLike(pars.UZ[:,i], data.ΩU[i]) + logpdf(ρprior, data.ΩU[i].ρ) + logpdf(TruncatedNormal(data.ΩU[i].ρ, propsd, 0, Inf), ρprop))

        if log(rand(Uniform())) < rat
            pars.ρU[i] = ρprop
            data.ΩU[i] = Cprop
            pars.Uaccept[i] = 1
        end

    end

    for i in 1:data.k
        propsd = pars.propSV[i]
        ρprop = rand(TruncatedNormal(data.ΩU[i].ρ, propsd, 0, 5))
        Cprop = copy(data.ΩV[i])
        Cprop.ρ = ρprop
        Cprop = MaternKernel(Cprop)
        ρprior = InverseGamma(2, 2)

        rat = ρLike(pars.VZ[:,i], Cprop) + logpdf(ρprior, ρprop) + logpdf(TruncatedNormal(ρprop, propsd, 0, Inf), data.ΩV[i].ρ) - 
                (ρLike(pars.VZ[:,i], data.ΩV[i]) + logpdf(ρprior, data.ΩV[i].ρ) + logpdf(TruncatedNormal(data.ΩV[i].ρ, propsd, 0, Inf), ρprop))

       
        if log(rand(Uniform())) < rat
            pars.ρV[i] = ρprop
            data.ΩV[i] = Cprop
            pars.Vaccept[i] = 1
        end

    end

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
                    pars.propSD[j] = pars.propSD[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD[j] = pars.propSD[j] / 0.7
                else
                    pars.propSD[j] = pars.propSD[j]
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
                    pars.propSD[j] = pars.propSD[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD[j] = pars.propSD[j] / 0.7
                else
                    pars.propSD[j] = pars.propSD[j]
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
                    pars.propSD[j] = pars.propSD[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD[j] = pars.propSD[j] / 0.7
                else
                    pars.propSD[j] = pars.propSD[j]
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
    ρU_post = Array{Float64}(undef, data.k, keep_samps)
    ρV_post = Array{Float64}(undef, data.k, keep_samps)
    ν_post = Array{Float64}(undef, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)
    propSU_post = Array{Float64}(undef, data.k, burnin)
    Uaccept_post = Array{Float64}(undef, data.k, burnin)
    propSV_post = Array{Float64}(undef, data.k, burnin)
    Vaccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        # pars = update_ρ(data, pars)
        # pars = update_ν(data, pars)

        # D acceptance rate
        Daccept_post[:,i] = pars.Daccept
        pars.Daccept = zeros(data.k)

        if mod(i, 10) == 0
            acceptRate = mean(Daccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSD[j] = pars.propSD[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSD[j] = pars.propSD[j] / 0.7
                else
                    pars.propSD[j] = pars.propSD[j]
                end
            end
        end
        propSD_post[:,i] = pars.propSD

        # ρU acceptance rate
        Uaccept_post[:,i] = pars.Uaccept
        pars.Uaccept = zeros(data.k)
    
        if mod(i, 10) == 0
            acceptRate = mean(Uaccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSU[j] = pars.propSU[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSU[j] = pars.propSU[j] / 0.7
                else
                    pars.propSU[j] = pars.propSU[j]
                end
            end
        end
        propSU_post[:,i] = pars.propSU


        # ρV acceptance rate
        Vaccept_post[:,i] = pars.Vaccept
        pars.Vaccept = zeros(data.k)
    
        if mod(i, 10) == 0
            acceptRate = mean(Vaccept_post[:,(i-9):i], dims = 2)
            for j in 1:data.k
                if acceptRate[j] < 0.25
                    pars.propSV[j] = pars.propSV[j] * 0.7
                elseif acceptRate[j] > 0.45
                    pars.propSV[j] = pars.propSV[j] / 0.7
                else
                    pars.propSV[j] = pars.propSV[j]
                end
            end
        end
        propSV_post[:,i] = pars.propSV

        ProgressMeter.next!(p)

    end
    
    p = Progress(keep_samps, desc = "Sampling..."; showspeed = true, enabled = show_progress)
    for i in 1:keep_samps

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
        ρU_post[:,i] = pars.ρU
        ρV_post[:,i] = pars.ρV
        ν_post[i] = pars.ν

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post, ν_post)
    
    return posterior, pars

end