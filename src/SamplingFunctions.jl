######################################################################
#### Sampling Functions
######################################################################

#### Update β
function update_β(data::MixedEffectData, pars::MixedEffectPars)

    Ztilde = reshape(data.Z - pars.U * diagm(pars.D) * pars.V', :)
    m = (1/pars.σ) * data.X' * Ztilde
    S = (1/pars.σ) * data.X' * data.X + diagm(1/100*ones(data.p))

    pars.β = rand(MvNormalCanon(m, Hermitian(S)))
    pars.m = data.X * pars.β
    pars.M = reshape(pars.m, data.n, data.m)

    return pars
    
end

#### Update U
function update_U(data::MixedEffectData, pars::MixedEffectPars)

    U = pars.U
    UZ = pars.UZ

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(U[:,inds]')
        E = data.Z - pars.M - U[:,inds] * diagm(pars.D[inds]) * pars.V[:,inds]'

        NON = Hermitian(N' * pars.ΩU[i].K * N)
        NONinv = inv(NON)
        
        m = pars.D[i] * (1/pars.σ) * N' * E * pars.V[:,i]
        S = pars.D[i]^2 * (1/pars.σU[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.n - data.k + 1))

        UZ[:,i] = rand(MvNormalCanon(m, Hermitian(S)))
        UZ[:,i] = UZ[:,i] / norm(UZ[:,i])
        U[:,i] = N * UZ[:,i]
        pars.NU[:,:,i] = N
        pars.NΩUN[:,:,i] = NON
        pars.NΩUNinv[:,:,i] = NONinv
    end

    pars.U = U
    pars.UZ = UZ

    return pars
    
end

function update_U(data::RandomEffectData, pars::RandomEffectPars)

    U = pars.U
    UZ = pars.UZ

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(U[:,inds]')
        E = data.Z - U[:,inds] * diagm(pars.D[inds]) * pars.V[:,inds]'

        NON = Hermitian(N' * pars.ΩU[i].K * N)
        NONinv = inv(NON)
        
        m = pars.D[i] * (1/pars.σ) * N' * E * pars.V[:,i]
        S = pars.D[i]^2 * (1/pars.σU[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.n - data.k + 1))

        UZ[:,i] = rand(MvNormalCanon(m, Hermitian(S)))
        UZ[:,i] = UZ[:,i] / norm(UZ[:,i])
        U[:,i] = N * UZ[:,i]
        pars.NU[:,:,i] = N
        pars.NΩUN[:,:,i] = NON
        pars.NΩUNinv[:,:,i] = NONinv
    end

    pars.U = U
    pars.UZ = UZ

    return pars
    
end

#### Update V
function update_V(data::MixedEffectData, pars::MixedEffectPars)

    V = pars.V
    VZ = pars.VZ

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(V[:,inds]')
        E = data.Z - pars.M - pars.U[:,inds] * diagm(pars.D[inds]) * V[:,inds]'

        NON = Hermitian(N' * pars.ΩV[i].K * N)
        NONinv = inv(NON)
        
        m = pars.D[i] * (1/pars.σ) * N' * E' * pars.U[:,i]
        S = pars.D[i]^2 * (1/pars.σV[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.m - data.k + 1))

        VZ[:,i] = rand(MvNormalCanon(m, Hermitian(S)))
        VZ[:,i] = VZ[:,i] / norm(VZ[:,i])
        V[:,i] = N * VZ[:,i]
        pars.NV[:,:,i] = N
        pars.NΩVN[:,:,i] = NON
        pars.NΩVNinv[:,:,i] = NONinv
    end

    pars.V = V
    pars.VZ = VZ

    return pars
    
end

function update_V(data::RandomEffectData, pars::RandomEffectPars)

    V = pars.V
    VZ = pars.VZ

    for i in 1:data.k
        inds = i .∉ Vector(1:data.k)
        N = nullspace(V[:,inds]')
        E = data.Z - pars.U[:,inds] * diagm(pars.D[inds]) * V[:,inds]'

        NON = Hermitian(N' * pars.ΩV[i].K * N)
        NONinv = inv(NON)
        
        m = pars.D[i] * (1/pars.σ) * N' * E' * pars.U[:,i]
        S = pars.D[i]^2 * (1/pars.σV[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.m - data.k + 1))

        VZ[:,i] = rand(MvNormalCanon(m, Hermitian(S)))
        VZ[:,i] = VZ[:,i] / norm(VZ[:,i])
        V[:,i] = N * VZ[:,i]
        pars.NV[:,:,i] = N
        pars.NΩVN[:,:,i] = NON
        pars.NΩVNinv[:,:,i] = NONinv
    end

    pars.V = V
    pars.VZ = VZ

    return pars
    
end

#### Update D
function likeD(d, i, data::MixedEffectData, pars::MixedEffectPars)

    Dtest = copy(diagm(pars.D))
    Dtest[i,i] = d

    # SigU = pars.NU[:,:,i]' * pars.ΩU[i].K * pars.NU[:,:,i]
    # SigV = pars.NV[:,:,i]' * pars.ΩV[i].K * pars.NV[:,:,i]
    SigUinv = pars.NΩUNinv[:,:,i]./pars.σU[i]
    SigVinv = pars.NΩVNinv[:,:,i]./pars.σV[i]
    SigUdet = logdet(pars.NΩUN[:,:,i])
    SigVdet = logdet(pars.NΩVN[:,:,i])
    

    ll = (-((data.n-data.k+1)/2) * log(2*pi*pars.σU[i]) - (1/2) * SigUdet - (1/2)*(d .* pars.UZ[:,i])' * SigUinv * (d .* pars.UZ[:,i]) + (data.n-data.k-1) * log(d) ) +
    (-((data.m-data.k+1)/2) * log(2*pi*pars.σV[i]) - (1/2) * SigVdet - (1/2)*(d .* pars.VZ[:,i])' * SigVinv * (d .* pars.VZ[:,i]) + (data.m-data.k-1) * log(d) ) +
    (-((data.m*data.n)/2) * log(2*pi*pars.σ) -(1/2) * (1/pars.σ) * tr((data.Z .- pars.M .- pars.U * Dtest * pars.V')'*(data.Z .- pars.M .- pars.U * Dtest * pars.V')))
    return ll

end

function likeD(d, i, data::RandomEffectData, pars::RandomEffectPars)

    Dtest = copy(diagm(pars.D))
    Dtest[i,i] = d

    # SigU = pars.NU[:,:,i]' * pars.ΩU[i].K * pars.NU[:,:,i]
    # SigV = pars.NV[:,:,i]' * pars.ΩV[i].K * pars.NV[:,:,i]
    SigUinv = pars.NΩUNinv[:,:,i]./pars.σU[i]
    SigVinv = pars.NΩVNinv[:,:,i]./pars.σV[i]
    SigUdet = logdet(pars.NΩUN[:,:,i])
    SigVdet = logdet(pars.NΩVN[:,:,i])
    
    ll = (-((data.n-data.k+1)/2) * log(2*pi*pars.σU[i]) - (1/2) * SigUdet - (1/2)*(d .* pars.UZ[:,i])' * SigUinv * (d .* pars.UZ[:,i]) + (data.n-data.k-1) * log(d) ) +
    (-((data.m-data.k+1)/2) * log(2*pi*pars.σV[i]) - (1/2) * SigVdet - (1/2)*(d .* pars.VZ[:,i])' * SigVinv * (d .* pars.VZ[:,i]) + (data.m-data.k-1) * log(d) ) +
    (-((data.m*data.n)/2) * log(2*pi*pars.σ) -(1/2) * (1/pars.σ) * tr((data.Z .- pars.U * Dtest * pars.V')'*(data.Z .- pars.U * Dtest * pars.V')))
    return ll

end


function update_D(data::MixedEffectData, pars::MixedEffectPars)

    D = copy(pars.D)
    propSD = pars.propSD
    Daccept = pars.Daccept

    for i in 1:data.k

        # propD = rand(TruncatedNormal(D[i], propSD[i], 0, Inf))
        propD = rand(truncated(Normal(D[i], propSD[i]), 0, Inf))
        
        # rat = likeD(propD, i, data, pars) + logpdf(TruncatedNormal(propD, propSD[i], 0, Inf), D[i]) - 
        #         (likeD(D[i], i, data, pars) + logpdf(TruncatedNormal(D[i], propSD[i], 0, Inf), propD))
        rat = likeD(propD, i, data, pars) + logpdf(truncated(Normal(propD, propSD[i]), 0, Inf), D[i]) - 
                (likeD(D[i], i, data, pars) + logpdf(truncated(Normal(D[i], propSD[i]), 0, Inf), propD))

        if log(rand(Uniform())) < rat
            D[i] = propD
            Daccept[i] = 1
        end
        
    end

    pars.D = D
    pars.Daccept = Daccept

    return pars
    
end

function update_D(data::RandomEffectData, pars::RandomEffectPars)

    D = copy(pars.D)
    propSD = pars.propSD
    Daccept = pars.Daccept

    for i in 1:data.k

        # propD = rand(TruncatedNormal(D[i], propSD[i], 0, Inf))
        propD = rand(truncated(Normal(D[i], propSD[i]), 0, Inf))
        
        # rat = likeD(propD, i, data, pars) + logpdf(TruncatedNormal(propD, propSD[i], 0, Inf), D[i]) - 
        #         (likeD(D[i], i, data, pars) + logpdf(TruncatedNormal(D[i], propSD[i], 0, Inf), propD))
        rat = likeD(propD, i, data, pars) + logpdf(truncated(Normal(propD, propSD[i]), 0, Inf), D[i]) - 
                (likeD(D[i], i, data, pars) + logpdf(truncated(Normal(D[i], propSD[i]), 0, Inf), propD))

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
function update_σ(data::MixedEffectData, pars::MixedEffectPars)

    # M = reshape(pars.U * diagm(pars.D) * pars.V',:)
    # y = reshape(data.Y, :)
    
    # a_hat = 2 + (data.n*data.m)/2
    # b_hat = 2 + 0.5*(y-M)' * (y-M)

    # σ = rand(InverseGamma(a_hat, b_hat))

    # pars.σ = σ



    σ = pars.σ
    y = reshape(pars.U * diagm(pars.D) * pars.V',:)
    z = reshape(data.Z, :)
    ν = 2
    A = 1e6

    a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σ))
    σ = rand(InverseGamma((data.n*data.m+ν)/2, ν/a + 0.5 * (z-pars.m-y)' * (z-pars.m-y)))

    pars.σ = σ

    return pars
    
end

function update_σ(data::RandomEffectData, pars::RandomEffectPars)

    # M = reshape(pars.U * diagm(pars.D) * pars.V',:)
    # y = reshape(data.Y, :)
    
    # a_hat = 2 + (data.n*data.m)/2
    # b_hat = 2 + 0.5*(y-M)' * (y-M)

    # σ = rand(InverseGamma(a_hat, b_hat))

    # pars.σ = σ



    σ = pars.σ
    y = reshape(pars.U * diagm(pars.D) * pars.V',:)
    z = reshape(data.Z, :)
    ν = 1
    A = 1e6

    a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σ))
    σ = rand(InverseGamma((data.n*data.m+ν)/2, ν/a + 0.5 * (z-y)' * (z-y)))

    pars.σ = σ

    return pars
    
end


#### Update σU
function update_σU(data::Data, pars::Pars)

    σU = pars.σU
    # a_hat = data.n/2 + 2

    # for i in 1:data.k
    #     b_hat = 2 + 0.5 * (pars.D[i]^2 * pars.UZ[:,i]' * data.ΩU.Kinv * pars.UZ[:,i])
    #     σU[i] = rand(InverseGamma(a_hat, b_hat))
    # end

    ν = 1
    A = 1e6
    for i in 1:data.k
        a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σU[i]))
        σU[i] = rand(InverseGamma((data.n-data.k+1+ν)/2, ν/a + 0.5 * (pars.D[i]^2 * pars.UZ[:,i]' * pars.NΩUNinv[:,:,i] * pars.UZ[:,i])))
    end

    pars.σU = σU

    return pars
    
end


#### Update σV
function update_σV(data::Data, pars::Pars)

    σV = pars.σV
    # a_hat = data.m/2 + 2

    # for i in 1:data.k
    #     b_hat = 2 + 0.5 * (pars.D[i]^2 * pars.VZ[:,i]' * data.ΩV.Kinv * pars.VZ[:,i])
    #     σV[i] = rand(InverseGamma(a_hat, b_hat))
    # end

    ν = 1
    A = 1e6
    for i in 1:data.k
        a = rand(InverseGamma((ν+1)/2, (1/A^2) + ν/σV[i]))
        σV[i] = rand(InverseGamma((data.m-data.k+1+ν)/2, ν/a + 0.5 * (pars.D[i]^2 * pars.VZ[:,i]' * pars.NΩVNinv[:,:,i] * pars.VZ[:,i])))
    end

    pars.σV = σV

    return pars
    
end


#### Update ρ
function ρLike(u, C, N, σ, d)

    return logpdf(MvNormal(zeros(size(u, 1)), Hermitian(σ * N' * C.K * N)), d*u)

end

function ρLike(u, d, Σ)

    return logpdf(MvNormal(zeros(size(u, 1)), Σ), d*u)

end


function update_ρU(data::Data, pars::Pars, Ω::T) where T <: DependentCorrelation

    for i in 1:data.k
        propsd = pars.propSU[i]
        # ρprop = rand(TruncatedNormal(pars.ΩU[i].ρ, propsd, 0, maximum(pars.ΩU[1].d)/2))
        # ρprop = rand(truncated(Normal(pars.ΩU[i].ρ, propsd), 0, maximum(pars.ΩU[1].d)/2))
        ρprop = rand(truncated(Normal(pars.ΩU[i].ρ, propsd), 0, pars.ρUMax[i]))
        Cprop = copy(pars.ΩU[i])
        Cprop.ρ = ρprop
        Cprop = updateCorrelation(Cprop)
        # ρprior = InverseGamma(2, 2)
        ρprior = Uniform(0, pars.ρUMax[i])

        Σprop =  pars.σU[i] * Hermitian(pars.NU[:,:,i]' * Cprop.K * pars.NU[:,:,i])
        Σcurr = pars.σU[i] * pars.NΩUN[:,:,i]
        
        # rat = ρLike(pars.UZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(TruncatedNormal(ρprop, propsd, 0, Inf), pars.ΩU[i].ρ) - 
        #         (ρLike(pars.UZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩU[i].ρ) + logpdf(TruncatedNormal(pars.ΩU[i].ρ, propsd, 0, Inf), ρprop))
        rat = ρLike(pars.UZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(truncated(Normal(ρprop, propsd), 0, Inf), pars.ΩU[i].ρ) - 
                (ρLike(pars.UZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩU[i].ρ) + logpdf(truncated(Normal(pars.ΩU[i].ρ, propsd), 0, Inf), ρprop))

        
        if log(rand(Uniform())) < rat
            pars.ΩU[i] = Cprop
            pars.Uaccept[i] = 1
        end

    end

    return pars
    
end

function update_ρU(data::Data, pars::Pars, Ω::T) where T <: IndependentCorrelation
    return pars
end

function update_ρV(data::Data, pars::Pars, Ω::T) where T <: DependentCorrelation

    for i in 1:data.k
        propsd = pars.propSV[i]
        # ρprop = rand(TruncatedNormal(pars.ΩV[i].ρ, propsd, 0, maximum(pars.ΩV[1].d)/2))
        # ρprop = rand(truncated(Normal(pars.ΩV[i].ρ, propsd), 0, maximum(pars.ΩV[1].d)/2))
        ρprop = rand(truncated(Normal(pars.ΩV[i].ρ, propsd), 0, pars.ρVMax[i]))
        Cprop = copy(pars.ΩV[i])
        Cprop.ρ = ρprop
        Cprop = updateCorrelation(Cprop)
        # ρprior = InverseGamma(2, 2)
        ρprior = Uniform(0, pars.ρVMax[i])

        Σprop =  pars.σV[i] * Hermitian(pars.NV[:,:,i]' * Cprop.K * pars.NV[:,:,i])
        Σcurr = pars.σV[i] * pars.NΩVN[:,:,i]
        
        # rat = ρLike(pars.VZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(TruncatedNormal(ρprop, propsd, 0, Inf), pars.ΩV[i].ρ) - 
        #         (ρLike(pars.VZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩV[i].ρ) + logpdf(TruncatedNormal(pars.ΩV[i].ρ, propsd, 0, Inf), ρprop))
        rat = ρLike(pars.VZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(truncated(Normal(ρprop, propsd), 0, Inf), pars.ΩV[i].ρ) - 
                (ρLike(pars.VZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩV[i].ρ) + logpdf(truncated(Normal(pars.ΩV[i].ρ, propsd), 0, Inf), ρprop))

        if log(rand(Uniform())) < rat
            pars.ΩV[i] = Cprop
            pars.Vaccept[i] = 1
        end

    end

    return pars
    
end

function update_ρV(data::Data, pars::Pars, Ω::T) where T <: IndependentCorrelation
    return pars
end


function update_ρUGrouped(data::Data, pars::Pars, Ω::T) where T <: DependentCorrelation

    propsd = pars.propSU[1]
    ρprop = rand(truncated(Normal(pars.ΩU[1].ρ, propsd), 0, pars.ρUMax[1]))
    Cprop = copy(pars.ΩU[1])
    Cprop.ρ = ρprop
    Cprop = updateCorrelation(Cprop)
    ρprior = Uniform(0, pars.ρUMax[1])
    
    
    rat = 0
    for i in 1:data.k
        
        Σprop =  pars.σU[i] * Hermitian(pars.NU[:,:,i]' * Cprop.K * pars.NU[:,:,i])
        Σcurr = pars.σU[i] * pars.NΩUN[:,:,i]
    
        rat += ρLike(pars.UZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(truncated(Normal(ρprop, propsd), 0, Inf), pars.ΩU[i].ρ) - 
                (ρLike(pars.UZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩU[i].ρ) + logpdf(truncated(Normal(pars.ΩU[i].ρ, propsd), 0, Inf), ρprop))
    
    end
    
    if log(rand(Uniform())) < rat
        for i in 1:data.k
            pars.ΩU[i] = Cprop
            pars.Uaccept[i] = 1
        end
    end

    return pars
    
end

function update_ρUGrouped(data::Data, pars::Pars, Ω::T) where T <: IndependentCorrelation
    return pars
end

function update_ρVGrouped(data::Data, pars::Pars, Ω::T) where T <: DependentCorrelation

    propsd = pars.propSV[1]
    ρprop = rand(truncated(Normal(pars.ΩV[1].ρ, propsd), 0, pars.ρVMax[1]))
    Cprop = copy(pars.ΩV[1])
    Cprop.ρ = ρprop
    Cprop = updateCorrelation(Cprop)
    ρprior = Uniform(0, pars.ρVMax[1])
    
    
    rat = 0
    for i in 1:data.k
        
        Σprop =  pars.σV[i] * Hermitian(pars.NV[:,:,i]' * Cprop.K * pars.NV[:,:,i])
        Σcurr = pars.σV[i] * pars.NΩVN[:,:,i]
    
        rat += ρLike(pars.VZ[:,i], pars.D[i], Σprop) + logpdf(ρprior, ρprop) + logpdf(truncated(Normal(ρprop, propsd), 0, Inf), pars.ΩV[i].ρ) - 
                (ρLike(pars.VZ[:,i], pars.D[i], Σcurr) + logpdf(ρprior, pars.ΩV[i].ρ) + logpdf(truncated(Normal(pars.ΩV[i].ρ, propsd), 0, Inf), ρprop))
    
    end
    
    if log(rand(Uniform())) < rat
        for i in 1:data.k
            pars.ΩV[i] = Cprop
            pars.Vaccept[i] = 1
        end
    end

    return pars
    
end

function update_ρVGrouped(data::Data, pars::Pars, Ω::T) where T <: IndependentCorrelation
    return pars
end




#### Sampling Function
"""
    SampleSVD(data::Data, pars::Pars; nits = 10000, burnin = 5000, show_progress = true)

Runs the MCMC sampler for the Bayesian SVD model.

    See also [`Pars`](@ref), [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVDGrouped`](@ref).

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

ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5
ϵ = 2

U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)


ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)
``` 
"""
function SampleSVD(data::MixedEffectData, pars::MixedEffectPars; nits = 1000, burnin = 500, show_progress = true)

    keep_samps = Int(nits - burnin)

    β_post = Array{Float64}(undef, data.p, keep_samps)
    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρU_post = Array{Float64}(undef, data.k, keep_samps)
    ρV_post = Array{Float64}(undef, data.k, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)
    propSU_post = Array{Float64}(undef, data.k, burnin)
    Uaccept_post = Array{Float64}(undef, data.k, burnin)
    propSV_post = Array{Float64}(undef, data.k, burnin)
    Vaccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_β(data, pars)
        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρU(data, pars, pars.ΩU[1])
        pars = update_ρV(data, pars, pars.ΩV[1])

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
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:keep_samps

        pars = update_β(data, pars)
        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρU(data, pars, pars.ΩU[1])
        pars = update_ρV(data, pars, pars.ΩV[1])
    
        β_post[:,i] = pars.β
        U_post[:,:,i] = pars.U
        V_post[:,:,i] = pars.V
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρU_post[:,i] = [pars.ΩU[i].ρ for i in 1:data.k]
        ρV_post[:,i] = [pars.ΩV[i].ρ for i in 1:data.k]

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, β_post, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)
    
    return posterior, pars

end

function SampleSVD(data::RandomEffectData, pars::RandomEffectPars; nits = 1000, burnin = 500, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρU_post = Array{Float64}(undef, data.k, keep_samps)
    ρV_post = Array{Float64}(undef, data.k, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)
    propSU_post = Array{Float64}(undef, data.k, burnin)
    Uaccept_post = Array{Float64}(undef, data.k, burnin)
    propSV_post = Array{Float64}(undef, data.k, burnin)
    Vaccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρU(data, pars, pars.ΩU[1])
        pars = update_ρV(data, pars, pars.ΩV[1])

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
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:keep_samps

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρU(data, pars, pars.ΩU[1])
        pars = update_ρV(data, pars, pars.ΩV[1])
    
        U_post[:,:,i] = pars.U
        V_post[:,:,i] = pars.V
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρU_post[:,i] = [pars.ΩU[i].ρ for i in 1:data.k]
        ρV_post[:,i] = [pars.ΩV[i].ρ for i in 1:data.k]

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)
    
    return posterior, pars

end



"""
    SampleSVDGrouped(data::Data, pars::Pars; nits = 10000, burnin = 5000, show_progress = true)

    Runs the MCMC sampler for the Bayesian SVD model. Exact same documentation as SampleSVD except now the length-scales for U and V
    are restricted to be the same.

See also [`Pars`](@ref), [`Data`](@ref), [`Posterior`](@ref), and [`SampleSVD`](@ref).

"""
function SampleSVDGrouped(data::MixedEffectData, pars::MixedEffectPars; nits = 1000, burnin = 500, show_progress = true)

    keep_samps = Int(nits - burnin)

    β_post = Array{Float64}(undef, data.p, keep_samps)
    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρU_post = Array{Float64}(undef, data.k, keep_samps)
    ρV_post = Array{Float64}(undef, data.k, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)
    propSU_post = Array{Float64}(undef, data.k, burnin)
    Uaccept_post = Array{Float64}(undef, data.k, burnin)
    propSV_post = Array{Float64}(undef, data.k, burnin)
    Vaccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_β(data, pars)
        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρUGrouped(data, pars, pars.ΩU[1])
        pars = update_ρVGrouped(data, pars, pars.ΩV[1])

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
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:keep_samps

        pars = update_β(data, pars)
        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρUGrouped(data, pars, pars.ΩU[1])
        pars = update_ρVGrouped(data, pars, pars.ΩV[1])
    
        β_post[:,i] = pars.β
        U_post[:,:,i] = pars.U
        V_post[:,:,i] = pars.V
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρU_post[:,i] = [pars.ΩU[i].ρ for i in 1:data.k]
        ρV_post[:,i] = [pars.ΩV[i].ρ for i in 1:data.k]

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, β_post, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)
    
    return posterior, pars

end

function SampleSVDGrouped(data::RandomEffectData, pars::RandomEffectPars; nits = 1000, burnin = 500, show_progress = true)

    keep_samps = Int(nits - burnin)

    U_post = Array{Float64}(undef, data.n, data.k, keep_samps)
    V_post = Array{Float64}(undef, data.m, data.k, keep_samps)
    D_post = Array{Float64}(undef, data.k, keep_samps)
    σ_post = Array{Float64}(undef, keep_samps)
    σU_post = Array{Float64}(undef, data.k, keep_samps)
    σV_post = Array{Float64}(undef, data.k, keep_samps)
    ρU_post = Array{Float64}(undef, data.k, keep_samps)
    ρV_post = Array{Float64}(undef, data.k, keep_samps)
    propSD_post = Array{Float64}(undef, data.k, burnin)
    Daccept_post = Array{Float64}(undef, data.k, burnin)
    propSU_post = Array{Float64}(undef, data.k, burnin)
    Uaccept_post = Array{Float64}(undef, data.k, burnin)
    propSV_post = Array{Float64}(undef, data.k, burnin)
    Vaccept_post = Array{Float64}(undef, data.k, burnin)

    p = Progress(burnin, desc = "Burnin..."; showspeed = true, enabled = show_progress)
    # @showprogress 1 "Burnin..." for i in 1:burnin
    for i in 1:burnin

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρUGrouped(data, pars, pars.ΩU[1])
        pars = update_ρVGrouped(data, pars, pars.ΩV[1])

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
    # @showprogress 1 "Sampling..." for i in 1:keep_samps
    for i in 1:keep_samps

        pars = update_D(data, pars)
        pars = update_U(data, pars)
        pars = update_V(data, pars)
        pars = update_σ(data, pars)
        pars = update_σU(data, pars)
        pars = update_σV(data, pars)
        pars = update_ρUGrouped(data, pars, pars.ΩU[1])
        pars = update_ρVGrouped(data, pars, pars.ΩV[1])
    
        U_post[:,:,i] = pars.U
        V_post[:,:,i] = pars.V
        D_post[:,i] = pars.D
        σ_post[i] = pars.σ
        σU_post[:,i] = pars.σU
        σV_post[:,i] = pars.σV
        ρU_post[:,i] = [pars.ΩU[i].ρ for i in 1:data.k]
        ρV_post[:,i] = [pars.ΩV[i].ρ for i in 1:data.k]

        ProgressMeter.next!(p)
    
    end
    
    posterior = Posterior(data, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)
    
    return posterior, pars

end


