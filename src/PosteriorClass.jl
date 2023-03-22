######################################################################
#### Posterior Structures
######################################################################

# IdentityPosterior
# ExponentialPosterior
# MaternPosterior
# SplinePosterior

"""
    Posterior

Structure of type posterior with subtypes Identity, Exponential, Gaussian, or Matern.
Contains the raw posterior samples and some means and 95% quantiles of parameters.
Plotting associated with the structure.

See also [`Pars`](@ref), [`Data`](@ref), and [`SampleSVD`](@ref).

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

plot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple])
plot(posterior, t, size = (900, 500), basis = 'V', linewidth = 2, c = [:red :green :purple])

# for spatial basis functions, provide x and y
plot(posterior, x, y)
```
"""
abstract type Posterior end

struct IdentityPosterior <: Posterior 
    U
    UZ
    V
    VZ
    D
    σ
    σU
    σV

    U_hat
    UZ_hat
    V_hat
    VZ_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat

    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end

struct ExponentialPosterior <: Posterior 
    U
    UZ
    V
    VZ
    D
    σ
    σU
    σV
    ρ

    U_hat
    UZ_hat
    V_hat
    VZ_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat
    ρ_hat

    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end

struct GaussianPosterior <: Posterior 
    U
    UZ
    V
    VZ
    D
    σ
    σU
    σV
    ρ

    U_hat
    UZ_hat
    V_hat
    VZ_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat
    ρ_hat

    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end

struct MaternPosterior <: Posterior 
    U
    UZ
    V
    VZ
    D
    σ
    σU
    σV
    ρU
    ρV
    ν

    U_hat
    UZ_hat
    V_hat
    VZ_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat
    ρU_hat
    ρV_hat
    ν_hat

    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end


function Posterior(data::IdentityData, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post)

    # lU = reduce(hcat, [[norm(U_post[:,i,j]) for i in axes(U_post,2)] for j in axes(U_post,3)])
    # lV = reduce(hcat, [[norm(V_post[:,i,j]) for i in axes(V_post,2)] for j in axes(V_post,3)])

    # l_corr = lU .* lV
    # D_corr = D_post .* l_corr

    # Ubar = permutedims(cat(dims = 3, [U_post[j,:,:] ./ lU for j in axes(U_post,1)]...), (3,1,2))
    # Vbar = permutedims(cat(dims = 3, [V_post[j,:,:] ./ lV for j in axes(V_post,1)]...), (3,1,2))

    U_hat = mean(U_post, dims = 3)[:,:,1]
    # U_hat = mean(Ubar, dims = 3)[:,:,1]
    UZ_hat = mean(UZ_post, dims = 3)[:,:,1]
    V_hat = mean(V_post, dims = 3)[:,:,1]
    # V_hat = mean(Vbar, dims = 3)[:,:,1]
    VZ_hat = mean(VZ_post, dims = 3)[:,:,1]
    D_hat = mean(D_post, dims = 2)[:,1]
    # D_hat = mean(D_corr, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)

    U_hpd = hcat(collect.([hpd(U_post[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(V_post[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_post[k, :]) for k in 1:data.k])...)
    # U_hpd = hcat(collect.([hpd(Ubar[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    # V_hpd = hcat(collect.([hpd(Vbar[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    # D_hpd = hcat(collect.([hpd(D_corr[k, :]) for k in 1:data.k])...)

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    IdentityPosterior(U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post,
        U_hat, UZ_hat, V_hat, VZ_hat, D_hat, σ_hat, σU_hat, σV_hat,
        U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)
    # IdentityPosterior(Ubar, UZ_post, Vbar, VZ_post, D_corr, σ_post, σU_post, σV_post,
    #     U_hat, UZ_hat, V_hat, VZ_hat, D_hat, σ_hat, σU_hat, σV_hat,
    #     U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)
    
end

function Posterior(data::ExponentialData, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρ_post)

    lU = reduce(hcat, [[norm(U_post[:,i,j]) for i in axes(U_post,2)] for j in axes(U_post,3)])
    lV = reduce(hcat, [[norm(V_post[:,i,j]) for i in axes(V_post,2)] for j in axes(V_post,3)])

    l_corr = lU .* lV
    D_corr = D_post .* l_corr

    Ubar = permutedims(cat(dims = 3, [U_post[j,:,:] ./ lU for j in axes(U_post,1)]...), (3,1,2))
    Vbar = permutedims(cat(dims = 3, [V_post[j,:,:] ./ lV for j in axes(V_post,1)]...), (3,1,2))

    U_hat = mean(Ubar, dims = 3)[:,:,1]
    UZ_hat = mean(UZ_post, dims = 3)[:,:,1]
    V_hat = mean(Vbar, dims = 3)[:,:,1]
    VZ_hat = mean(VZ_post, dims = 3)[:,:,1]
    D_hat = mean(D_corr, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)
    ρ_hat = mean(ρ_post)

    U_hpd = hcat(collect.([hpd(Ubar[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(Vbar[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_corr[k, :]) for k in 1:data.k])...)

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    ExponentialPosterior(Ubar, UZ_post, Vbar, VZ_post, D_corr, σ_post, σU_post, σV_post, ρ_post,
        U_hat, UZ_hat, V_hat, VZ_hat, D_hat, σ_hat, σU_hat, σV_hat, ρ_hat,
        U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)
    
end

function Posterior(data::GaussianData, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρ_post)

    lU = reduce(hcat, [[norm(U_post[:,i,j]) for i in axes(U_post,2)] for j in axes(U_post,3)])
    lV = reduce(hcat, [[norm(V_post[:,i,j]) for i in axes(V_post,2)] for j in axes(V_post,3)])

    l_corr = lU .* lV
    D_corr = D_post .* l_corr

    # Ubar = permutedims(cat(dims = 3, [U_post[j,:,:] ./ lU for j in axes(U_post,1)]...), (3,1,2))
    # Vbar = permutedims(cat(dims = 3, [V_post[j,:,:] ./ lV for j in axes(V_post,1)]...), (3,1,2))
    Ubar = cat(dims = 3, [U_post[:,:,j] ./ lU[:,j]' for j in axes(U_post,3)]...)
    Vbar = cat(dims = 3, [V_post[:,:,j] ./ lV[:,j]' for j in axes(V_post,3)]...)

    U_hat = mean(Ubar, dims = 3)[:,:,1]
    UZ_hat = mean(UZ_post, dims = 3)[:,:,1]
    V_hat = mean(Vbar, dims = 3)[:,:,1]
    VZ_hat = mean(VZ_post, dims = 3)[:,:,1]
    D_hat = mean(D_corr, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)
    ρ_hat = mean(ρ_post)

    U_hpd = hcat(collect.([hpd(Ubar[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(Vbar[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_corr[k, :]) for k in 1:data.k])...)

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    GaussianPosterior(Ubar, UZ_post, Vbar, VZ_post, D_corr, σ_post, σU_post, σV_post, ρ_post,
        U_hat, UZ_hat, V_hat, VZ_hat, D_hat, σ_hat, σU_hat, σV_hat, ρ_hat,
        U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)
    
end

function Posterior(data::MaternData, U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post, ν_post)

    
    U_hat = mean(U_post, dims = 3)[:,:,1]
    UZ_hat = mean(UZ_post, dims = 3)[:,:,1]
    V_hat = mean(V_post, dims = 3)[:,:,1]
    VZ_hat = mean(VZ_post, dims = 3)[:,:,1]
    D_hat = mean(D_post, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)
    ρU_hat = mean(ρU_post)
    ρV_hat = mean(ρV_post)
    ν_hat = mean(ν_post)

    U_hpd = hcat(collect.([hpd(U_post[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(V_post[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_post[k, :]) for k in 1:data.k])...)

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    MaternPosterior(U_post, UZ_post, V_post, VZ_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post, ν_post,
        U_hat, UZ_hat, V_hat, VZ_hat, D_hat, σ_hat, σU_hat, σV_hat, ρU_hat, ρV_hat, ν_hat,
        U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)

    
end



Plots.@recipe function f(p::Posterior, x; basis = 'U')
    seriestype  :=  :path
    linewidth --> 1
    legend --> :outerright
    fillalpha --> 0.4
    if basis == 'U'
        label --> ["U" * string(i) * " Post" for i in (1:size(p.U_hat,2))']
        ribbon --> (p.U_hat .- p.U_lower, p.U_upper .- p.U_hat)
    else
        label --> ["V" * string(i) * " Post" for i in (1:size(p.U_hat,2))']
        ribbon --> (p.V_hat .- p.V_lower, p.V_upper .- p.V_hat)
    end
    if basis == 'U'
        return x, p.U_hat
    elseif basis == 'V'
        return x, p.V_hat
    else
        throw(error("Need to specify basis = 'U' or basis = 'V'."))
    end
    
end

Plots.@recipe function f(p::Posterior, x, y; basis = 'U')
    # seriestype  :=  :heatmap
    layout := @layout[grid(2, size(p.U_hat, 2))]

    if basis == 'U'
        Bmean = p.U_hat
        Bvar = var(p.U, dims = 3)[:,:,1]
        labelname = ["U" * string(i) * " Mean" for i in (1:size(p.U_hat,2))']
        labelnamevar = ["U" * string(i) * " Variance" for i in (1:size(p.U_hat,2))']
    else
        Bmean = p.V_hat
        Bvar = var(p.V, dims = 3)[:,:,1]
        labelname = ["V" * string(i) * " Mean" for i in (1:size(p.V_hat,2))']
        labelnamevar = ["V" * string(i) * " Variance" for i in (1:size(p.V_hat,2))']
    end

    j = 0
    for i in 1:size(p.U_hat, 2)
        @series begin
            j = j+1
            seriestype := :heatmap
            subplot := j
            c --> :balance
            clim --> (-maximum(extrema(Bmean)), maximum(extrema(Bmean)))
            title --> labelname[i]
            x, y, reshape(Bmean[:,i], length(x), length(y))
        end
    end
    for i in 1:size(p.U_hat, 2)
        @series begin
            j = j+1
            seriestype := :heatmap
            subplot := j
            c --> :OrRd
            clim --> (0, maximum(extrema(Bvar[:,i])))
            title --> labelnamevar[i]
            x, y, reshape(Bvar[:,i], length(x), length(y))
        end
    end
    
end