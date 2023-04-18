######################################################################
#### Posterior Structures
######################################################################


"""
    Posterior

Structure of type posterior with subtypes Identity, Exponential, Gaussian, or Matern.
Contains the raw posterior samples and some means and 95% quantiles of parameters.
Plotting associated with the structure.

See also [`Pars`](@ref), [`Data`](@ref), and [`SampleSVD`](@ref).

# Examples
```
k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)

plot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple])
plot(posterior, t, size = (900, 500), basis = 'V', linewidth = 2, c = [:red :green :purple])

# for spatial basis functions, provide x and y
plot(posterior, x, y)
```
"""
abstract type Posterior end

struct MixedEffectPosterior <: Posterior

    β
    U
    V
    D
    σ
    σU
    σV
    ρU
    ρV

    β_hat
    U_hat
    V_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat
    ρU_hat
    ρV_hat

    β_lower
    β_upper
    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end

struct RandomEffectPosterior <: Posterior

    U
    V
    D
    σ
    σU
    σV
    ρU
    ρV

    U_hat
    V_hat
    D_hat
    σ_hat
    σU_hat
    σV_hat
    ρU_hat
    ρV_hat

    U_lower
    U_upper
    V_lower
    V_upper
    D_lower
    D_upper
end


function Posterior(data::MixedEffectData, β_post, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)

    β_hat = mean(β_post, dims = 2)[:,1]
    U_hat = mean(U_post, dims = 3)[:,:,1]
    V_hat = mean(V_post, dims = 3)[:,:,1]
    D_hat = mean(D_post, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)
    ρU_hat = mean(ρU_post)
    ρV_hat = mean(ρV_post)

    β_hpd = hcat(collect.([hpd(β_post[p, :]) for p in 1:data.p])...)
    U_hpd = hcat(collect.([hpd(U_post[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(V_post[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_post[k, :]) for k in 1:data.k])...)

    β_lower = β_hpd[1,:]
    β_upper = β_hpd[2,:]

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    MixedEffectPosterior(β_post, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post,
        β_hat, U_hat, V_hat, D_hat, σ_hat, σU_hat, σV_hat, ρU_hat, ρV_hat,
        β_lower, β_upper, U_lower, U_upper, V_lower, V_upper, D_lower, D_upper)

    
end

function Posterior(data::RandomEffectData, U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post)

    U_hat = mean(U_post, dims = 3)[:,:,1]
    V_hat = mean(V_post, dims = 3)[:,:,1]
    D_hat = mean(D_post, dims = 2)[:,1]
    σ_hat = mean(σ_post)
    σU_hat = mean(σU_post, dims = 2) 
    σV_hat = mean(σV_post, dims = 2)
    ρU_hat = mean(ρU_post)
    ρV_hat = mean(ρV_post)

    U_hpd = hcat(collect.([hpd(U_post[n, k, :]) for n in 1:data.n, k in 1:data.k])...)
    V_hpd = hcat(collect.([hpd(V_post[m, k, :]) for m in 1:data.m, k in 1:data.k])...)
    D_hpd = hcat(collect.([hpd(D_post[k, :]) for k in 1:data.k])...)

    U_lower = reshape(U_hpd[1,:], data.n, data.k)
    U_upper = reshape(U_hpd[2,:], data.n, data.k)

    V_lower = reshape(V_hpd[1,:], data.m, data.k)
    V_upper = reshape(V_hpd[2,:], data.m, data.k)

    D_lower = D_hpd[1,:]
    D_upper = D_hpd[2,:]

    RandomEffectPosterior(U_post, V_post, D_post, σ_post, σU_post, σV_post, ρU_post, ρV_post,
        U_hat, V_hat, D_hat, σ_hat, σU_hat, σV_hat, ρU_hat, ρV_hat,
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