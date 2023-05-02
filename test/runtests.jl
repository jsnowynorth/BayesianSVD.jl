using BayesianSVD

using MultivariateStats # not sure what for
using Statistics # statistics support
using Distributions # distribution support
using LinearAlgebra # linear algebra support
using Random # random sampling support
using SpecialFunctions # for Bessel function
using Distances # compute distances

using Test


@testset "BayesianSVD.jl" begin

    # generated data parameters
    m = 100
    n = 100
    x = range(-5, 5, n)
    t = range(0, 10, m)

    ΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())


    D = [40, 30, 20, 10, 5]
    k = 5
    ϵ = 0.1
    
    # generate data
    U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ)

    # linear trend
    β = [-2, 0.6, 1.2, -0.9]
    X = rand(Normal(), n*m, length(β))
    Ztrend = Z + reshape(X*β, n, m)

    # all model parameters
    k = 5
    ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
    ΩV = IdentityCorrelation(t)

    # Random Effect only
    data = Data(Z, x, t, k)
    pars = Pars(data, ΩU, ΩV)
    posterior, pars = SampleSVD(data, pars; nits = 10, burnin = 5)
    
    # Mixed Effect
    dataMixed = Data(Z, X, x, t, k)
    parsMixed = Pars(dataMixed, ΩU, ΩV)
    posteriorMixed, parsMixed = SampleSVD(dataMixed, parsMixed; nits = 10, burnin = 5)

    @testset "Dimension Check" begin
        @test size(U) == (n,k)
        @test size(V) == (m,k)
        @test size(Y) == (n,m)
        @test size(Z) == (n,m)
    end

    @testset "Model Set-Up Check" begin
        @test typeof(ΩU) <: BayesianSVD.DependentCorrelation
        @test typeof(ΩV) <: BayesianSVD.IndependentCorrelation
        @test typeof(ΩU) <: MaternCorrelation
        @test typeof(ΩV) <: IdentityCorrelation
    end

    @testset "Random Effect Model Check" begin
        @test typeof(data) <: BayesianSVD.RandomEffectData
        @test typeof(pars) <: BayesianSVD.RandomEffectPars
        @test typeof(posterior) <: BayesianSVD.RandomEffectPosterior
        @test typeof(data) <: Data
        @test typeof(pars) <: Pars
        @test typeof(posterior) <: BayesianSVD.Posterior
    end

    @testset "Mixed Effect Model Check" begin
        @test typeof(dataMixed) <: BayesianSVD.MixedEffectData
        @test typeof(parsMixed) <: BayesianSVD.MixedEffectPars
        @test typeof(posteriorMixed) <: BayesianSVD.MixedEffectPosterior
        @test typeof(dataMixed) <: Data
        @test typeof(parsMixed) <: Pars
        @test typeof(posteriorMixed) <: BayesianSVD.Posterior
    end

end

# https://julialang.org/contribute/developing_package/
# https://bjack205.github.io/tutorial/2021/07/16/julia_package_setup.html
# https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/11-developing-julia-packages


