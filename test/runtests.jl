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

    n = 50
    # @test n == 50
    x = range(-5, 5, n)
    ΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Distances.Euclidean())
    Random.seed!(2)
    k = 5
    Φ = PON(n, k, ΣU.K)

    @test size(Φ) == (n,k)

end

# https://julialang.org/contribute/developing_package/
# https://bjack205.github.io/tutorial/2021/07/16/julia_package_setup.html
# https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/11-developing-julia-packages


