using BayesianSVD
using Documenter

using MultivariateStats # not sure what for
using Statistics # statistics support
using Distributions # distribution support
using LinearAlgebra # linear algebra support
using Plots # plotting support
using Random # random sampling support
using SpecialFunctions # for Bessel function
using ProgressMeter # show progress of sampler
using Distances # compute distances
using RecipesBase # special plots

DocMeta.setdocmeta!(BayesianSVD, :DocTestSetup, :(using BayesianSVD); recursive=true)

makedocs(;
    modules=[BayesianSVD],
    authors="Josh North <jsnowynorth@gmail.com> and contributors",
    repo="https://github.com/jsnowynorth/BayesianSVD.jl/blob/{commit}{path}#{line}",
    sitename="BayesianSVD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "nothing") == "true",
        canonical="https://jsnowynorth.github.io/BayesianSVD.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => ["Simulate Data" => "simulateData.md",
                        "Bayesian SVD" => "example1.md"],
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/jsnowynorth/BayesianSVD.jl.git",
    devbranch="main",
)