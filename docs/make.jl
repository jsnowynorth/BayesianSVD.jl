using BayesianSVD
using Documenter

DocMeta.setdocmeta!(BayesianSVD, :DocTestSetup, :(using BayesianSVD); recursive=true)

makedocs(;
    modules=[BayesianSVD],
    authors="Josh North <jsnowynorth@gmail.com> and contributors",
    repo="https://github.com/jsnowynorth/BayesianSVD.jl/blob/{commit}{path}#{line}",
    sitename="BayesianSVD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jsnowynorth.github.io/BayesianSVD.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jsnowynorth/BayesianSVD.jl",
    devbranch="main",
)
