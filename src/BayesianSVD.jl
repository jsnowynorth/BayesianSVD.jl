
######################################################################
#### Joshua North
#### Bayesian Basis Functions
######################################################################

"""
    BayesianSVD

Here is my package.
"""
module BayesianSVD

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

MultivariateStats, Statistics, Distributions, LinearAlgebra, Plots, Random, SpecialFunctions, ProgressMeter, Distances, RecipesBase

include("../src/SimulationFunctions.jl")
include("../src/HelperFunctions.jl")
include("../src/KernelFunctions.jl")
include("../src/DataClass.jl")
include("../src/ParameterClass.jl")
include("../src/PosteriorClass.jl")
include("../src/SamplingFunctions.jl")

export 
    # overall sample function
    PON,

    # overall sample function
    SampleSVD,

    # Covariance Kernels
    IdentityKernel,
    ExponentialKernel,
    GaussianKernel,
    MaternKernel,

    # Parameter and Data
    Pars,
    Data

end
