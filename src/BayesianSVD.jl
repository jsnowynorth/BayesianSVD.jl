
######################################################################
#### Joshua North
#### Bayesian Basis Functions
######################################################################

__precompile__()

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

######################################################################
#### Source Files
######################################################################
include("../src/HelperFunctions.jl")
include("../src/CorrelationFunctions.jl")
include("../src/SimulationFunctions.jl")
include("../src/DataClass.jl")
include("../src/ParameterClass.jl")
include("../src/PosteriorClass.jl")
include("../src/SamplingFunctions.jl")

export 
    # overall sample function
    PON,
    GenerateData,

    # helper functions
    posteriorCoverage,
    hpd,
    CreateDesignMatrix,

    # overall sample function
    SampleSVD,
    SampleSVDGrouped,

    # Correlation Kernels
    IdentityCorrelation,
    ExponentialCorrelation,
    GaussianCorrelation,
    MaternCorrelation,
    SparseCorrelation,
    ARCorrelation,
    Correlation,
    IndependentCorrelation,
    DependentCorrelation,

    # Parameter and Data
    Pars,
    Data,

    # Posterior concat function
    Posterior

end