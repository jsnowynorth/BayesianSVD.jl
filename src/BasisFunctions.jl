######################################################################
#### Joshua North
#### Bayesian Basis Functions
######################################################################


######################################################################
#### Load Packages
######################################################################
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