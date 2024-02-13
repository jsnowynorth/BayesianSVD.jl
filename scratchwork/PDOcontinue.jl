
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 19-April-2023
#### Description: File to continue sampling the PDO basis functions
########################################################################

#### load packages
using BayesianSVD
using JLD2

######################################################################
#### Read In Data
######################################################################

fileNumber = parse(Int64, ARGS[1]) # should be the next number in line (i.e., if PDO_2.jl exists, then should be 3)

@load "../results/PDOResults/PDO_" * string(fileNumber - 1) * ".jld2" data pars posterior

posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 1000) # used to keep continue burnin
# posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 0) # used to sample

@save "../results/PDOResults/PDO_" * string(fileNumber) * ".jld2" data pars posterior
