
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 20-December-2023
#### Description: File to continue sampling the the real world example model
########################################################################

#### load packages
using BayesianSVD
using JLD2

######################################################################
#### Read In Data
######################################################################

fileNumber = parse(Int64, ARGS[1]) # should be the next number in line (i.e., if run_2.jl exists, then should be 3)
samplingIndicator = parse(Bool, ARGS[2]) # if true, sample from model post-burnin. if false, burnin
# nits = parse(Int64, ARGS[3]) # number of samples to draw
# burnin = parse(Int64, ARGS[4]) # number of samples to burn (when equal to nits, no samples will be saved and still in burnin phase)


@load "../BSVDresults/run_" * string(fileNumber - 1) * ".jld2" data pars posterior

if samplingIndicator
    # used to sample post-burnin
    posterior, pars = SampleSVD(data, pars; nits = parse(Int64, ARGS[3]), burnin = 0)
else
    # used to burnin
    posterior, pars = SampleSVD(data, pars; nits = parse(Int64, ARGS[3]), burnin = parse(Int64, ARGS[4]))
end

@save "../BSVDresults/run_" * string(fileNumber) * ".jld2" data pars posterior
