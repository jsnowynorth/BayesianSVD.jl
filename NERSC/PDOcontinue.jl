
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

fileNumber = ARGS[1] # should be the next number in line (i.e., if PDO_2.jl exists, then should be 3)

data, pars, posterior = jldopen("../results/PDOResults/PDO_" * string(fileNumber-1) * ".jld2")


posterior, pars = SampleSVD(data, pars; nits = 500, burnin = 500) # used to keep continue burnin
posterior, pars = SampleSVD(data, pars; nits = 500, burnin = 0) # used to sample


jldsave("../results/PDOResults/PDO_" * string(fileNumber) * ".jld2"; data, pars, posterior)
# data, pars, posterior = jldopen("../results/PDOResults/PDO.jld2")



#endregion

