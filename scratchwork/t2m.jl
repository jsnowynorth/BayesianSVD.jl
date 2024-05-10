########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 20-December-2023
#### Description: File to start sampling the t2m basis functions
########################################################################

#### data info

# Monthly maximum two meter air temperature
# longitude: -128 to -116
# latitude: 44 to 53
# time: 1979-01 to 2021-12

########################################################################
#### load packages
########################################################################
#region

using Distances
using Missings
using Statistics
using LinearAlgebra
using DataFrames, DataFramesMeta, Chain
using NetCDF, Dates
using Plots
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON
using JLD2
using BayesianSVD

# include("/pscratch/sd/j/jsnorth/ERA5MonthlyMax/Code/anomolyFunctions.jl")
include("anomolyFunctions.jl")

#endregion

########################################################################
#### load data
########################################################################
#region

# fileName = "/Users/JSNorth/Desktop/ERA5t2m.nc"
fileName = "../ERA5t2m.nc"

#### netcdf info
# ncinfo(fileName)

#### load lat, lon, and time
lat = ncread(fileName, "latitude")
lon = ncread(fileName, "longitude")
T = ncread(fileName, "time")

T = DateTime(1900,01,01,00,00,00) .+ Dates.Hour.(T)
yrmolist = Dates.yearmonth.(T)
yrmo = unique(Dates.yearmonth.(T))

#### load and reshape sst data and convert to celcius
t2m = ncread(fileName, "VAR_2T") .- 273.15

#endregion


########################################################################
#### anomoly calculation
########################################################################
#region

anomalyAll, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon);

#endregion

########################################################################
#### run BSVD
########################################################################
#region

Nx, Ny, Nt = size(anomalyAll)

locs = reduce(hcat,reshape([[x, y] for x = lon, y = lat], Nx * Ny))

t = convert(Vector{Float64}, Vector(1:Nt))
Z = convert(Matrix{Float64}, reshape(anomalyAll, Nx*Ny, Nt))

k = 10
ΩU = MaternCorrelation(lon, lat, ρ = 400, ν = 3.5, metric = Haversine(6371))
ΩV = IdentityCorrelation(t)
data = Data(Z, locs, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 100, burnin = 50)

# jldsave("../BSVDresults/run_1.jld2"; data, pars, posterior)
# data, pars, posterior = jldopen("../results/PDOResults/PDO.jld2")
@save "../BSVDresults/run_1.jld2" data pars posterior

#endregion