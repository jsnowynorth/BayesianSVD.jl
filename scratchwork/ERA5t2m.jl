########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 19-December-2023
#### Description: Basis decomposition on t2m data from ERA5
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

include("anomolyFunctions.jl")

#endregion

########################################################################
#### load data
########################################################################
#region

fileName = "/Users/JSNorth/Desktop/ERA5t2m.nc"

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
# maximum(Distances.pairwise(Haversine(6371), locs))/100
ΩU = MaternCorrelation(lon, lat, ρ = 400, ν = 3.5, metric = Haversine(6371))
ΩV = IdentityCorrelation(t)
data = Data(Z, locs, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 100, burnin = 50)


Plots.plot(posterior.ρU')


#endregion


########################################################################
#### algorithmic SVD
########################################################################
#region

svdA = svd(Z)


U = reshape(svdA.U[:,1:9], length(lon), length(lat), :)
U = reshape(posterior.U_hat, Nx, Ny, k)

crange = (-1.01, 1.01) .* maximum(abs, U)
nsteps = 20
nticks = 7

na_coasts = "/Users/JSNorth/Desktop/custom.geo.json"
geo = GeoJSON.read(read(na_coasts, String))


fig = Figure(resolution = (1800, 1200))
ax = [GeoAxis(fig[i, j], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:3, j in 1:3]
for (i, axis) in enumerate(ax)
    crange = (-1.01, 1.01) .* maximum(abs, U[:,:,i])
    CairoMakie.contourf!(axis, lon, lat, U[:,:,i], 
        colormap = :balance, colorrange = crange, 
        levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
    poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
    axis.title = "U Basis Function " * string(i)
end
fig





Vall = svdA.V[:,1:9]
Vall = posterior.V_hat[:,1:9]

# quantile(reshape(U, :), range(0, 1, step = 0.1))
crange = (-1.01, 1.01) .* maximum(abs, Vall)
nsteps = 20
nticks = 7

startdate = DateTime(1980, 06, 01) # start date
dts = [DateTime(yrmo[i][1], yrmo[i][2]) for i in axes(yrmo, 1)] # get date corresponding to each time 
dtsticks = Dates.format.(dts, "yyyy-mm")
tkmarks = (1:24:length(dts), string.(dtsticks[1:24:end]))


fig = Figure(resolution = (1800, 1200))
ax = [Axis(fig[i, j], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((1, length(dts)), (crange))) for i in 1:3, j in 1:3]

for (i, axis) in enumerate(ax)
    CairoMakie.lines!(axis, 1:516, mean(Vall[:,i,:], dims = 2)[:,1], color = :black, label = false, linewidth = 2)
    CairoMakie.vlines!(axis, 512, ymin = -1, ymax = 1, color = :red, label = false, linewidth = 2)
    CairoMakie.vlines!(axis, 505, ymin = -1, ymax = 1, color = :blue, label = false, linewidth = 2)
    CairoMakie.hlines!(axis, quantile(reshape(Vall[1:500,1,:], :), [0.025, 0.975]), xmin = 0, xymax = 130, color = :blue, label = false, linewidth = 2)
    axis.title = "V Basis Function " * string(i)
end
fig


#endregion