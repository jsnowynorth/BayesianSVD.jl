
########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 01-January-2024
#### Description: Process the t2m results
########################################################################

########################################################################
#### Load packages
########################################################################
#region

using BayesianSVD
using Distances
using Missings
using Statistics
using DataFrames, DataFramesMeta, Chain
using NetCDF, Dates
using JLD2
using Plots
using LinearAlgebra

using CairoMakie, GeoMakie, GeoMakie.GeoJSON

include("../NERSC/anomolyFunctions.jl")


#endregion


########################################################################
#### Define plotting theme
########################################################################
#region

bold_theme = Theme(
    Axis = (
        linewidth = 5,
        titlesize = 20,
        xticklabelsize = 20, 
        yticklabelsize = 20, 
        titlefont = :bold, 
        xticklabelfont = :bold, 
        yticklabelfont = :bold,
    );
    Colorbar = (
        ticklabelsize = 18,
        ticklabelfont = :bold,
    )
)

geo = GeoJSON.read(read("data/custom.geo.json", String))

# https://eric.clst.org/tech/usgeojson/
stateBoundaries = GeoJSON.read(read("data/gz_2010_us_040_00_500k.json", String))

#endregion


########################################################################
#### load data
########################################################################
#region

fileName = "data/ERA5t2m.nc"

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
#### set up spatial and model parameters
########################################################################
#region

Nx, Ny, Nt = size(anomalyAll) # get dimensions

locs = reduce(hcat,reshape([[x, y] for x = lon, y = lat], Nx * Ny)) # sort locations

t = convert(Vector{Float64}, Vector(1:Nt)) # time index
Z = convert(Matrix{Float64}, reshape(anomalyAll, Nx*Ny, Nt)) # orient data

k = 10 # rank or number of basis functions


#endregion


########################################################################
#### deterministic SVD
########################################################################
#region


svdZ = svd(Z)
svdU = reshape(svdZ.U[:,1:k], Nx, Ny, k)
svdV = svdZ.V[:,1:k]

(cumsum(svdZ.S.^2) ./ sum(svdZ.S.^2))[1:10]


# using DelimitedFiles
# writedlm("data/Ut2m.csv", hcat(locs', svdZ.U[:,1:50]), ',')


#endregion



########################################################################
#### Plot of the data and a decomposition
########################################################################
#region

dtsnewind = T .> DateTime(2010,01,01)
dtsnew = T[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

dtstart = "2021-05"
dateStartInd = (1:Nt)[floor.(T, Dates.Month) .== DateTime(dtstart)][1]

function dataPlot()

    dataRange = round.((-1.01, 1.01) .* maximum(abs, anomalyAll[:,:,(dateStartInd):(dateStartInd+4)]), digits = 3)
    nsteps = 20
    nticks = 7

    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [GeoAxis(fig[i, j], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for j in 1:2, i in 1:2]

    cPlt = CairoMakie.contourf!(ax1[1], lon, lat, anomalyAll[:,:,dateStartInd], 
            colormap = :balance, colorrange = dataRange, 
            levels = range(dataRange[1], dataRange[2], step = (dataRange[2]-dataRange[1])/nsteps))

    for (i, axis) in enumerate(ax1)
        CairoMakie.contourf!(axis, lon, lat, anomalyAll[:,:,i + dateStartInd - 1], 
            colormap = :balance, colorrange = dataRange, 
            levels = range(dataRange[1], dataRange[2], step = (dataRange[2]-dataRange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        axis.title = Dates.format(T[i + dateStartInd - 1], "yyyy-mm")
    end

    CairoMakie.Colorbar(fig[1:2,3], cPlt, ticks = round.(range(dataRange[1], dataRange[2], length = nticks), digits = 3), height = Relative(0.82))
    
    hidedecorations!.(ax1)

    fig

end

g = with_theme(dataPlot, bold_theme)
# save("figures/dataPlot.png", g)


#endregion




########################################################################
#### load in BSVD results
########################################################################
#region

### to load in multiple files
@load "data/ERA5Samples/run_6.jld2" data pars posterior
P = posterior
for i in 7:10
    @load "data/ERA5Samples/run_" * string(i) * ".jld2" data pars posterior
    P = vcat(P, posterior)
end

posterior = Posterior(data, P)


#endregion




########################################################################
#### Plot showing the difference in deterministic vs probabilistic U ests
########################################################################
#region


function UDiffPlots()

    Nx, Ny, k = size(svdU)
    Nsamps = size(posterior.U, 3)

    Upost = reshape(posterior.U_hat, Nx, Ny, k)

    UPostDiff = [posterior.U[:,:,i] .- reshape(svdU[:,:,1:k], Nx*Ny,:) for i in axes(posterior.U, 3)]
    UPostDiffMean = reshape(mean(UPostDiff), Nx, Ny, k)

    UPostDiff = reshape(reduce(hcat, UPostDiff), Nx*Ny, k, Nsamps)

    lQ = [quantile(UPostDiff[i,j,:], 0.025) for i in axes(UPostDiff, 1), j in axes(UPostDiff, 2)]
    uQ = [quantile(UPostDiff[i,j,:], 0.975) for i in axes(UPostDiff, 1), j in axes(UPostDiff, 2)]
    PostDiffCI = [lQ[i,j] < 0 < uQ[i,j] for i in axes(uQ, 1), j in axes(uQ, 2)]


    crange = round.((-1.01, 1.01) .* maximum([maximum(abs, Upost), maximum(abs, svdU)]), digits = 3)
    crangeDiff = (-1.01, 1.01) .* maximum(abs, UPostDiffMean)
    nsteps = 31
    nticks = 5


    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [GeoAxis(fig[i, 1], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax2 = [GeoAxis(fig[i, 2], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax3 = [GeoAxis(fig[i, 3], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]

    pA = CairoMakie.contourf!(ax1[1], lon, lat, svdU[:,:, 1+ basis_start], 
        colormap = :balance, colorrange = crange, 
        levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

    # ax1 - algorithmic basis functions
    for (i, axis) in enumerate(ax1)
        CairoMakie.contourf!(axis, lon, lat, svdU[:,:,i + basis_start], 
            colormap = :balance, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        # axis.title = "Algorithmic U Basis Function " * string(i)
    end

    # ax2 - probabilistic basis functions
    for (i, axis) in enumerate(ax2)
        CairoMakie.contourf!(axis, lon, lat, Upost[:,:,i + basis_start], 
            colormap = :balance, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        # axis.title = "Probabilistic U Basis Function " * string(i)
    end

    # ax3 - posterior difference
    for (i, axis) in enumerate(ax3)

        crangeDiff = round.((-1.01, 1.01) .* maximum(abs, UPostDiffMean[:,:,i + basis_start]), digits = 3)

        p = CairoMakie.contourf!(axis, lon, lat, UPostDiffMean[:,:,i + basis_start], 
            colormap = :balance, colorrange = crangeDiff, 
            levels = range(crangeDiff[1], crangeDiff[2], step = (crangeDiff[2]-crangeDiff[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        # axis.title = "U Basis Function " * string(i)
        CairoMakie.Colorbar(fig[i,4], p, ticks = round.(range(crangeDiff[1], crangeDiff[2], length = nticks), digits = 3))

        CairoMakie.scatter!(axis, locs[1,.!PostDiffCI[:,i + basis_start]], locs[2,.!PostDiffCI[:,i + basis_start]], color = (:black, 0.4), marker = 'x')
    end


    CairoMakie.Colorbar(fig[6,1:2], pA, ticks = round.(range(crange[1], crange[2], length = nticks), digits = 3), vertical = false)


    hidedecorations!.(ax1[1:5])
    hidedecorations!.(ax2[1:5])
    hidedecorations!.(ax3[1:5])
    fig

end

basis_start = 0
g = with_theme(UDiffPlots, bold_theme)
# save("figures/Ubasis15.png", g)


basis_start = 5
g = with_theme(UDiffPlots, bold_theme)
save("figures/Ubasis610.png", g)


#endregion


########################################################################
#### Plot showing the difference in deterministic vs probabilistic V ests
########################################################################
#region


function VDiffPlots()

    dtsnewind = T .> DateTime(2010,01,01)
    dtsnew = T[dtsnewind]
    dtsticks = Dates.format.(dtsnew, "yyyy-mm")
    tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

    # dateStarInd = dtsticks .== dtstar
    # dateStarIndLong = floor.(T, Dates.Month) .== DateTime(dtstar)

    Nt, k = size(svdV)
    Nsamps = size(posterior.V, 3)

    Vpost = reshape(posterior.V_hat, Nt, k)

    VPostDiff = [posterior.V[:,:,i] .- svdV[:,1:k] for i in axes(posterior.V, 3)]
    # VPostDiffMean = reshape(mean(VPostDiff), Nt, k)

    VPostDiff = reshape(reduce(hcat, VPostDiff), Nt, k, Nsamps)

    lQDiff = [quantile(VPostDiff[i,j,:], 0.025) for i in axes(VPostDiff, 1), j in axes(VPostDiff, 2)]
    uQDiff = [quantile(VPostDiff[i,j,:], 0.975) for i in axes(VPostDiff, 1), j in axes(VPostDiff, 2)]
    PostDiffCI = [lQDiff[i,j] < 0 < uQDiff[i,j] for i in axes(lQDiff, 1), j in axes(lQDiff, 2)]

    lQ = [quantile(posterior.V[i,j,:], 0.025) for i in axes(posterior.V, 1), j in axes(posterior.V, 2)]
    uQ = [quantile(posterior.V[i,j,:], 0.975) for i in axes(posterior.V, 1), j in axes(posterior.V, 2)]


    crange = round.((-1.01, 1.01) .* maximum([maximum(abs, Vpost), maximum(abs, svdV)]), digits = 3)

    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [CairoMakie.Axis(fig[i, 1], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((1, length(dtsnew)), (crange))) for i in 1:5]
    
    # ax1 - algorithmic basis functions
    for (i, axis) in enumerate(ax1)
        CairoMakie.lines!(axis, 1:sum(dtsnewind), svdV[dtsnewind, i + basis_start], color = :black, label = false, linewidth = 1)
        CairoMakie.lines!(axis, 1:sum(dtsnewind), Vpost[dtsnewind, i + basis_start], color = :blue, label = false, linewidth = 1)
        CairoMakie.band!(axis, 1:sum(dtsnewind), lQ[dtsnewind, i + basis_start], uQ[dtsnewind, i + basis_start], color = (:blue, 0.5), label = false, linealpha = 0)
        CairoMakie.vlines!(axis, (1:sum(dtsnewind))[.!PostDiffCI[dtsnewind, i + basis_start]], ymin = -1, ymax = 1, color = (:red, 0.5))
        # CairoMakie.scatter!(axis, (1:sum(dtsnewind))[dateStarInd], svdV[dateStarIndLong, i + basis_start], marker = :star6, markersize = 20, color = :red)
    end

    fig

end


# dtstar = "2021-06"

basis_start = 0
g = with_theme(VDiffPlots, bold_theme)
# save("figures/Vbasis15.png", g)

basis_start = 5
g = with_theme(VDiffPlots, bold_theme)
# save("figures/Vbasis610.png", g)


#endregion










################################
######### Scratch work - not run
################################





########################################################################
#### anomoly plots
########################################################################
#region


dtsInd = T .> DateTime(2020,01,01)
dtsSelect = T[dtsInd]

Nsamps = size(posterior.V, 3)


AnomolyPost = [posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[dtsInd,:,i]' for i in axes(posterior.U, 3)]
AnomolyPost = reshape(reduce(hcat, AnomolyPost), Nx*Ny, sum(dtsInd), size(posterior.U, 3))

ObsAnomoly = reshape(anomalyAll[:,:,dtsInd], Nx*Ny, sum(dtsInd))

AnomolyMean = mean(AnomolyPost, dims = 3)[:,:,1]
AnomlQ = [quantile(AnomolyPost[i,j,:], 0.025) for i in axes(AnomolyPost, 1), j in axes(AnomolyPost, 2)]
AnomuQ = [quantile(AnomolyPost[i,j,:], 0.975) for i in axes(AnomolyPost, 1), j in axes(AnomolyPost, 2)]
AnomCI = [AnomlQ[i,j] < ObsAnomoly[i,j] < AnomuQ[i,j] for i in axes(AnomuQ, 1), j in axes(AnomuQ, 2)]


AnomObs = reshape(ObsAnomoly, Nx, Ny, sum(dtsInd))
AnomMean = reshape(AnomolyMean, Nx, Ny, sum(dtsInd))
AnomSD = reshape(sqrt.(var(AnomolyPost, dims = 3)[:,:,1]), Nx, Ny, sum(dtsInd))

# i=16

# Plots.contourf(reshape(AnomolyMean[:,i], Nx, Ny))
# Plots.contourf(anomalyAll[:,:,dtsInd][:,:,i])

function AnomolyPlots()

    crange = round.(1.01 .* (minimum([minimum(AnomMean[:,(start_time+1):(start_time+5),:]), minimum(AnomObs[:,:,(start_time+1):(start_time+5)])]), maximum([maximum(AnomMean[:,(start_time+1):(start_time+5),:]), maximum(AnomObs[:,:,(start_time+1):(start_time+5)])])), digits = 3)
    # crangeDiff = (-1.01, 1.01) .* maximum(abs, EPostDiffMean[:,:,(start_time+1):(start_time+5)])
    crangeSD = round.(extrema(AnomSD[:,:,(start_time+1):(start_time+5)]), digits = 3)
    nsteps = 20
    nticks = 7

    na_coasts = "/Users/JSNorth/Desktop/custom.geo.json"
    geo = GeoJSON.read(read(na_coasts, String))


    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [GeoAxis(fig[i, 1], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax2 = [GeoAxis(fig[i, 2], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax3 = [GeoAxis(fig[i, 3], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]

    pA = CairoMakie.contourf!(ax1[1], lon, lat, AnomObs[:,:,1 + start_time], 
        colormap = :Oranges_9, colorrange = crange, 
        levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

    # ax1 - algorithmic basis functions
    for (i, axis) in enumerate(ax1)
        CairoMakie.contourf!(axis, lon, lat, AnomObs[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        axis.title = "Observed " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end

    # ax2 - probabilistic basis functions
    for (i, axis) in enumerate(ax2)
        CairoMakie.contourf!(axis, lon, lat, AnomMean[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        axis.title = "Ensemble Mean " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end

    # ax3 - posterior difference
    for (i, axis) in enumerate(ax3)

        # crangeDiff = round.((-1.01, 1.01) .* maximum(abs, UPostDiffMean[:,:,i]), digits = 4)
        crangeSD = round.(extrema(AnomSD[:,:,(i+start_time)]), digits = 4)

        p = CairoMakie.contourf!(axis, lon, lat, AnomSD[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crangeSD, 
            levels = range(crangeSD[1], crangeSD[2], step = (crangeSD[2]-crangeSD[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        # axis.title = "U Basis Function " * string(i)
        CairoMakie.Colorbar(fig[i,4], p, ticks = round.(range(crangeSD[1], crangeSD[2], length = nticks), digits = 4))

        # CairoMakie.scatter!(axis, locs[1,.!EPostDiffCI[:,i + start_time]], locs[2,.!EPostDiffCI[:,i + start_time]], color = (:black, 0.4), marker = 'x')

        axis.title = "Ensemble SD " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end


    CairoMakie.Colorbar(fig[6,1:2], pA, ticks = round.(range(crange[1], crange[2], length = nticks), digits = 3), vertical = false)


    hidedecorations!.(ax1[1:5])
    hidedecorations!.(ax2[1:5])
    hidedecorations!.(ax3[1:5])
    fig

end

start_time = 16

g = with_theme(AnomolyPlots, bold_theme)



#endregion




########################################################################
#### heat ensemble plots
########################################################################
#region

# t2mCentered, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon)
# reconstructSurface(t2mCentered, t2mMonMean, t2mWeightedMean, betas, lon, lat)

# anomalyAll, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon);

dtsInd = T .> DateTime(2020,01,01)
dtsSelect = T[dtsInd]
# dtsticks = Dates.format.(dtsnew, "yyyy-mm")
# tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

# i=1
# reconstructSurface(reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]', Nx, Ny, Nt), t2mMonMean, t2mWeightedMean, betas, lat, lon) # for full time series
# reconstructSurface(reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[dtsInd,:,i]', Nx, Ny, sum(dtsInd)), t2mMonMean, t2mWeightedMean, betas, lat, lon) # for subset time series

Nsamps = size(posterior.V, 3)

t2mObs = reconstructSurface(anomalyAll[:,:,dtsInd], t2mMonMean, t2mWeightedMean, betas, lat, lon)
t2mObs = reshape(t2mObs, :, sum(dtsInd))


# latent mean distribution
EnsemblePost = [reshape(reconstructSurface(reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[dtsInd,:,i]', Nx, Ny, sum(dtsInd)), t2mMonMean, t2mWeightedMean, betas, lat, lon), Nx*Ny, sum(dtsInd)) for i in axes(posterior.U, 3)]

# posterior predictive distribution
using Distributions
PostPredVals = [rand(Normal(0, sqrt(posterior.σ[i]))) for i in axes(posterior.U, 3)] # posterior predictive distribution
EnsemblePost = [reshape(reconstructSurface(reshape(posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[dtsInd,:,i]' .+ PostPredVals[i], Nx, Ny, sum(dtsInd)), t2mMonMean, t2mWeightedMean, betas, lat, lon), Nx*Ny, sum(dtsInd)) for i in axes(posterior.U, 3)]

EnsemblePost = reshape(reduce(hcat, EnsemblePost), Nx*Ny, sum(dtsInd), size(posterior.U, 3))


Emean = mean(EnsemblePost, dims = 3)[:,:,1]
ElQ = [quantile(EnsemblePost[i,j,:], 0.025) for i in axes(EnsemblePost, 1), j in axes(EnsemblePost, 2)]
EuQ = [quantile(EnsemblePost[i,j,:], 0.975) for i in axes(EnsemblePost, 1), j in axes(EnsemblePost, 2)]
ECI = [ElQ[i,j] < t2mObs[i,j] < EuQ[i,j] for i in axes(EuQ, 1), j in axes(EuQ, 2)]



EPostDiff = [EnsemblePost[:,:,i] .- t2mObs for i in axes(EnsemblePost, 3)]
EPostDiffMean = reshape(mean(EPostDiff), Nx, Ny, sum(dtsInd))
EPostDiff = reshape(reduce(hcat, EPostDiff), Nx*Ny, sum(dtsInd), Nsamps)

lQ = [quantile(EPostDiff[i,j,:], 0.025) for i in axes(EPostDiff, 1), j in axes(EPostDiff, 2)]
uQ = [quantile(EPostDiff[i,j,:], 0.975) for i in axes(EPostDiff, 1), j in axes(EPostDiff, 2)]
EPostDiffCI = [lQ[i,j] < 0 < uQ[i,j] for i in axes(uQ, 1), j in axes(uQ, 2)]
# Plots.contourf(reshape(Emean[:,1], Nx, Ny))
# Plots.contourf(reshape(t2mObs[:,1], Nx, Ny))



t2mObs = reshape(t2mObs, Nx, Ny, sum(dtsInd))
Emean = reshape(Emean, Nx, Ny, sum(dtsInd))
ESD = reshape(sqrt.(var(EnsemblePost, dims = 3)[:,:,1]), Nx, Ny, sum(dtsInd))


function EnsemblePlots()

    crange = round.(1.01 .* (minimum([minimum(EnsemblePost[:,(start_time+1):(start_time+5),:]), minimum(t2mObs[:,:,(start_time+1):(start_time+5)])]), maximum([maximum(EnsemblePost[:,(start_time+1):(start_time+5),:]), maximum(t2mObs[:,:,(start_time+1):(start_time+5)])])), digits = 3)
    # crangeDiff = (-1.01, 1.01) .* maximum(abs, EPostDiffMean[:,:,(start_time+1):(start_time+5)])
    crangeSD = round.(extrema(ESD[:,:,(start_time+1):(start_time+5)]), digits = 3)
    nsteps = 20
    nticks = 7

    na_coasts = "/Users/JSNorth/Desktop/custom.geo.json"
    geo = GeoJSON.read(read(na_coasts, String))

    # https://eric.clst.org/tech/usgeojson/
    stateBoundaries = GeoJSON.read(read("/Users/JSNorth/Desktop/gz_2010_us_040_00_500k.json", String))


    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [GeoAxis(fig[i, 1], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax2 = [GeoAxis(fig[i, 2], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]
    ax3 = [GeoAxis(fig[i, 3], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for i in 1:5]

    pA = CairoMakie.contourf!(ax1[1], lon, lat, t2mObs[:,:,1 + start_time], 
        colormap = :Oranges_9, colorrange = crange, 
        levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))

    # ax1 - algorithmic basis functions
    for (i, axis) in enumerate(ax1)
        CairoMakie.contourf!(axis, lon, lat, t2mObs[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        axis.title = "Observed " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end

    # ax2 - probabilistic basis functions
    for (i, axis) in enumerate(ax2)
        CairoMakie.contourf!(axis, lon, lat, Emean[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crange, 
            levels = range(crange[1], crange[2], step = (crange[2]-crange[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        axis.title = "Ensemble Mean " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end

    # ax3 - posterior difference
    for (i, axis) in enumerate(ax3)

        # crangeDiff = round.((-1.01, 1.01) .* maximum(abs, UPostDiffMean[:,:,i]), digits = 4)
        crangeSD = round.(extrema(ESD[:,:,(i+start_time)]), digits = 4)

        p = CairoMakie.contourf!(axis, lon, lat, ESD[:,:,i + start_time], 
            colormap = :Oranges_9, colorrange = crangeSD, 
            levels = range(crangeSD[1], crangeSD[2], step = (crangeSD[2]-crangeSD[1])/nsteps))
        poly!(axis, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        poly!(axis, stateBoundaries; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
        # axis.title = "U Basis Function " * string(i)
        CairoMakie.Colorbar(fig[i,4], p, ticks = round.(range(crangeSD[1], crangeSD[2], length = nticks), digits = 4))

        # CairoMakie.scatter!(axis, locs[1,.!EPostDiffCI[:,i + start_time]], locs[2,.!EPostDiffCI[:,i + start_time]], color = (:black, 0.4), marker = 'x')

        axis.title = "Ensemble SD " * Dates.format(dtsSelect[i+start_time], "yyyy-mm")
    end


    CairoMakie.Colorbar(fig[6,1:2], pA, ticks = round.(range(crange[1], crange[2], length = nticks), digits = 3), vertical = false)


    hidedecorations!.(ax1[1:5])
    hidedecorations!.(ax2[1:5])
    hidedecorations!.(ax3[1:5])
    fig

end

start_time = 16

g = with_theme(EnsemblePlots, bold_theme)
save("/Users/JSNorth/Desktop/ERA5Results/EnsembleMaySep2021.png", g)


#endregion


########################################################################
#### covariance figures
########################################################################
#region


# plot the correlation instead of covariance

Cest = [(1/(Nx*Ny-1)) * (posterior.U[:,:,i] * diagm(posterior.D[:,i] .^2) * posterior.U[:,:,i]' .+ posterior.σ[i] * I(Nx*Ny)) for i in sample(1:size(posterior.U, 3), 100, replace = false)]

Cmean = mean(Cest)

CCorMean = [Cmean[i,j] / sqrt(Cmean[i,i] * Cmean[j,j]) for i in axes(Cmean,1), j in axes(Cmean, 2)]


Plots.contourf(Cmean)
Plots.contourf(CCorMean)

loc = 1375
locs[:,loc]
Plots.contourf(lon, reverse(lat), (reshape(CCorMean[:, loc], Nx, Ny)')[end:-1:1,:], c = :oxy)
Plots.scatter!([locs[1,loc]], [locs[2,loc]], c = :blue, label = false, markersize = 5)


cov(reshape(anomalyAll, Nx*Ny, Nt), dims = 2)

# Plots.contourf(lon, reverse(lat), (reshape(cor(reshape(anomalyAll, Nx*Ny, Nt), dims = 2)[:, loc], Nx, Ny)')[end:-1:1,:], c = :oxy)
# Plots.scatter!([locs[1,loc]], [locs[2,loc]], c = :blue, label = false, markersize = 5)


loc = 1420
locs[:,loc]
Plots.contourf(lon, reverse(lat), (reshape(CCorMean[:, loc], Nx, Ny)')[end:-1:1,:], c = :oxy)
Plots.scatter!([locs[1,loc]], [locs[2,loc]], c = :blue, label = false, markersize = 5)



# plot of the variance
Plots.contourf(lon, reverse(lat), (reshape(diag(Cmean), Nx, Ny)')[end:-1:1,:], c = :oxy)

locs[:,1590]

loclist = [1375, 1420, 380]

CCorMean[:, loclist]

function CorrPlots()

    cVarrange = round.((0.99, 1.01) .* extrema(diag(Cmean)), digits = 3)
    cCorrange = round.((0.99, 1) .* extrema(CCorMean[:, loclist]), digits = 3)
    nsteps = 20
    nticks = 7

    loc1, loc2, loc3 = loclist

    na_coasts = "/Users/JSNorth/Desktop/custom.geo.json"
    geo = GeoJSON.read(read(na_coasts, String))


    fig = Figure(resolution = (1800, 600), figure_padding = 20)
    ax1 = GeoAxis(fig[1, 1], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53))
    ax2 = GeoAxis(fig[1, 2], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53))
    ax3 = GeoAxis(fig[1, 3], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53))

    p1 = CairoMakie.contourf!(ax1, lon, lat, reshape(CCorMean[:, loc1], Nx, Ny), 
        colormap = :Oranges_9, colorrange = cCorrange, 
        levels = range(cCorrange[1], cCorrange[2], step = (cCorrange[2]-cCorrange[1])/nsteps))
    CairoMakie.scatter!(ax1, [locs[1,loc1]], [locs[2,loc1]], color = :blue, markersize = 20)
    poly!(ax1, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)

    CairoMakie.contourf!(ax2, lon, lat, reshape(CCorMean[:, loc2], Nx, Ny), 
        colormap = :Oranges_9, colorrange = cCorrange, 
        levels = range(cCorrange[1], cCorrange[2], step = (cCorrange[2]-cCorrange[1])/nsteps))
    CairoMakie.scatter!(ax2, [locs[1,loc2]], [locs[2,loc2]], color = :blue, markersize = 20)
    poly!(ax2, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)

    CairoMakie.contourf!(ax3, lon, lat, reshape(CCorMean[:, loc3], Nx, Ny), 
        colormap = :Oranges_9, colorrange = cCorrange, 
        levels = range(cCorrange[1], cCorrange[2], step = (cCorrange[2]-cCorrange[1])/nsteps))
    CairoMakie.scatter!(ax3, [locs[1,loc3]], [locs[2,loc3]], color = :blue, markersize = 20)
    poly!(ax3, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)


    CairoMakie.Colorbar(fig[1, 4], p1, ticks = round.(range(cCorrange[1], cCorrange[2], length = nticks), digits = 3), height = Relative(0.72))

    hidedecorations!(ax1)
    hidedecorations!(ax2)
    hidedecorations!(ax3)

    fig



end


# loclist = [1375, 1420, 380]
loclist = [1375, 380, 1590]
g = with_theme(CorrPlots, bold_theme)
save("/Users/JSNorth/Desktop/ERA5Results/CorrPlots.png", g)




function VarPlot()
    
    cVarrange = round.((0.99, 1.01) .* extrema(diag(Cmean)), digits = 3)
    nsteps = 20
    nticks = 7

    na_coasts = "/Users/JSNorth/Desktop/custom.geo.json"
    geo = GeoJSON.read(read(na_coasts, String))

    fig = Figure(resolution = (1200, 1000), figure_padding = 35)
    ax1 = GeoAxis(fig[1, 1], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53))

    p11 = CairoMakie.contourf!(ax1, lon, lat, reshape(diag(Cmean), Nx, Ny), 
        colormap = :Oranges_9, colorrange = cVarrange, 
        levels = range(cVarrange[1], cVarrange[2], step = (cVarrange[2]-cVarrange[1])/nsteps))
    poly!(ax1, geo; strokecolor = :black, strokewidth = 1, color = (:black, 0), shading = false)
    CairoMakie.Colorbar(fig[1,2], p11, ticks = round.(range(cVarrange[1], cVarrange[2], length = nticks), digits = 3), height = Relative(0.83))

    hidedecorations!(ax1)
    fig

end

g = with_theme(VarPlot, bold_theme)
save("/Users/JSNorth/Desktop/ERA5Results/VarPlots.png", g)









#endregion