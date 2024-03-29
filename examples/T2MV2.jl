
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
using NetCDF
using Distances
using Missings
using Statistics
using DataFrames, DataFramesMeta, Chain
using Dates
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
#### set up spatial and model parameters
########################################################################
#region

Zobs = t2m .- mean(t2m)

# Zobs = Zobs ./ var(Zobs)


Nx, Ny, Nt = size(Zobs) # get dimensions

locs = reduce(hcat,reshape([[x, y] for x = lon, y = lat], Nx * Ny)) # sort locations

t = convert(Vector{Float64}, Vector(1:Nt)) # time index
Z = convert(Matrix{Float64}, reshape(Zobs, Nx*Ny, Nt)) # orient data

k = 10 # rank or number of basis functions


# moinds = mod.(1:Nt, 12)
# moinds[moinds .== 0] .= 12

# X = reduce(hcat, [moinds .== i for i in 1:12])
# X = convert(Array{Float64,2}, X)
# X[:,1] = fill(1, Nt)
# X = hcat(X, 1:Nt)

# betas = inv(X' * X) * X' * Z'

# Z = Z .- (X * betas)'




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
# writedlm("/Users/JSNorth/Desktop/lengthScales/Ut2m.csv", hcat(locs', svdZ.U[:,1:50]), ',')
# writedlm("/Users/JSNorth/Desktop/lengthScales/Vt2m.csv", hcat(t, svdZ.V[:,1:50]), ',')


size(svdU)

Plots.contourf(svdU[:,:,1], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,2], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,3], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,4], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,5], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,6], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,7], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,8], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,9], c = :balance, size = (1000, 600))
Plots.contourf(svdU[:,:,10], c = :balance, size = (1000, 600))


Plots.plot(svdZ.V[:,1], size = (1000, 600))
Plots.plot(svdZ.V[:,2], size = (1000, 600))
Plots.plot(svdZ.V[:,3], size = (1000, 600))
Plots.plot(svdZ.V[:,4], size = (1000, 600))
Plots.plot(svdZ.V[:,5], size = (1000, 600))

#endregion


########################################################################
#### run BSVD
########################################################################
#region

Nx, Ny, Nt = size(Zobs)

locs = reduce(hcat,reshape([[x, y] for x = lon, y = lat], Nx * Ny))

t = convert(Vector{Float64}, Vector(1:Nt))
Z = convert(Matrix{Float64}, reshape(anomalyAll, Nx*Ny, Nt))

k = 10
ΩU = MaternCorrelation(lon, lat, ρ = 400, ν = 3.5, metric = Haversine(6371))
ΩV = GaussianCorrelation(t, ρ = 3)

data = Data(Z, locs, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 100, burnin = 50)

# jldsave("../BSVDresults/run_1.jld2"; data, pars, posterior)
# data, pars, posterior = jldopen("../results/PDOResults/PDO.jld2")
# @save "../BSVDresults/run_1.jld2" data pars posterior



Plots.plot(posterior.ρU')
Plots.plot(posterior.ρV')

mean(posterior.ρU, dims = 2)
mean(posterior.ρV, dims = 2)

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

    dataRange = round.((-1.01, 1.01) .* maximum(abs, Zobs[:,:,(dateStartInd):(dateStartInd+4)]), digits = 3)
    nsteps = 20
    nticks = 7

    fig = Figure(resolution = (1200, 1200), figure_padding = 35)
    ax1 = [GeoAxis(fig[i, j], dest = "+proj=wintri +lon_0=-122", lonlims=(-128, -116), latlims = (44, 53)) for j in 1:2, i in 1:2]

    cPlt = CairoMakie.contourf!(ax1[1], lon, lat, Zobs[:,:,dateStartInd], 
            colormap = :balance, colorrange = dataRange, 
            levels = range(dataRange[1], dataRange[2], step = (dataRange[2]-dataRange[1])/nsteps))

    for (i, axis) in enumerate(ax1)
        CairoMakie.contourf!(axis, lon, lat, Zobs[:,:,i + dateStartInd - 1], 
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
# save("figures/Ubasis610.png", g)


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









