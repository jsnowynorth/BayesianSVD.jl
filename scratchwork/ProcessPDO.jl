
########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 04-May-2023
#### Description: Process the PDO results
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

#endregion


######################################################################
#### Read In Data
######################################################################
#region

#### source file name
# fileName = "data/HadISST_sst.nc"
fileName = "/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/data/HadISST_sst.nc"

#### netcdf info
ncinfo(fileName)

#### load lat, lon, and time
lat = ncread(fileName, "latitude")
lon = ncread(fileName, "longitude")
T = ncread(fileName, "time")

#### load and reshape sst data
sst = ncread(fileName, "sst")
sst = replace(sst, -1.0f30 => missing)
sst = replace(sst, -1000.0 => missing)
sst = reshape(permutedims(sst, (2,1,3)), length(lat), length(lon), length(T))
sst = sst[end:-1:1,:,:]

#endregion

######################################################################
#### Detrend Data
######################################################################
#region

#### functions to detrend

function rmMonAnnCyc(X)
    nx, ny, nt = size(X)
    mmean = reshape(reduce(hcat,[mean(X[:,:,(i:12:nt)], dims = 3) for i in 1:12]), nx, ny, 12)
    return mmean
end

function calcMonAnom(X, Xavg)
    nx, ny, nt = size(X)

    moinds = mod.(1:nt, 12)
    moinds[moinds .== 0] .= 12

    anom = reshape(reduce(hcat,[X[:,:,i] .- Xavg[:,:,moinds[i]] for i in 1:nt]), nx, ny, nt)
    return anom
    

end

function rmMonAnn(X)

    Xavg = rmMonAnnCyc(X)
    Xanom = calcMonAnom(X, Xavg)
    return Xanom
    
end


function areaWeight(lon, lat)

    R = 6.371e6
    RAD = π / 180

    locs = reduce(hcat,reshape([[x, y] for x = lon, y = lat], length(lon)*length(lat)))'
    
    urlocs = locs .+ 0.5
    lblocs = locs .- 0.5

    dy = abs.(urlocs[:,1] .- lblocs[:,1]) .* R
    dx = abs.(cos.(urlocs[:,2] * RAD) .- cos.(lblocs[:,2] * RAD)) .* R
    gridArea = dx .* dy
    gridWeight = gridArea ./ (4*π*R^2)

    return gridWeight
    
end

sstAnom = rmMonAnn(sst)


W = areaWeight(lon, lat)
sstWeighted = (reshape(sstAnom, length(lon)*length(lat),:) .* W) ./ sum(W)
sstFin = sstAnom .- mean(sstWeighted[.!isequal.(sstWeighted, missing)]) # do weighted mean here
Xt = hcat(ones(size(sstFin, 3)),1:size(sstFin, 3))
Y = reshape(sstFin, length(lat)*length(lon), :)
betas = [inv(Xt' * Xt) * Xt' * Y[i,:] for i in axes(Y,1)]
sstFin = reshape(reduce(hcat, [Y[i,:] .- betas[i][1] .* Xt[:,1] .- betas[i][2] .* Xt[:,2] for i in axes(Y,1)])', length(lat), length(lon), :)

sst = sstFin


#endregion

######################################################################
#### Choose Area Subset
######################################################################
#region

#### subset area
latselects = 20.0 .<= lat .<= 60.0
lonselects = (110 .<= lon) .| (lon .<= -100.0)
newlons = Vector(vcat(range(180, 359, step = 1), range(0, 179, step = 1)))

lonOrig = lon
latOrig = lat


sst = cat(sst[:,convert(BitVector,vcat(zeros(180),lonselects[181:360])),:], sst[:,convert(BitVector,vcat(lonselects[1:180],zeros(180))),:], dims = 2)
sst = sst[reverse(latselects),:,:]
lon = vcat(newlons[convert(BitVector,vcat(zeros(180),lonselects[181:360]))], newlons[convert(BitVector,vcat(lonselects[1:180],zeros(180)))])
lat = reverse(lat[latselects])

nlat, nlon, nT = size(sst)


#### set up date format
startdate = DateTime(1870,01,01) # start date
dts = startdate + Dates.Day.(convert.(Int64, floor.(T))) # get date corresponding to each time 
yrs = Dates.year.(dts) # get years to use as identifiers
mths = Dates.month.(dts) # get months to use as identifiers


#### remove missing data
land_id = [sum(isequal.(sst[i, j, :], missing)) > 0 ? false : true for i in axes(sst, 1), j in axes(sst, 2)]
coords = hcat(collect.(vec(collect(Base.product(lon, lat))))...)
location_inds = DataFrame(x = coords[1, :], y = coords[2, :], sea = reshape(land_id', :))
sea_inds = @chain location_inds begin
    @subset(:sea .== true)
end


# Plots.scatter(location_inds[!,:x], location_inds[!,:y], group = location_inds[!,:sea], label = false)



#### function to rejoin data to full data with missings
function rejoinData(Z, location_inds, sea_inds)
    
    datdf = DataFrame(Z, :auto)
    obs = DataFrame(x = sea_inds[:,1], y = sea_inds[:,2])
    datobs = hcat(obs, datdf)
    all_locs = DataFrame(x = location_inds[:,1], y = location_inds[:,2], ind = location_inds[:,3])

    df = leftjoin(all_locs, datobs, on = [:x, :y])
    df = sort(df, [:y, :x])

    return Matrix(select(df, Not([:x, :y, :ind])))

end


# permutedims(sst[:, :, :], (2, 1, 3))
Z = reshape(permutedims(sst, (2, 1, 3)), :, length(dts))
# Z = reshape(sst[:, :, :], :, length(dts))  
Z = Array{Float32,2}(Z[location_inds[:, :sea], :])




#endregion

######################################################################
#### Visualize Data
######################################################################

#region

#### Z rejoin
# Zplt = rejoinData(Z, location_inds, sea_inds)

#### select the first size EOFs for plotting
svdZ = svd(Z)
U = -rejoinData(svdZ.U[:,1:6], location_inds, sea_inds)


# Plots.contourf(sstFin[:,:,end])


#### SST and PDO plot

stmarks = (lon[1:24:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:24:150])
dtsnewind = dts .> DateTime(1990,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

V = -svdZ.V[dtsnewind,1] ./ sqrt(var(svdZ.V[:,1]))
z = zeros(length(V))
l = [V[i] .< 0 ? V[i] : 0 for i in 1:length(V)]
u = [V[i] .> 0 ? V[i] : 0 for i in 1:length(V)]



function PDO_plot()
    # set up plot
    g = CairoMakie.Figure(resolution = (1300, 800), figure_padding = (10, 40, 10, 10))
    ax11 = Axis(g[2,1])
    ax12 = Axis(g[2,2], xticks = stmarks)
    ax2  = Axis(g[3,1:2], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, V))))

    # plot time basis functions
    CairoMakie.lines!(ax2, 1:sum(dtsnewind), V, color = :black, label = false, linewidth = 2)
    CairoMakie.band!(ax2, 1:sum(dtsnewind), z, u, color = :red, label = false, linealpha = 0)
    CairoMakie.band!(ax2, 1:sum(dtsnewind), l, z, color = :blue, label = false, linealpha = 0)

    # plot SST
    b1 = CairoMakie.heatmap!(ax11, lonOrig, reverse(latOrig), sstFin[:,:,end]',
        colormap = :balance, colorrange = (-1.01, 1.01).*maximum(abs, skipmissing(sstFin[:,:,end])))
    #
    CairoMakie.Colorbar(g[1,1], b1, vertical = false)

    # box of region
    CairoMakie.lines!(ax11, [-100, -100], [20, 60], color = :black, linewidth = 2)
    CairoMakie.lines!(ax11, [-180, -100], [20, 20], color = :black, linewidth = 2)
    CairoMakie.lines!(ax11, [-180, -100], [60, 60], color = :black, linewidth = 2)
    CairoMakie.lines!(ax11, [110, 110], [20, 60], color = :black, linewidth = 2)
    CairoMakie.lines!(ax11, [110, 180], [20, 20], color = :black, linewidth = 2)
    CairoMakie.lines!(ax11, [110, 180], [60, 60], color = :black, linewidth = 2)

    # plot EOF
    b2 = CairoMakie.heatmap!(ax12, lon, lat, reshape(U[:,1], nlon, nlat),
        colormap = :balance, colorrange = (-1.01, 1.01).*maximum(abs, skipmissing(U[:,1])))
    #
    CairoMakie.Colorbar(g[1,2], b2, vertical = false)
    
    g
end

bold_theme = Theme(
    Axis = (
        linewidth = 5,
        titlesize = 30,
        xticklabelsize = 20, 
        yticklabelsize = 20, 
        titlefont = :bold, 
        xticklabelfont = :bold, 
        yticklabelfont = :bold,
    );
    Colorbar = (
        ticklabelsize = 20,
        ticklabelfont = :bold,
    )
)

g = with_theme(PDO_plot, bold_theme)

# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDOoriginal.png", g)
# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDOoriginal.eps", g)

#endregion

########################################################################
#### variogram of each basis function
########################################################################
#region

# svdZ = svd(Z)
# U = svdZ.U[:,1:50]

# writedlm("/Users/JSNorth/Desktop/U.csv", [locs_obs U], ',')

# table = (; U1 = U[:,1], U2 = U[:,2])
# coord = [(locs_obs[i,1], locs_obs[i,2]) for i in axes(locs_obs, 1)]

# georef(U, locs_obs)

#endregion



######################################################################
#### Load in data
######################################################################

locs_obs = Matrix(select(sea_inds, [:x, :y]))
locs_obs = Matrix{Float64}(locs_obs)


# @load "/Users/JSNorth/Desktop/PDO_6.jld2" data pars posterior

@load "/Users/JSNorth/Desktop/PDOResults/PDO_6.jld2" data pars posterior
P = posterior
for i in 7:10
    @load "/Users/JSNorth/Desktop/PDOResults/PDO_" * string(i) * ".jld2" data pars posterior
    P = vcat(P, posterior)
end

posterior = Posterior(data, P)

posterior.D_hat
posterior.D_lower
posterior.D_upper

posterior.σ_hat
posterior.σU_hat
posterior.σV_hat



########################
#### Plot Bayesian PDO Index - requires model run!!!!
########################
#region

stmarks = (lon[1:24:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:24:150])
# dtsnewind = dts .> DateTime(1990,01,01)
dtsnewind = dts .> DateTime(2010,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

V = posterior.V_hat[dtsnewind,1] ./ sqrt(var(posterior.V_hat[:,1]))
z = zeros(length(V))
l = [V[i] .< 0 ? V[i] : 0 for i in eachindex(V)]
u = [V[i] .> 0 ? V[i] : 0 for i in eachindex(V)]



function PDOBayes()
    # set up plot
    g = CairoMakie.Figure(resolution = (1400, 600), figure_padding = (10, 40, 10, 10))
    ax  = Axis(g[1,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, V))), backgroundcolor = :grey80)

    # plot time basis functions
    CairoMakie.band!(ax, 1:sum(dtsnewind), z, u, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(ax, 1:sum(dtsnewind), l, z, color = (:blue, 0.5), label = false, linealpha = 0)

    for i in axes(posterior.V, 3)
        CairoMakie.lines!(ax, 1:sum(dtsnewind), posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])), color = :yellow, label = false, linewidth = 1)
    end

    CairoMakie.lines!(ax, 1:sum(dtsnewind), V, color = :black, label = false, linewidth = 2)
    g
end

g = with_theme(PDOBayes, bold_theme)


save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDObayes.png", g)

#endregion


########################################################################
#### Posterior Plot of PDO CI
########################################################################
#region

# dtsnewind = dts .> DateTime(1990,01,01)
dtsnewind = dts .> DateTime(2010,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))

V = posterior.V_hat[dtsnewind,1] ./ sqrt(var(posterior.V_hat[:,1]))
z = zeros(length(V))
l = [V[i] .< 0 ? V[i] : 0 for i in eachindex(V)]
u = [V[i] .> 0 ? V[i] : 0 for i in eachindex(V)]


Vpost = reduce(hcat, [posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])) for i in axes(posterior.V, 3)])
Vhpd = hcat(collect.([hpd(Vpost[n,:]) for n in axes(Vpost, 1)])...)


function PDOBayesCI()
    # set up plot
    g = CairoMakie.Figure(resolution = (1800, 600), figure_padding = (10, 40, 10, 10))
    ax  = Axis(g[1,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, V))), backgroundcolor = :grey80)

    # plot time basis functions
    CairoMakie.band!(ax, 1:sum(dtsnewind), z, u, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(ax, 1:sum(dtsnewind), l, z, color = (:blue, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(ax, 1:sum(dtsnewind), Vhpd[1,:], Vhpd[2,:], color = (:yellow), label = false)

    CairoMakie.lines!(ax, 1:sum(dtsnewind), V, color = :black, label = false, linewidth = 2)
    g
end

g = with_theme(PDOBayesCI, bold_theme)



#endregion


########################################################################
#### Posterior Difference in Bayes and Algorithmic
########################################################################
#region

# stmarks = (lon[1:24:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:24:150])
dtsnewind = dts .> DateTime(1990,01,01)
# dtsnewind = dts .> DateTime(2010,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (((Vector(1:sum(dtsnewind)) .- 0.5) .* (1/length(dtsnew)))[1:24:end], string.(dtsticks[1:24:end]))

# posterior difference and CI for the bayes and algorithmic estimate
Vorig = -svdZ.V[dtsnewind,1] ./ sqrt(var(svdZ.V[:,1]))
Vdiff = [(posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i]))) .- Vorig for i in axes(posterior.V, 3)]
Vdiffmean = mean(Vdiff)
CI = hcat(collect.([quantile(reduce(hcat, Vdiff)[i,:], (0.025, 0.975)) for i in eachindex(Vdiffmean)])...)

# check significance and reture 1 if both positive, 0 if no sig, -1 if both negative
CIsign = sign.(CI)
CIsignind = (CIsign[1,:] .== CIsign[2,:]) .* CIsign[2,:]

CIgray = CI[:,CIsignind .== 0]
CIred = CI[:,CIsignind .== 1]
CIblue = CI[:,CIsignind .== -1]

xindsgray = (1:sum(dtsnewind))[CIsignind .== 0]
xindsred = (1:sum(dtsnewind))[CIsignind .== 1]
xindsblue = (1:sum(dtsnewind))[CIsignind .== -1]



function PDOBayesDiff()
    # set up plot
    g = CairoMakie.Figure(resolution = (1800, 600), figure_padding = (10, 40, 10, 10))
    ax  = Axis(g[1,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = ((0, 1), ((-1.05, 1.05).*maximum(abs, CI))))

    CairoMakie.hspan!(ax, CIgray[1,:], CIgray[2,:], 
        xmin = (xindsgray .- 1) .* (1/length(dtsnew)), 
        xmax = (xindsgray .- 0) .* (1/length(dtsnew)),
        color = (:gray, 0.5), label = false, linealpha = 0)
    #
    CairoMakie.hspan!(ax, CIred[1,:], CIred[2,:], 
        xmin = (xindsred .- 1) .* (1/length(dtsnew)), 
        xmax = (xindsred .- 0) .* (1/length(dtsnew)),
        color = (:red, 0.5), label = false, linealpha = 0)
    #
    CairoMakie.hspan!(ax, CIblue[1,:], CIblue[2,:], 
        xmin = (xindsblue .- 1) .* (1/length(dtsnew)), 
        xmax = (xindsblue .- 0) .* (1/length(dtsnew)),
        color = (:blue, 0.5), label = false, linealpha = 0)
    #

    CairoMakie.lines!(ax, (Vector(1:sum(dtsnewind)) .- 0.5) .* (1/length(dtsnew)), Vdiffmean, color = :black, label = false, linewidth = 2)
    CairoMakie.hlines!(ax, 0, color = :black, label = false, linewidth = 2)
    g
end

g = with_theme(PDOBayesDiff, bold_theme)

# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDObayesDiff.png", g)


#endregion


########################################################################
#### Posterior plot of spatial basis
########################################################################
#region


svdU = rejoinData(svdZ.U[:,1:6], location_inds, sea_inds)
U = rejoinData(-posterior.U_hat, location_inds, sea_inds)
diff1 = mean([rejoinData(-posterior.U[:,:,i], location_inds, sea_inds)[:,1] .- svdU[:,1] for i in axes(posterior.U, 3)])



stmarks = (lon[1:37:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:37:150])

cr = (-1.01, 1.01).*maximum(abs, skipmissing(vcat(U[:,1], svdU[:,1])))

quantile(skipmissing(diff1), range(0, 1, step = 0.01))
# crDiff = (-0.9, 0.9).*maximum(abs, skipmissing(diff1))
crDiff = (-0.0008, 0.0008)
nsteps = 20
nticks = 5


# function PDOSpatial()
#     # set up plot
#     g = CairoMakie.Figure(resolution = (1800, 500), linewidth = 5)
#     ax11 = Axis(g[1,1], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "Bayesian Estimate")
#     ax12 = Axis(g[1,2], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "Algorithmic Estimate")
#     ax13  = Axis(g[1,3], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "Difference")


#     # plot EOF
#     b1 = CairoMakie.contourf!(ax11, lon, lat, reshape(U[:,1], nlon, nlat),
#         colormap = :balance, colorrange = cr, levels = range(cr[1], cr[2], step = (cr[2]-cr[1])/nsteps))
#     #

#     b2 = CairoMakie.contourf!(ax12, lon, lat, reshape(svdU[:,1], nlon, nlat),
#         colormap = :balance, colorrange = cr, levels = range(cr[1], cr[2], step = (cr[2]-cr[1])/nsteps))
#     #
#     CairoMakie.Colorbar(g[2,1:2], b1, ticks = round.(range(cr[1], cr[2], length = nticks), digits = 3), vertical = false, flipaxis = false)


#     b3 = CairoMakie.contourf!(ax13, lon, lat, reshape(diff1, nlon, nlat),
#         colormap = :PRGn_11, colorrange = crDiff, levels = range(crDiff[1], crDiff[2], step = (crDiff[2]-crDiff[1])/nsteps))
#     #
#     CairoMakie.Colorbar(g[1,4], b3, ticks = round.(range(crDiff[1], crDiff[2], length = nticks), digits = 4))

#     g
# end


function PDOSpatial()
    # set up plot
    g = CairoMakie.Figure(resolution = (1000, 1200), linewidth = 5)
    ax11 = Axis(g[1,1], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "Bayesian Estimate")
    ax12 = Axis(g[2,1], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "Algorithmic Estimate")
    ax13  = Axis(g[3,1], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "Difference")


    # plot EOF
    b1 = CairoMakie.contourf!(ax11, lon, lat, reshape(U[:,1], nlon, nlat),
        colormap = :balance, colorrange = cr, levels = range(cr[1], cr[2], step = (cr[2]-cr[1])/nsteps))
    #

    b2 = CairoMakie.contourf!(ax12, lon, lat, reshape(svdU[:,1], nlon, nlat),
        colormap = :balance, colorrange = cr, levels = range(cr[1], cr[2], step = (cr[2]-cr[1])/nsteps))
    #
    CairoMakie.Colorbar(g[1:2,2], b1, ticks = round.(range(cr[1], cr[2], length = nticks), digits = 3), labelpadding = 300)


    b3 = CairoMakie.contourf!(ax13, lon, lat, reshape(diff1, nlon, nlat),
        colormap = :balance, colorrange = crDiff, levels = range(crDiff[1], crDiff[2], step = (crDiff[2]-crDiff[1])/nsteps))
    #
    CairoMakie.Colorbar(g[3,2], b3, ticks = round.(range(crDiff[1], crDiff[2], length = nticks), digits = 4))
    colgap!(g.layout, 30)
    g
end


g = with_theme(PDOSpatial, bold_theme)

# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDObayesSpace.png", g)



#endregion


########################################################################
#### 6 panel plot
########################################################################
#region


stmarks = (lon[1:24:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:24:150])
# stmarks = (lon[1:37:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:37:150])
# dtsnewind = dts .> DateTime(1990,01,01)
dtsnewind = dts .> DateTime(2010,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))


# Spatial EOF 
AEOF = rejoinData(svdZ.U[:,1:6], location_inds, sea_inds)[:,1]
PEOF = rejoinData(-posterior.U_hat, location_inds, sea_inds)[:,1]
EOFdiff = mean([rejoinData(-posterior.U[:,:,i], location_inds, sea_inds)[:,1] .- AEOF for i in axes(posterior.U, 3)])


# PDO index
PPDOdraws = reduce(hcat, [posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])) for i in axes(posterior.V, 3)])
# Vhpd = hcat(collect.([hpd(Vpost[n,:]) for n in axes(Vpost, 1)])...)
PPDO = mean(PPDOdraws, dims = 2)[:,1]
PPDOdiff = PPDO .- PPDOdraws
PPDOdiffCI = hcat(collect.([quantile(PPDOdiff[n,:], (0.01, 0.99)) for n in axes(PPDOdiff, 1)])...)

# 99% hpd interval
# PPDOhpd = hcat(collect.([hpd(PPDOdraws[n,:], p = 0.99) for n in axes(PPDOdraws, 1)])...)
# 99% CI
PPDOCI = hcat(collect.([quantile(PPDOdraws[n,:], (0.01, 0.99)) for n in axes(PPDOdraws, 1)])...)


    

Pz = zeros(length(PPDO))
Pl = [PPDO[i] .< 0 ? PPDO[i] : 0 for i in eachindex(PPDO)]
Pu = [PPDO[i] .> 0 ? PPDO[i] : 0 for i in eachindex(PPDO)]

APDO = -svdZ.V[dtsnewind,1] ./ sqrt(var(svdZ.V[:,1]))
Az = zeros(length(APDO))
Al = [APDO[i] .< 0 ? APDO[i] : 0 for i in eachindex(APDO)]
Au = [APDO[i] .> 0 ? APDO[i] : 0 for i in eachindex(APDO)]

PPDOdiff = PPDOdraws .- APDO
PPDOdiffCI = hcat(collect.([quantile(PPDOdiff[n,:], (0.01, 0.99)) for n in axes(PPDOdiff, 1)])...)



# Plot limits and color bars
PDOlimits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, hcat(PPDO, APDO))))
PDOdifflimits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, PPDOdiff)))
EOFlimits = ((110, 259), (20, 60))
EOFclims = (-1.05, 1.05).*maximum(abs, skipmissing(hcat(AEOF, PEOF)))
crDiff = (-0.0008, 0.0008)
nsteps = 20
nticks = 5


function PDOFigure()
    #### figure layout
    fig = CairoMakie.Figure(resolution = (1600, 1200), linewidth = 5)
    a = Axis(fig[1,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOlimits, title = "a)", titlealign = :left) # A-PDO
    b = Axis(fig[1,2], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "b)", titlealign = :left) # A-EOF
    c  = Axis(fig[2,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOlimits, title = "c)", titlealign = :left) # P-PDO
    d  = Axis(fig[2,2], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "d)", titlealign = :left) # P-EOF
    e  = Axis(fig[3,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOdifflimits, title = "e)", titlealign = :left) # P-PDO Deviation
    f  = Axis(fig[3,2], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "f)", titlealign = :left) # A-EOF minus P-EOF

    #### panel a
    CairoMakie.band!(a, 1:sum(dtsnewind), Az, Au, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(a, 1:sum(dtsnewind), Al, Az, color = (:blue, 0.5), label = false, linealpha = 0)
    CairoMakie.lines!(a, 1:sum(dtsnewind), APDO, color = :black, label = false, linewidth = 1)

    #### panel b
    b2 = CairoMakie.contourf!(b, lon, lat, reshape(AEOF, nlon, nlat),
            colormap = :balance, colorrange = EOFclims, levels = range(EOFclims[1], EOFclims[2], step = (EOFclims[2]-EOFclims[1])/nsteps))
    #
    CairoMakie.Colorbar(fig[1:2,3], b2, ticks = round.(range(EOFclims[1], EOFclims[2], length = nticks), digits = 3), labelpadding = 300)


    #### panel c
    CairoMakie.band!(c, 1:sum(dtsnewind), Pz, Pu, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(c, 1:sum(dtsnewind), Pl, Pz, color = (:blue, 0.5), label = false, linealpha = 0)
    CairoMakie.lines!(c, 1:sum(dtsnewind), PPDO, color = :black, label = false, linewidth = 1)
    # for i in axes(posterior.V, 3)
    #     CairoMakie.lines!(c, 1:sum(dtsnewind), posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])), color = :gray, label = false, linewidth = 1)
    # end
    CairoMakie.band!(c, 1:sum(dtsnewind), PPDOCI[1,:], PPDOCI[2,:], color = (:yellow), label = false)
    CairoMakie.lines!(c, 1:sum(dtsnewind), PPDO, color = :black, label = false, linewidth = 1)


    #### panel d
    CairoMakie.contourf!(d, lon, lat, reshape(PEOF, nlon, nlat),
            colormap = :balance, colorrange = EOFclims, levels = range(EOFclims[1], EOFclims[2], step = (EOFclims[2]-EOFclims[1])/nsteps))
    #

    #### panel e
    # for i in axes(PPDOdiff, 2)
    for i in 1:100:5000
        CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiff[:,i], color = :gray, label = false, linewidth = 1)
    end
    CairoMakie.lines!(e, 1:sum(dtsnewind), mean(PPDOdiff, dims = 2)[:,1], color = :black, label = false, linewidth = 2)
    CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiffCI[1,:], color = :red, label = false, linewidth = 2)
    CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiffCI[2,:], color = :red, label = false, linewidth = 2)


    #### panel f
    b3 = CairoMakie.contourf!(f, lon, lat, reshape(EOFdiff, nlon, nlat),
            colormap = :balance, colorrange = crDiff, levels = range(crDiff[1], crDiff[2], step = (crDiff[2]-crDiff[1])/nsteps))
    #
    CairoMakie.Colorbar(fig[3,3], b3, ticks = round.(range(crDiff[1], crDiff[2], length = nticks), digits = 4))

    fig
end

g = with_theme(PDOFigure, bold_theme)

# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDOFigure.png", g)
# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDOFigure.eps", g)


#endregion





########################################################################
#### 6 panel plot for presentation
########################################################################
#region


stmarks = (lon[1:24:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:24:150])
# stmarks = (lon[1:37:150], string.(vcat(110.5:179.5, -179.5:-100.5))[1:37:150])
# dtsnewind = dts .> DateTime(1990,01,01)
dtsnewind = dts .> DateTime(2010,01,01)
dtsnew = dts[dtsnewind]
dtsticks = Dates.format.(dtsnew, "yyyy-mm")
tkmarks = (1:24:length(dtsnew), string.(dtsticks[1:24:end]))


# Spatial EOF 
AEOF = rejoinData(svdZ.U[:,1:6], location_inds, sea_inds)[:,1]
PEOF = rejoinData(-posterior.U_hat, location_inds, sea_inds)[:,1]
EOFdiff = mean([rejoinData(-posterior.U[:,:,i], location_inds, sea_inds)[:,1] .- AEOF for i in axes(posterior.U, 3)])


# PDO index
PPDOdraws = reduce(hcat, [posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])) for i in axes(posterior.V, 3)])
# Vhpd = hcat(collect.([hpd(Vpost[n,:]) for n in axes(Vpost, 1)])...)
PPDO = mean(PPDOdraws, dims = 2)[:,1]
PPDOdiff = PPDO .- PPDOdraws
PPDOdiffCI = hcat(collect.([quantile(PPDOdiff[n,:], (0.01, 0.99)) for n in axes(PPDOdiff, 1)])...)

# 99% hpd interval
# PPDOhpd = hcat(collect.([hpd(PPDOdraws[n,:], p = 0.99) for n in axes(PPDOdraws, 1)])...)
# 99% CI
PPDOCI = hcat(collect.([quantile(PPDOdraws[n,:], (0.01, 0.99)) for n in axes(PPDOdraws, 1)])...)


    

Pz = zeros(length(PPDO))
Pl = [PPDO[i] .< 0 ? PPDO[i] : 0 for i in eachindex(PPDO)]
Pu = [PPDO[i] .> 0 ? PPDO[i] : 0 for i in eachindex(PPDO)]

APDO = -svdZ.V[dtsnewind,1] ./ sqrt(var(svdZ.V[:,1]))
Az = zeros(length(APDO))
Al = [APDO[i] .< 0 ? APDO[i] : 0 for i in eachindex(APDO)]
Au = [APDO[i] .> 0 ? APDO[i] : 0 for i in eachindex(APDO)]

PPDOdiff = PPDOdraws .- APDO
PPDOdiffCI = hcat(collect.([quantile(PPDOdiff[n,:], (0.01, 0.99)) for n in axes(PPDOdiff, 1)])...)



# Plot limits and color bars
PDOlimits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, hcat(PPDO, APDO))))
PDOdifflimits = ((1, length(dtsnew)), ((-1.05, 1.05).*maximum(abs, PPDOdiff)))
EOFlimits = ((110, 259), (20, 60))
EOFclims = (-1.05, 1.05).*maximum(abs, skipmissing(hcat(AEOF, PEOF)))
crDiff = (-0.0008, 0.0008)
nsteps = 20
nticks = 5


function PDOFigure()
    #### figure layout
    fig = CairoMakie.Figure(resolution = (1600, 1200), linewidth = 5)
    a = Axis(fig[1,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOlimits, title = "Algorithmic PDO Index", titlealign = :left) # A-PDO
    b = Axis(fig[1,2], limits = ((110, 259), (20, 60)), xticks = stmarks, title = "Algorithmic Spatial Basis Function", titlealign = :left) # A-EOF
    c  = Axis(fig[2,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOlimits, title = "Probablistic PDO Index", titlealign = :left) # P-PDO
    d  = Axis(fig[2,2], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "Probablistic Spatial Basis Function", titlealign = :left) # P-EOF
    e  = Axis(fig[3,1], xticks = tkmarks, xticklabelrotation=-π/4, limits = PDOdifflimits, title = "Probablistic - Algorithmic PDO Index", titlealign = :left) # P-PDO Deviation
    f  = Axis(fig[3,2], limits = ((110, 259), (20.5, 59.5)), xticks = stmarks, title = "Probablistic - Algorithmic Spatial Basis Function", titlealign = :left) # A-EOF minus P-EOF

    #### panel a
    CairoMakie.band!(a, 1:sum(dtsnewind), Az, Au, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(a, 1:sum(dtsnewind), Al, Az, color = (:blue, 0.5), label = false, linealpha = 0)
    CairoMakie.lines!(a, 1:sum(dtsnewind), APDO, color = :black, label = false, linewidth = 1)

    #### panel b
    b2 = CairoMakie.contourf!(b, lon, lat, reshape(AEOF, nlon, nlat),
            colormap = :balance, colorrange = EOFclims, levels = range(EOFclims[1], EOFclims[2], step = (EOFclims[2]-EOFclims[1])/nsteps))
    #
    CairoMakie.Colorbar(fig[1:2,3], b2, ticks = round.(range(EOFclims[1], EOFclims[2], length = nticks), digits = 3), labelpadding = 300)


    #### panel c
    CairoMakie.band!(c, 1:sum(dtsnewind), Pz, Pu, color = (:red, 0.5), label = false, linealpha = 0)
    CairoMakie.band!(c, 1:sum(dtsnewind), Pl, Pz, color = (:blue, 0.5), label = false, linealpha = 0)
    CairoMakie.lines!(c, 1:sum(dtsnewind), PPDO, color = :black, label = false, linewidth = 1)
    # for i in axes(posterior.V, 3)
    #     CairoMakie.lines!(c, 1:sum(dtsnewind), posterior.V[dtsnewind,1,i] ./ sqrt(var(posterior.V[:,1,i])), color = :gray, label = false, linewidth = 1)
    # end
    CairoMakie.band!(c, 1:sum(dtsnewind), PPDOCI[1,:], PPDOCI[2,:], color = (:yellow), label = false)
    CairoMakie.lines!(c, 1:sum(dtsnewind), PPDO, color = :black, label = false, linewidth = 1)


    #### panel d
    CairoMakie.contourf!(d, lon, lat, reshape(PEOF, nlon, nlat),
            colormap = :balance, colorrange = EOFclims, levels = range(EOFclims[1], EOFclims[2], step = (EOFclims[2]-EOFclims[1])/nsteps))
    #

    #### panel e
    # for i in axes(PPDOdiff, 2)
    for i in 1:100:5000
        CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiff[:,i], color = :gray, label = false, linewidth = 1)
    end
    CairoMakie.lines!(e, 1:sum(dtsnewind), mean(PPDOdiff, dims = 2)[:,1], color = :black, label = false, linewidth = 2)
    CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiffCI[1,:], color = :red, label = false, linewidth = 2)
    CairoMakie.lines!(e, 1:sum(dtsnewind), PPDOdiffCI[2,:], color = :red, label = false, linewidth = 2)


    #### panel f
    b3 = CairoMakie.contourf!(f, lon, lat, reshape(EOFdiff, nlon, nlat),
            colormap = :balance, colorrange = crDiff, levels = range(crDiff[1], crDiff[2], step = (crDiff[2]-crDiff[1])/nsteps))
    #
    CairoMakie.Colorbar(fig[3,3], b3, ticks = round.(range(crDiff[1], crDiff[2], length = nticks), digits = 4))

    fig
end

g = with_theme(PDOFigure, bold_theme)

# save("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/PDOFigureSlides.png", g)


#endregion