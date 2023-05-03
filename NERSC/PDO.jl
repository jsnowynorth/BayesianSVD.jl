
########################################################################
#### Author: Joshua North
#### Project: BayesianSpatialBasisFunctions
#### Date: 19-April-2023
#### Description: File to start sampling PDO basis functions
########################################################################


#### data from http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.anom/

# data info
# SOURCES .NOAA .NCDC .ERSST .version5 .anom
# X (140E) (290E) RANGEEDGES
# T (Jan 1926) (Jul 2022) RANGEEDGES
# Y (45S) (45N) RANGEEDGES


#### load packages
using BayesianSVD
using Distances
using Missings
using Statistics
using DataFrames, DataFramesMeta, Chain
using NetCDF, Dates
using JLD2

######################################################################
#### Read In Data
######################################################################

#### source file name
fileName = "../data/HadISST_sst.nc"
# fileName = "/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/data/HadISST_sst.nc"



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
newlons = Vector(vcat(range(160, 359, step = 1), range(0, 159, step = 1)))

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
location_inds = DataFrame(x = coords[1, :], y = coords[2, :], sea = reshape(land_id, :))
sea_inds = @chain location_inds begin
    @subset(:sea .== true)
end

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

Z = reshape(sst[:, :, :], :, length(dts))  
Z = Array{Float32,2}(Z[location_inds[:, :sea], :])

#endregion


########################################################################
#### Run model
########################################################################
#region

locs_obs = Matrix(select(sea_inds, [:x, :y]))
locs_obs = Matrix{Float64}(locs_obs)

t = convert(Vector{Float64}, Vector(1:nT))
Z = convert(Matrix{Float64}, Z)

k = 3
ΩU = MaternCorrelation(locs_obs, ρ = maximum(Distances.pairwise(Haversine(6371), locs_obs'))/100, ν = 3.5, metric = Haversine(6371))
ΩV = IdentityCorrelation(t)
data = Data(Z, locs_obs, t, k)
pars = Pars(data, ΩU, ΩV)

posterior, pars = SampleSVD(data, pars; nits = 500, burnin = 500)


jldsave("../results/PDO_1.jld2"; data, pars, posterior)
# data, pars, posterior = jldopen("../results/PDOResults/PDO.jld2")



#endregion

