########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 19-December-2023
#### Description: Functions used for computing anomolies
########################################################################



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
    return Xanom, Xavg
    
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


"""
    anomalyDetrend(F, lat, lon)

Computes the anomalies of a field and removes the climatology.

# Arguments
    - F: observed field of size (lat, lon, time)
    - lat: vector of latitudes
    - lon: vector of longitudes

# Output
    - FFin: Final detrended data set of size (lat, lon, time)
    - FMonMean: monthly means of size (lat, lon, 12)
    - FWeightedMean: scalar of global mean accounting for cell weight
    - betas: betas for temporal trend removing

# Examples
```
# given two-meter temperature array t2m of size (lat, lon, time)
t2mCentered, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon)
```
"""
function anomalyDetrend(F, lat, lon)

    FAnom, FMonMean = rmMonAnn(F)

    W = areaWeight(lon, lat)
    FWeighted = (reshape(FAnom, length(lon)*length(lat),:) .* W) ./ sum(W)
    FWeightedMean = mean(FWeighted[.!isequal.(FWeighted, missing)])
    FFin = FAnom .- FWeightedMean # do weighted mean here
    Xt = hcat(ones(size(FFin, 3)),1:size(FFin, 3))
    Y = reshape(FFin, length(lat)*length(lon), :)
    betas = [inv(Xt' * Xt) * Xt' * Y[i,:] for i in axes(Y,1)]
    FFin = reshape(reduce(hcat, [Y[i,:] .- betas[i][1] .* Xt[:,1] .- betas[i][2] .* Xt[:,2] for i in axes(Y,1)])', length(lon), length(lat), :)
    
    return FFin, FMonMean, FWeightedMean, betas

end


"""
    reconstructSurface(F, monthlyAverage, weightedMean, betas, lon, lat)

Reconstructs the surface from the anomaly field. Undoes the anomalyDetrend(F, lat, lon) function.

# Arguments
    - F: anomaly data of size (lat, lon, time)
    - monthlyAverage: monthly means of size (lat, lon, 12)
    - weightedMean: scalar of global mean accounting for cell weight
    - betas: betas for temporal trend removing
    - lat: vector of latitudes
    - lon: vector of longitudes

# Output
    - Fnew: Reconstructed observation field

# Examples
```
# given two-meter temperature array t2m of size (lat, lon, time)
t2mCentered, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon)
reconstructSurface(t2mCentered, t2mMonMean, t2mWeightedMean, betas, lon, lat)
```
"""
function reconstructSurface(F, monthlyAverage, weightedMean, betas, lat, lon)
    
    nx, ny, nt = size(F)
    Xt = hcat(ones(size(F, 3)),1:size(F, 3))
    Ytmp = reshape(F, length(lat)*length(lon), :)
    Ynew = reshape(reduce(hcat, [Ytmp[i,:] .+ betas[i][1] .* Xt[:,1] .+ betas[i][2] .* Xt[:,2] for i in axes(Ytmp,1)])', length(lon), length(lat), :)
    AnomNew = Ynew .+ weightedMean

    moinds = mod.(1:nt, 12)
    moinds[moinds .== 0] .= 12
    Fnew = reshape(reduce(hcat,[AnomNew[:,:,i] .+ monthlyAverage[:,:,moinds[i]] for i in 1:nt]), nx, ny, nt)

    return Fnew

end


"""
    anomalyDetrendWeights(F, monthlyAverage, betas, lat, lon)

Detrends a new observation F given the monthlyAverage (lat, lon, 12) and betas for time trend.

# Arguments
    - F: anomaly data of size (lat, lon, time)
    - monthlyAverage: monthly means of size (lat, lon, 12)
    - betas: betas for temporal trend removing
    - lat: vector of latitudes
    - lon: vector of longitudes

# Output
    - FFin: Final detrended data set of size (lat, lon, time)
    - FWeightedMean: scalar of global mean accounting for cell weight

# Examples
```
# given two-meter temperature array t2m of size (lat, lon, time)
t2mCentered, t2mMonMean, t2mWeightedMean, betas = anomalyDetrend(t2m, lat, lon)
t2mCenteredPred, t2mWeightedMeanPred = anomalyDetrendWeights(t2m[:,:,121:end], t2mMonMean, betas, lat, lon)
```
"""
function anomalyDetrendWeights(F, monthlyAverage, betas, lat, lon)

    FAnom = calcMonAnom(F, monthlyAverage)

    W = areaWeight(lon, lat)
    FWeighted = (reshape(FAnom, length(lon)*length(lat),:) .* W) ./ sum(W)
    FWeightedMean = mean(FWeighted[.!isequal.(FWeighted, missing)])
    FFin = FAnom .- FWeightedMean
    Xt = hcat(ones(size(FFin, 3)),1:size(FFin, 3))
    Y = reshape(FFin, length(lat)*length(lon), :)
    FFin = reshape(reduce(hcat, [Y[i,:] .- betas[i][1] .* Xt[:,1] .- betas[i][2] .* Xt[:,2] for i in axes(Y,1)])', length(lon), length(lat), :)
    
    return FFin, FWeightedMean

end;
