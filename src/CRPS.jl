


ECDF(posterior.U[1,1,:], -0.145)

test = StatsBase.ecdf(posterior.U[1,1,:])
test(posterior.U[1,1,:])

Y = posterior.U[1,1,:]
x = U[1,1]

function CRPS(Y, x)
    cdfY = StatsBase.ecdf(Y)
    sY = sort(Y)
    regionInd = sY .<= x
    S1 = cdfY(sY[regionInd]) .^2
    S2 = (cdfY(sY[.!regionInd]) .- 1) .^2
    return mean(vcat(S1, S2))
end

ecrps = [CRPS(posterior.U[i,j,:], U[i,j]) for i in axes(U,1), j in axes(U,2)]

Plots.plot(ecrps[:,1])
Plots.plot!(ecrps[:,5])