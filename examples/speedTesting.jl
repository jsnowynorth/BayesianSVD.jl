

using BayesianSVD
using Distances, Plots, Random, Distributions, LinearAlgebra, Statistics
using Kronecker
using BenchmarkTools


######################################################################
#### Generate Some Data
######################################################################
#region

m = 100
n = 500
x = range(-20, 20, n)
t = range(0, 10, m)

ΣU = MaternCorrelation(x, ρ = 4, ν = 3.5, metric = Euclidean())
ΣV = MaternCorrelation(t, ρ = 1, ν = 3.5, metric = Euclidean())


D = [40, 30, 20, 10, 5]
k = 5
ϵ = 0.01

Random.seed!(2)
U, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2; SNR = true)

#endregion

########################################################################
#### Null space computation
########################################################################
#region

U1 = U[:,2:end]

N = eigen(diagm(ones(n)) - U1 * inv(U1' * U1) * U1').vectors
nullspace(U1')

svd(U1'; full = true).Vt[k:(end),:]'

svd(U1; full = true).U[:,k:(end)]

function NS(A)
    k = size(A, 2)
    svd(A; full = true).U[:,(k+1):(end)]
end

u = U[:,2:end]

@benchmark NS(u) samples = 10000 evals = 10
@benchmark nullspace(u') samples = 10000 evals = 10



#endregion

########################################################################
#### model test
########################################################################
#region


k = 5
ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())
ΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())
data = Data(Z, x, t, k)
pars = Pars(data, ΩU, ΩV)


U = pars.U
UZ = pars.UZ

for i in 1:data.k
    inds = i .∉ Vector(1:data.k)
    N = nullspace(U[:,inds]')
    E = data.Z - U[:,inds] * diagm(pars.D[inds]) * pars.V[:,inds]'

    NON = Hermitian(N' * pars.ΩU[i].K * N)
    NONinv = inv(NON)
    
    m = pars.D[i] * (1/pars.σ) * N' * E * pars.V[:,i]
    S = pars.D[i]^2 * (1/pars.σU[i]) .* NONinv + pars.D[i]^2 * (1/pars.σ) * diagm(ones(data.n - data.k + 1))

    UZ[:,i] = rand(MvNormalCanon(m, Hermitian(S)))
    UZ[:,i] = UZ[:,i] / norm(UZ[:,i])
    U[:,i] = N * UZ[:,i]
    pars.NU[:,:,i] = N
    pars.NΩUN[:,:,i] = NON
    pars.NΩUNinv[:,:,i] = NONinv
end

pars.U = U
pars.UZ = UZ


(inv(NON) * NON)[(end-10):end,(end-10):end]
((NON \ diagm(ones(n-k+1))) * NON)[(end-10):end,(end-10):end]

((NON \ diagm(ones(n-k+1))) * NON)[140:150,140:150]

(((N' * pars.ΩU[i].K * N) \ diagm(ones(n-k+1))) * (N' * pars.ΩU[i].K * N))[(end-10):end,(end-10):end]


inv(cholesky(NON)) .- inv(NON)
inv(cholesky(NON)) .- (NON \ diagm(ones(n-k+1)))

M = rand(30, 30)
M = NON .+ 0.1*diagm(ones(196))

NON = Hermitian(N' * (pars.ΩU[i].K .+ 0.1*diagm(ones(n))) * N)

Ω = (pars.ΩU[i].K .+ 0.0001*diagm(ones(n))) .- 0.0001

Ω[Ω .< 0] .= 0

M = N' * (pars.ΩU[i].K .+ 0.0001*diagm(ones(n))) * N
M = N' * Ω * N
M = N' * pars.ΩU[i].K * N
det(M)

ainds = 1:250
binds = 251:(n-k+1)

# ainds = 1:10
# binds = 11:30

A = M[ainds, ainds]
B = M[ainds, binds]
C = M[binds, ainds]
D = M[binds, binds]

Anew = inv(A) + inv(A)*B*inv(D-C*inv(A)*B)*C*inv(A)
Bnew = -inv(A)*B*inv(D-C*inv(A)*B)
Cnew = -inv(D-C*inv(A)*B)*C*inv(A)
Dnew = inv(D-C*inv(A)*B)

inv(M) .- [Anew Bnew; Cnew Dnew]

[Anew Bnew; Cnew Dnew] * [A B; C D]

inv(M)
(M \ diagm(ones(n-k+1)))



function Finv(M, indA, indB)

    A = M[indA, indA]
    B = M[indA, indB]
    C = M[indB, indA]
    D = M[indB, indB]

    t1 = inv(D-C*inv(A)*B)
    AI = inv(A)
    AIB = inv(A)*B
    AIC = C*inv(A)

    Anew = AI + AIB*t1*AIC
    Bnew = -AIB*t1
    Cnew = -t1*AIC
    Dnew = t1

    [Anew Bnew; Cnew Dnew]
    
end


@benchmark Finv(M, ainds, binds) samples = 10000 evals = 10

@benchmark inv(M) samples = 10000 evals = 10

@benchmark (M \ diagm(ones(n-k+1))) samples = 10000 evals = 10




# posterior, pars = SampleSVD(data, pars; nits = 2500, burnin = 1000)

#endregion




Random.seed!(1)

k = 5
n = 100
x = range(-5, 5, n)
Σ = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())

Z = rand(MvNormal(zeros(n), Σ.K), k)
X1 = Array{Float64}(undef, n, k)
X2 = Array{Float64}(undef, n, k)

X1[:,1] = Z[:,1]
for i in 2:k
  Xtmp = X1[:,1:(i-1)]
  P = Xtmp*inv(Xtmp'*Xtmp)*Xtmp'
  X1[:,i] = (diagm(ones(n)) - P) * Z[:,i]
end

Z2 = copy(Z[:,end:-1:1])
X2[:,1] = Z2[:,1]

for i in 2:k
    Xtmp = X2[:,1:(i-1)]
    P = Xtmp*inv(Xtmp'*Xtmp)*Xtmp'
    X2[:,i] = (diagm(ones(n)) - P) * Z2[:,i]
  end

X1 = X1 ./ [norm(X1[:,i]) for i in 1:k]'
X2 = X2 ./ [norm(X2[:,i]) for i in 1:k]'

plot(X1[:,2])
plot(X2)

plot(X1[:,1])
plot!(X2[:,5])


