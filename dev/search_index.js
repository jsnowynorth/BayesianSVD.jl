var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"api/#Functions","page":"API","title":"Functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Examples for BayesianSVD.","category":"page"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [BayesianSVD]","category":"page"},{"location":"api/#BayesianSVD.BayesianSVD","page":"API","title":"BayesianSVD.BayesianSVD","text":"BayesianSVD\n\nHere is my package.\n\n\n\n\n\n","category":"module"},{"location":"api/#BayesianSVD.Data-Tuple{Any, IdentityKernel, IdentityKernel, Any}","page":"API","title":"BayesianSVD.Data","text":"Data(Y, ΩU::KernelFunction, ΩV::KernelFunction, k)\n\nCreates the data class.\n\nSee also Pars, Posterior, and SampleSVD.\n\nArguments\n\nY: data of dimension n × m\nΩU::KernelFunction: Kernel for U matrix, of dimension n × n\nΩV::KernelFunction: Kernel for V matrix, of dimension m × m\nk: number of basis functions to keep\n\nTo Do\n\nallow for ΩU and ΩV to be of different types (i.e., ΩU is Matern and ΩV is Identity)\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\n# Identity Covariances\nΩU = IdentityKernel(x, metric = Distances.Euclidean())\nΩV = IdentityKernel(t, metric = Distances.Euclidean())\ndata = Data(Y, ΩU, ΩV, k)\n\n# Matern Covariances\nΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())\nΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Distances.Euclidean())\ndata = Data(Y, ΩU, ΩV, k)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.IdentityKernel-Tuple{Any}","page":"API","title":"BayesianSVD.IdentityKernel","text":"IdentityKernel(x)\nIdentityKernel(x, y)\nIdentityKernel(X::Matrix{Float64})\n\nCreates IdentityKernel <: KernelFunction\n\nSee also Data and MaternKernel.\n\nArguments\n\nx: vector of values at which to evaluate the kernel (dimension 1)\ny: vector of values at which to evaluate the kernel (dimension 2)\nX::Matrix{Float64}: matrix of values at which to evaluate the kernel (all possible combinations)\n\nReturn\n\nK::Array{Float64}\nKinv::Array{Float64}\nlogdet::Float64\n\nExamples\n\nn = 100\nx = range(-5, 5, n)\nΩU = IdentityKernel(x)\n\nn = 100\nx = range(-5, 5, n)\ny = range(-5, 5, n)\nΩU = IdentityKernel(x, y)\n\nlocs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\nΩU = IdentityKernel(locs')\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.MaternKernel-Tuple{Any}","page":"API","title":"BayesianSVD.MaternKernel","text":"MaternKernel(x; ρ = 1, ν = 1, metric = Euclidean())\nMaternKernel(x, y; ρ = 1, ν = 1, metric = Euclidean())\nMaternKernel(X::Matrix{Float64}; ρ = 1, ν = 1, metric = Euclidean())\n\nCreates MaternKernel <: KernelFunction with correlation function\n\nK_nu rho(s s) = frac2^1-nuGamma(nu)left(2nu fracs-srhoright)^nuJ_nuleft(2nu fracs-srhoright)\n\nfor ss in mathcalS, where Gamma is the gamma function, J_nu is the Bessel function of the second kind, and rho nu  are hyperparameters that describe the length-scale and differentiability, respectively.\n\nSee also Data and IdentityKernel.\n\nArguments\n\nx: vector of values at which to evaluate the kernel (dimension 1)\ny: vector of values at which to evaluate the kernel (dimension 2)\nX::Matrix{Float64}: matrix of values at which to evaluate the kernel (all possible combinations)\n\nOptional Arguments\n\nρ = 1: defines the effective range\nν = 1: smoothing parameter\nmetric = Euclidean(): metric to compute the distance\n\nReturn\n\nd::Matrix{Float64}\nρ::Float64\nν::Float64\nmetric::Metric\nK::Array{Float64}\nKinv::Array{Float64}\nlogdet::Float64\n\nExamples\n\nn = 100\nx = range(-5, 5, n)\nΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())\n\nn = 100\nx = range(-5, 5, n)\ny = range(-5, 5, n)\nΩU = MaternKernel(x, y, ρ = 4, ν = 4, metric = Distances.Euclidean())\n\nlocs = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\nΩU = MaternKernel(locs', ρ = 4, ν = 4, metric = Distances.Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.Pars-Tuple{BayesianSVD.IdentityData}","page":"API","title":"BayesianSVD.Pars","text":"Pars(data::Data)\n\nCreates the parameter class of type typeof(data).'\n\nSee also Data, Posterior, and SampleSVD.\n\nArguments\n\ndata: data structure of type Data\n\nExamples\n\nTo Do.\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.Posterior","page":"API","title":"BayesianSVD.Posterior","text":"Posterior\n\nStructure of type posterior with subtypes Identity, Exponential, Gaussian, or Matern. Contains the raw posterior samples and some means and 95% quantiles of parameters. Plotting associated with the structure.\n\nSee also Pars, Data, and SampleSVD.\n\nExamples\n\n\nRandom.seed!(2)\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())\nΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())\n\nk = 5\nΦ = PON(n, k, ΣU.K)\nΨ = PON(n, k, ΣV.K)\n\nD = diagm([40, 20, 10, 5, 2])\n\nϵ = rand(Normal(0, sqrt(0.01)), n, m)\nY = Φ * D * Ψ' + ϵ # n × m\n\n\nΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())\nΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Distances.Euclidean())\ndata = Data(Y, ΩU, ΩV, k)\npars = Pars(data)\n\nposterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = true)\n\nplot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple])\nplot(posterior, t, size = (900, 500), basis = 'V', linewidth = 2, c = [:red :green :purple])\n\n# for spatial basis functions, provide x and y\nplot(posterior, x, y)\n\n\n\n\n\n","category":"type"},{"location":"api/#BayesianSVD.PON-Tuple{Any, Any, Any}","page":"API","title":"BayesianSVD.PON","text":"PON(n, k, Sigma)\n\nCreate an n by k orthonormal matrix with covariance matrix Sigma.\n\nArguments\n\nn: Number of locations \nk: Number of basis functions\nSigma: covariance matrix\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())\nΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())\n\nk = 5\nΦ = PON(n, k, ΣU.K)\nΨ = PON(n, k, ΣV.K)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.SampleSVD-Tuple{BayesianSVD.IdentityData, BayesianSVD.IdentityPars}","page":"API","title":"BayesianSVD.SampleSVD","text":"SampleSVD(data::Data, pars::Pars; nits = 10000, burnin = 5000, show_progress = true)\n\nRuns the MCMC sampler for the Bayesian SVD model.\n\nSee also Pars, Data, and Posterior.\n\nArguments\n\ndata: Data structure of type Identity, Exponential, Gaussian, or Matern\npars: Parameter structure of type Identity, Exponential, Gaussian, or Matern\n\nOptional Arguments\n\nnits = 10000: Total number of posterior samples to compute\nburnin = 5000: Number of samples discarded as burnin\nshow_progress = true: Indicator on whether to show a progress bar (true) or not (false).\n\nExamples\n\n\nRandom.seed!(2)\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())\nΣV = MaternKernel(t, ρ = 3, ν = 4, metric = Euclidean())\n\nk = 5\nΦ = PON(n, k, ΣU.K)\nΨ = PON(n, k, ΣV.K)\n\nD = diagm([40, 20, 10, 5, 2])\n\nϵ = rand(Normal(0, sqrt(0.01)), n, m)\nY = Φ * D * Ψ' + ϵ # n × m\n\n\nΩU = MaternKernel(x, ρ = 4, ν = 4, metric = Distances.Euclidean())\nΩV = MaternKernel(t, ρ = 4, ν = 4, metric = Distances.Euclidean())\ndata = Data(Y, ΩU, ΩV, k)\npars = Pars(data)\n\nposterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500, show_progress = true)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.hpd-Tuple{Any}","page":"API","title":"BayesianSVD.hpd","text":"hpd(x; p=0.95)\n\nComputes the highest posterior density interval of x.\n\nArguments\n\nx: vector of data.\n\nOptional Arguments\n\np: value between 0 and 1 for the probability level.\n\nExamples\n\nx = [1, 3, 2, 5, .2, 1,9]\nhpd(x)\n\n\n\n\n\n","category":"method"},{"location":"examples/","page":"Examples","title":"Examples","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#Simulating-Random-Basis-Functions","page":"Examples","title":"Simulating Random Basis Functions","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"We start by simulating a random orthonormal matrix.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using BayesianSVD, Random, Plots\n\n# set seed\nRandom.seed!(2)\n\n# size of matrix is n (locations) by k (basis functions)\nn = 100\nk = 5\n\n # domain\nx = range(-5, 5, n)\n\n# covariance matrix\nΣ = MaternKernel(x, ρ = 3, ν = 4, metric = Euclidean())\n\n# random n by k matrix with structure\nΦ = PON(n, k, Σ.K)\n\n# plot the basis functions\nPlots.plot(Φ)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"#BayesianSVD","page":"Home","title":"BayesianSVD","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BayesianSVD.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Detailed-API","page":"Home","title":"Detailed API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [BayesianSVD]","category":"page"}]
}
