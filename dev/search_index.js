var documenterSearchIndex = {"docs":
[{"location":"simulateData/","page":"Examples","title":"Examples","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"simulateData/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"simulateData/#D-space-and-1-D","page":"Examples","title":"1-D space and 1-D","text":"","category":"section"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"This example goes over how to simulate random basis functions.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Load some packages that we will need and set a seed.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"using BayesianSVD\nusing Distances, Plots, Random, Distributions, LinearAlgebra\n\n# set seed\nRandom.seed!(2)","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"The function GenerateData() will simulate data from the model","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Z = UDV + epsilon","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"where Z is a n (space) by m (time) matrix, U is a n times k matrix, V is a m times k matrix, and epsilon sim N(0 sigma^2). We start by setting up the desired dimensions for the simulated data.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"# set dimensions\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\n# covariance matrices\nΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\n\n\nD = [40, 20, 10, 5, 2] # sqrt of eigenvalues\nk = 5 # number of basis functions \nϵ = 2 # noise\n\nRandom.seed!(2)\nU, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)\nnothing # hide","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"We can then plot the spatial basis functions","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Here is a plot of the spatial basis functions.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Plots.plot(x, U, xlabel = \"Space\", ylabel = \"Value\", label = [\"U\" * string(i) for i in (1:k)'])","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"and temporal basis functions.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Plots.plot(t, V, xlabel = \"Time\", ylabel = \"Value\", label = [\"V\" * string(i) for i in (1:k)'])","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"Last, we can look at the simulated smooth and noisy data.","category":"page"},{"location":"simulateData/","page":"Examples","title":"Examples","text":"l = Plots.@layout [a b]\np1 = Plots.contourf(x, t, Y', clim = (-1.05, 1.05).*maximum(abs, Z), title = \"Smooth\", c = :balance)\np2 = Plots.contourf(x, t, Z', clim = (-1.05, 1.05).*maximum(abs, Z), title = \"Noisy\", c = :balance)\nPlots.plot(p1, p2, layout = l, size = (1000, 400), margin = 5Plots.mm, xlabel = \"Space\", ylabel = \"Time\")","category":"page"},{"location":"api/","page":"API","title":"API","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [BayesianSVD]","category":"page"},{"location":"api/#BayesianSVD.BayesianSVD","page":"API","title":"BayesianSVD.BayesianSVD","text":"BayesianSVD\n\nHere is my package.\n\n\n\n\n\n","category":"module"},{"location":"api/#BayesianSVD.ARCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.ARCorrelation","text":"ARCorrelation(t; ρ = 1, metric = Euclidean())\n\nCreate an AR(1) correlation matrix of type ARCorrelation <: DependentCorrelation <: Correlation.\n\nSee also IdentityCorrelation, ExponentialCorrelation, GaussianCorrelation, and MaternCorrelation.\n\nArguments\n\nt: vector of times\n\nOptional Arguments\n\nρ = 1: correlation parameter\nmetric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = SparseCorrelation(x, ρ = 1, metric = Euclidean())\nΩ = SparseCorrelation(x, y, ρ = 1, metric = Euclidean())\nΩ = SparseCorrelation(X', ρ = 1, metric = Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.Data-Tuple{Matrix{Float64}, Matrix{Float64}, Any, Any, Int64}","page":"API","title":"BayesianSVD.Data","text":"Data(Y, ulocs, vlocs, k)\nData(Y, X, ulocs, vlocs, k)\n\nCreates the data class.\n\nSee also Pars, Posterior, and SampleSVD.\n\nArguments\n\nY: data of dimension n × m\nX: covariate matrix of dimension nm × p\nulocs: locations for the U basis functions, corresponds to the locations for the rows of Y\nvlocs: locations for the V basis functions, corresponds to the locations for the columns of Y\nk: number of basis functions to keep\n\nExamples\n\nk = 5\nΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\ndata = Data(Z, x, t, k)\npars = Pars(data, ΩU, ΩV)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.ExponentialCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.ExponentialCorrelation","text":"ExponentialCorrelation(x; ρ = 1, metric = Euclidean())\nExponentialCorrelation(x, y; ρ = 1, metric = Euclidean())\nExponentialCorrelation(X; ρ = 1, metric = Euclidean())\n\nCreate an Exponential correlation matrix of type ExponentialCorrelation <: DependentCorrelation <: Correlation.\n\nSee also IdentityCorrelation, GaussianCorrelation, MaternCorrelation, and SparseCorrelation.\n\nArguments\n\nx: vector of locations\ny: vector of locations for a second dimension\nX: matrix of all pairwise locations\n\nOptional Arguments\n\nρ = 1: length-scale parameter\nmetric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = ExponentialCorrelation(x, ρ = 1, metric = Euclidean())\nΩ = ExponentialCorrelation(x, y, ρ = 1, metric = Euclidean())\nΩ = ExponentialCorrelation(X', ρ = 1, metric = Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.GaussianCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.GaussianCorrelation","text":"GaussianCorrelation(x; ρ = 1, metric = Euclidean())\nGaussianCorrelation(x, y; ρ = 1, metric = Euclidean())\nGaussianCorrelation(X; ρ = 1, metric = Euclidean())\n\nCreate an Gaussian correlation matrix of type GaussianCorrelation <: DependentCorrelation <: Correlation.\n\nSee also IdentityCorrelation, ExponentialCorrelation, MaternCorrelation, and SparseCorrelation.\n\nArguments\n\nx: vector of locations\ny: vector of locations for a second dimension\nX: matrix of all pairwise locations\n\nOptional Arguments\n\nρ = 1: length-scale parameter\nmetric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = GaussianCorrelation(x, ρ = 1, metric = Euclidean())\nΩ = GaussianCorrelation(x, y, ρ = 1, metric = Euclidean())\nΩ = GaussianCorrelation(X', ρ = 1, metric = Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.IdentityCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.IdentityCorrelation","text":"IdentityCorrelation(x)\nIdentityCorrelation(x, y)\nIdentityCorrelation(X)\n\nCreate an Identity correlation matrix of type IdentityCorrelation <: IndependentCorrelation <: Correlation.\n\nSee also ExponentialCorrelation, GaussianCorrelation, MaternCorrelation, and SparseCorrelation.\n\nArguments\n\nx: vector of locations\ny: vector of locations for a second dimension\nX: matrix of all pairwise locations\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = IdentityCorrelation(x)\nΩ = IdentityCorrelation(x, y)\nΩ = IdentityCorrelation(X')\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.MaternCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.MaternCorrelation","text":"MaternCorrelation(x; ρ = 1, ν = 3.5 metric = Euclidean())\nMaternCorrelation(x, y; ρ = 1, ν = 3.5, metric = Euclidean())\nMaternCorrelation(X; ρ = 1, ν = 3.5, metric = Euclidean())\n\nCreate an Matern correlation matrix of type MaternCorrelation <: DependentCorrelation <: Correlation.\n\nK_nu rho(s s) = frac2^1-nuGamma(nu)left(2nu fracs-srhoright)^nuJ_nuleft(2nu fracs-srhoright)\n\nfor ss in mathcalS, where Gamma is the gamma function, J_nu is the Bessel function of the second kind, and rho nu  are hyperparameters that describe the length-scale and differentiability, respectively.\n\nSee also IdentityCorrelation, ExponentialCorrelation, GaussianCorrelation, and SparseCorrelation.\n\nArguments\n\nx: vector of locations\ny: vector of locations for a second dimension\nX: matrix of all pairwise locations\n\nOptional Arguments\n\nρ = 1: length-scale parameter\nν = 3.5: smoothness parameter\nmetric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = MaternCorrelation(x, ρ = 1, ν = 3.5, metric = Euclidean())\nΩ = MaternCorrelation(x, y, ρ = 1, ν = 3.5, metric = Euclidean())\nΩ = MaternCorrelation(X', ρ = 1, ν = 3.5, metric = Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.Pars-Tuple{BayesianSVD.MixedEffectData, Correlation, Correlation}","page":"API","title":"BayesianSVD.Pars","text":"Pars(data::Data, ΩU::Correlation, ΩV::Correlation)\n\nCreates the parameter class of type typeof(data).'\n\nSee also Data, Posterior, and SampleSVD.\n\nArguments\n\ndata: data structure of type Data\nΩU: data structure of type Correlation\nΩV: data structure of type Correlation\n\nExamples\n\nk = 5\nΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\ndata = Data(Z, x, t, k)\npars = Pars(data, ΩU, ΩV)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.Posterior","page":"API","title":"BayesianSVD.Posterior","text":"Posterior\n\nStructure of type posterior with subtypes Identity, Exponential, Gaussian, or Matern. Contains the raw posterior samples and some means and 95% quantiles of parameters. Plotting associated with the structure.\n\nSee also Pars, Data, and SampleSVD.\n\nExamples\n\nk = 5\nΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\ndata = Data(Z, x, t, k)\npars = Pars(data, ΩU, ΩV)\n\nposterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)\n\nplot(posterior, x, size = (900, 600), basis = 'U', linewidth = 2, c = [:red :green :purple])\nplot(posterior, t, size = (900, 500), basis = 'V', linewidth = 2, c = [:red :green :purple])\n\n# for spatial basis functions, provide x and y\nplot(posterior, x, y)\n\n\n\n\n\n","category":"type"},{"location":"api/#BayesianSVD.SparseCorrelation-Tuple{Any}","page":"API","title":"BayesianSVD.SparseCorrelation","text":"SparseCorrelation(x; ρ = 1, metric = Euclidean())\nSparseCorrelation(x, y; ρ = 1, metric = Euclidean())\nSparseCorrelation(X; ρ = 1, metric = Euclidean())\n\nCreate an Gaussian correlation matrix of type SparseCorrelation <: DependentCorrelation <: Correlation.\n\nSee also IdentityCorrelation, ExponentialCorrelation, GaussianCorrelation, and MaternCorrelation.\n\nArguments\n\nx: vector of locations\ny: vector of locations for a second dimension\nX: matrix of all pairwise locations\n\nOptional Arguments\n\nρ = 1: length-scale parameter\nmetric = Euclidean(): metric used for computing the distance between points. All distances in Distances.jl are supported.\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\nX = reduce(hcat,reshape([[x, y] for x = x, y = y], Nx * Ny))'\n\nΩ = SparseCorrelation(x, ρ = 1, metric = Euclidean())\nΩ = SparseCorrelation(x, y, ρ = 1, metric = Euclidean())\nΩ = SparseCorrelation(X', ρ = 1, metric = Euclidean())\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.CreateDesignMatrix-Tuple{Int64, Int64, Any, Any, Any}","page":"API","title":"BayesianSVD.CreateDesignMatrix","text":"CreateDesignMatrix(n::Int, m::Int, Xt, Xs, Xst; intercept = true)\n\nCreates a design matrix.\n\nSee also Data, Posterior, and SampleSVD.\n\nArguments\n\nn::Int: Number of spatial locations\nm::Int: Number of temporal locations\nXt: Time only covariates of dimension m x pₘ\nXs: Spatial only covariates of dimension n x pₙ\nXst: Space-time only covariates of dimension mn x pₘₙ\nintercept = true: should a global intercept be included\n\nReturns\n\nX: Design matrix of dimension mn x (pₘ + pₙ + pₘₙ) or mn x (1 + pₘ + pₙ + pₘₙ)\n\nExamples\n\nm = 75\nn = 50\n\n#### Spatial\nXs = hcat(sin.((2*pi .* x) ./ 10), cos.((2*pi .* x) ./ 10))\n\n#### Temporal\nXt = hcat(Vector(t))\n\n#### Spatio-temporal\nXst = rand(Normal(), n*m, 2)\n\n#### Create design matrix\nCreateDesignMatrix(n, m, Xt, Xs, Xst, intercept = true)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.GenerateData-Tuple{Correlation, Correlation, Any, Any, Any}","page":"API","title":"BayesianSVD.GenerateData","text":"GenerateData(ΣU::Correlation, ΣV::Correlation, D, k, ϵ; SNR = false)   GenerateData(ΣU::Vector{T}, ΣV::Vector{T}, D, k, ϵ; SNR = false) where T <: Correlation\n\nGenerate random basis functions and data given a correlation structure for U and V.\n\nArguments\n\nΣU::Correlation\nΣU::Correlation\nD: vector of length k\nk: number of basis functions\nϵ: standard devaiation of the noise, if SNR = true then ϵ is the signal to noise ratio\n\nOptional Arguments\n\nSNR = false: Boolean for if ϵ is standard deviation (false) or signal to noise ratio value (true)\n\nReturns\n\nU: U basis functions\nV: V basis functions\nY: True smooth surface\nZ: Noisy \"observed\" surface\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\n\n\nD = [40, 30, 20, 10, 5]\nk = 5\n\nRandom.seed!(2)\nU, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 0.1) # standard deviation\nU, V, Y, Z = GenerateData(ΣU, ΣV, D, k, 2, SNR = true) # signal to noise\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.PON-Tuple{Any, Any, Any}","page":"API","title":"BayesianSVD.PON","text":"PON(n, k, Σ)\n\nCreate an n by k orthonormal matrix with covariance matrix Sigma.\n\nArguments\n\nn: Number of locations \nk: Number of basis functions\nΣ: covariance matrix\n\nExamples\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternKernel(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΣV = MaternKernel(t, ρ = 3, ν = 3.5, metric = Euclidean())\n\nk = 5\nΦ = PON(n, k, ΣU.K)\nΨ = PON(n, k, ΣV.K)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.SampleSVD-Tuple{BayesianSVD.MixedEffectData, BayesianSVD.MixedEffectPars}","page":"API","title":"BayesianSVD.SampleSVD","text":"SampleSVD(data::Data, pars::Pars; nits = 10000, burnin = 5000, show_progress = true)\n\nRuns the MCMC sampler for the Bayesian SVD model.\n\nSee also Pars, Data, and Posterior.\n\nArguments\n\ndata: Data structure of type Identity, Exponential, Gaussian, or Matern\npars: Parameter structure of type Identity, Exponential, Gaussian, or Matern\n\nOptional Arguments\n\nnits = 10000: Total number of posterior samples to compute\nburnin = 5000: Number of samples discarded as burnin\nshow_progress = true: Indicator on whether to show a progress bar (true) or not (false).\n\nExamples\n\n\nRandom.seed!(2)\n\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\nΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\n\n\nD = [40, 30, 20, 10, 5]\nk = 5\nϵ = 2\n\nU, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)\n\n\nΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\ndata = Data(Z, x, t, k)\npars = Pars(data, ΩU, ΩV)\n\nposterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)\n\n\n\n\n\n","category":"method"},{"location":"api/#BayesianSVD.hpd-Tuple{Any}","page":"API","title":"BayesianSVD.hpd","text":"hpd(x; p=0.95)\n\nComputes the highest posterior density interval of x.\n\nArguments\n\nx: vector of data.\n\nOptional Arguments\n\np: value between 0 and 1 for the probability level.\n\nExamples\n\nx = [1, 3, 2, 5, .2, 1,9]\nhpd(x)\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"#BayesianSVD","page":"Home","title":"BayesianSVD","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BayesianSVD.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Our goal is to sample from the model","category":"page"},{"location":"","page":"Home","title":"Home","text":"Z = X beta + UDV + epsilon","category":"page"},{"location":"","page":"Home","title":"Home","text":"where Z is a n (space) by m (time) matrix, X beta is the mean of the process where X is a matrix of covariates, U is a n times k matrix, V is a m times k matrix, D is a k times k and epsilon sim N(0 sigma^2). Here, M = X beta is a fixed effect and Y = UDV is a random effect.","category":"page"},{"location":"#General-Workflow","page":"Home","title":"General Workflow","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The model is broken down into two pieces - Data and Pars - which are then passed in as arguments to a sampling function which performs MCMC and returns Posterior.","category":"page"},{"location":"#Setting-up-the-data","page":"Home","title":"Setting up the data","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To set up the model, we first set up our Data class, which contains subtypes MixedEffectData and RandomEffectData. The difference in the two is RandomEffectData assumes M equiv 0 and is not to be estimated.","category":"page"},{"location":"","page":"Home","title":"Home","text":"To set up a MixedEffectData model, we would use","category":"page"},{"location":"","page":"Home","title":"Home","text":"data = Data(Z, X, rowLocations, columnLocations, numberBasisFunctions)","category":"page"},{"location":"","page":"Home","title":"Home","text":"and To set up a RandomEffectData model, we would use","category":"page"},{"location":"","page":"Home","title":"Home","text":"data = Data(Z, rowLocations, columnLocations, numberBasisFunctions)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where now X is omitted from the function call.","category":"page"},{"location":"#Setting-up-the-parameters","page":"Home","title":"Setting up the parameters","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The second thing is to set up the Pars class. This is the same for both the MixedEffectData and RandomEffectData, and requires three arguments: data, the correlation matrix for U, and the correlation matrix for V. The function Pars() is simply a constructor for the Pars class and will generate k correlation matrices for U and V based on the supplied matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\npars = Pars(data, ΩU, ΩV)","category":"page"},{"location":"#Sampling","page":"Home","title":"Sampling","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To sample from the model, use the SampleSVD() function, supplying the data and pars as inputs. This will return a structure of class Posterior which has its own properties.","category":"page"},{"location":"","page":"Home","title":"Home","text":"posterior, pars = SampleSVD(data, pars; nits = 1000, burnin = 500)","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"CurrentModule = BayesianSVD","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"using BayesianSVD\nusing Distances, Plots, Random, Distributions, LinearAlgebra","category":"page"},{"location":"example/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"example/#Simulate-Data","page":"Examples","title":"Simulate Data","text":"","category":"section"},{"location":"example/","page":"Examples","title":"Examples","text":"The function GenerateData() will simulate data from the model","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Z = UDV + epsilon","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"where Z is a n (space) by m (time) matrix, U is a n times k matrix, V is a m times k matrix, D is a k times k and epsilon sim N(0 sigma^2). We start by setting up the desired dimensions for the simulated data.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"# set dimensions\nm = 100\nn = 100\nx = range(-5, 5, n)\nt = range(0, 10, m)\n\n# covariance matrices\nΣU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean())\nΣV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean())\n\n\nD = [40, 20, 10, 5, 2] # sqrt of eigenvalues\nk = 5 # number of basis functions \nϵ = 2 # noise\n\nRandom.seed!(3)\nU, V, Y, Z = GenerateData(ΣU, ΣV, D, k, ϵ, SNR = true)\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"We can then plot the spatial basis functions","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Here is a plot of the spatial basis functions.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Plots.plot(x, U, xlabel = \"Space\", ylabel = \"Value\", label = [\"U\" * string(i) for i in (1:k)'])","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"and temporal basis functions.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Plots.plot(t, V, xlabel = \"Time\", ylabel = \"Value\", label = [\"V\" * string(i) for i in (1:k)'])","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Last, we can look at the simulated smooth and noisy data.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"l = Plots.@layout [a b]\np1 = Plots.contourf(x, t, Y', clim = (-1.05, 1.05).*maximum(abs, Z), title = \"Smooth\", c = :balance)\np2 = Plots.contourf(x, t, Z', clim = (-1.05, 1.05).*maximum(abs, Z), title = \"Noisy\", c = :balance)\nPlots.plot(p1, p2, layout = l, size = (1000, 400), margin = 5Plots.mm, xlabel = \"Space\", ylabel = \"Time\")","category":"page"},{"location":"example/#D-space-and-1-D","page":"Examples","title":"1-D space and 1-D","text":"","category":"section"},{"location":"example/","page":"Examples","title":"Examples","text":"To sample from the Bayesian SVD model we use the SampleSVD() function (see ?SampleSVD() for help). This function requires two parameters - data::Data and pars::Pars. The parameter function is easy, is requres the one arguemt data::Data. The data function requires 4 arguments - (1) the data matrix, (2) covariance matrix ΩU::KernelFunction for the U basis matrix, (3) covariance matrix ΩV::KernelFunction for the V basis matrix, and (4) the number of basis functions k.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"k = 5\nΩU = MaternCorrelation(x, ρ = 3, ν = 3.5, metric = Euclidean()) # U covariance matrix\nΩV = MaternCorrelation(t, ρ = 3, ν = 3.5, metric = Euclidean()) # V covariance matrix\ndata = Data(Z, x, t, k) # data structure\npars = Pars(data, ΩU, ΩV) # parameter structure\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"We are now ready to sample from the model. Note, we recommend show_progress = false when running in a notebook and show_progress = true if you have output print in the REPL. Also, the sampler is slow in the notebooks but considerably faster outside of them.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"posterior, pars = SampleSVD(data, pars; nits = 10, burnin = 5, show_progress = false)\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Because the basis functions are only identifiable up to the sign, we need to orient them.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"trsfrm = ones(k)\nfor l in 1:k\n    if posteriorCoverage(U[:,l], posterior.U[:,l,:], 0.95) > posteriorCoverage(-U[:,l], posterior.U[:,l,:], 0.95)\n        continue\n    else\n        trsfrm[l] = -1.0\n    end\nend","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"We can now plot the output of the spatial basis functions","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Plots.plot(posterior, x, size = (800, 400), basis = 'U', linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = \"U\")\nPlots.plot!(x, (U' .* trsfrm)', label = false, color = \"black\", linewidth = 2)\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"(Image: Ubasis)","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"And the temporal basis functions.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Plots.plot(posterior, t, size = (800, 400), basis = 'V',  linewidth = 2, c = [:blue :red :magenta :orange :green], tickfontsize = 14, label = false, title = \"V\")\nPlots.plot!(t, (V' .* trsfrm)', label = false, color = \"black\", linewidth = 2)\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"(Image: Vbasis) ","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"Last, we can look at the difference between the target smooth surface, our estimate, and the algorithmic estimate.","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"\nsvdZ = svd(Z).U[:,1:k] * diagm(svd(Z).S[1:k]) * svd(Z).V[:,1:k]'\nZ_hat = posterior.U_hat * diagm(posterior.D_hat) * posterior.V_hat'\nZ_hat_diff = mean([posterior.U[:,:,i] * diagm(posterior.D[:,i]) * posterior.V[:,:,i]' .- Y for i in axes(posterior.U,3)])\n\nlims = (-1.05, 1.05).*maximum(abs, Z)\n\nl = Plots.@layout [a b c; d e f]\np1 = Plots.contourf(t, x, Z_hat, title = \"Y Hat\", c = :balance, clim = lims)\np2 = Plots.contourf(t, x, Y, title = \"Truth\", c = :balance, clim = lims)\np3 = Plots.contourf(t, x, svdZ, title = \"Algorithm\", c = :balance, clim = lims)\np4 = Plots.contourf(t, x, Z_hat .- Y, title = \"Y Hat - Truth\", c = :balance, clim = (-0.2, 0.2))\np5 = Plots.contourf(t, x, Z, title = \"Observed\", c = :balance, clim = lims)\np6 = Plots.contourf(t, x, svdZ .- Y, title = \"Algorithm - Truth\", c = :balance, clim = (-0.2, 0.2))\nPlots.plot(p1, p2, p3, p4, p5, p6, layout = l, size = (1400, 600))\nnothing # hide","category":"page"},{"location":"example/","page":"Examples","title":"Examples","text":"(Image: surfaceEstimate)","category":"page"}]
}
