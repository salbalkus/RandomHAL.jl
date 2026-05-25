using Test
using Random
using CausalTables
using Distributions
using StatsBase
using LinearAlgebra
using MLJBase
import LogExpFunctions: logistic

using Profile

using Tables
import Combinatorics: powerset

using RandomHAL

Random.seed!(1234)
dgp = @dgp(
        X1 ~ Beta(1, 1),
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.1))
    )

#using Copulas
#d = 3
#d_first = 3
#ρ = 0.05
#treat_shift = 2
#dgp = @dgp(
#        L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Beta(2,2), d))),
#        μ = (1 .+ 2 .* L[:, 1]) .* vec(mean(L[:,2:d_first] .- L[:,2:d_first] .^ (1/2), dims = 2)) .+ 0.5,
#        A ~ Bernoulli.(logistic.(2.5 .* μ)),
#        Y ~ Normal.((1 .+ treat_shift .* A) .* μ .+ 2, sqrt(0.5))
#    )

scm = StructuralCausalModel(dgp, :A, :Y)
n = 100
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))

function test_basis(smoothness)
    basis, all_sections, term_lengths = ha_basis_matrix(X, smoothness)#; basis_type = basis_type)
    @test size(basis)[1] == n
    #@test size(basis)[2] == n * (2^length(X) - 1) - (n - 1)
    @test basis[:, (n * (2^(length(X)-1)-1)) + 1] == ct.data.A
    lasso, β, β0, nz = fit_glmnet(basis, y::AbstractVector, Normal(); nlambda = 100, nfolds = 10)

    sections, knots = get_sections_and_knots(X, nz, all_sections, term_lengths)
    @test length.(sections) == length.(knots)
    hab = ha_basis_matrix(X, sections, knots, smoothness)
    @test all([basis[:, nz[i]] == hab[:, i] for i in 1:length(nz)])
end

@testset "Basis creation functions" begin
    test_basis(0)
    test_basis(1)
    test_basis(2)

    # Random Basis
    nfeatures = Int(round(n*log(n)))
    sections_r, knots_r = random_sections_and_knots(X, nfeatures)
    basis_r = ha_basis_matrix(X, sections_r, knots_r, 0)

    @test length(sections_r) == nfeatures
    @test length.(sections_r) == length.(knots_r)
end

@testset "Model Fitting" begin

    cttest = rand(scm, n)
    Xtest = responseparents(cttest)
    ytest = vec(responsematrix(cttest))

    cttestA = intervene(cttest, treat_all)
    XAtest = responseparents(cttestA)
    yAtest = vec(responsematrix(cttestA))
    true_mean = conmean(scm, cttest, :Y)

    # HAL
    model = HALRegressor(1)
    @time hal = machine(model, X, y) |> fit!

    halpreds = MLJBase.predict(hal, Xtest)
    halmse = mean((halpreds .- true_mean).^2)

    @test halmse < 0.05

    # Random HAL
    n_samples = Int(round(0.5 * n * log(n)))

    #model_rhal = RandomHALRegressor(0, 100, 5, n_samples, (guaranteed_sections = [[5]], interaction_order_weights = Weights([8, 4, 2, 1])))
    model_rhal = RandomHALRegressor(1, n_samples, (guaranteed_sections = [[4]],))
    @time rhal = machine(model_rhal, X, y) |> fit!

    rhalpreds = MLJBase.predict(rhal, Xtest)
    rhalmse = mean((rhalpreds .- true_mean).^2)

    # RMSE are bounded
    @test rhalmse < 0.05
    
    # Binary treatment
    Xbin = treatmentparents(ct)
    Xbintest = treatmentparents(cttest)
    A = vec(treatmentmatrix(ct))
    Atest = vec(treatmentmatrix(cttest))
    modelbin = HALBinaryClassifier(1, 0)
    trueprop = conmean(scm, cttest, :A)
    @time halbin = fit!(machine(modelbin, Xbin, A))

    halpredsbin = MLJBase.predict(halbin, Xbintest)
    halbinmse = mean((halpredsbin .- trueprop).^2)

    
    #min_nonzeros = Int(floor(sqrt(n)))
    model_brhal = RandomHALBinaryClassifier(1, n_samples, NamedTuple())
    @time rhalbin = machine(model_brhal, Xbin, A) |> fit!
    rhalpredsbin = MLJBase.predict(rhalbin, Xbintest)
    rhalbinmse = mean((rhalpredsbin .- trueprop).^2)

    @test halbinmse < 0.05
    @test rhalbinmse < 0.05

end
