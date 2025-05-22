using RandomHAL
using Pkg
Pkg.activate("test")
using Random
using Test
using CausalTables
using Distributions
using MLJ
import LogExpFunctions: logistic

using Tables
import Combinatorics: powerset

Random.seed!(1234)
dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2) - 2.5))),
        Y ~ (@. Normal(A + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.2))
    )
scm = StructuralCausalModel(dgp, :A, :Y)

n = 400
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))

function test_basis(smoothness, diff)
    basis, term_lengths = ha_basis_matrix(X, smoothness; diff = diff)

    @test size(basis)[1] == n
    @test size(basis)[2] == n * (2^length(X) - 1) - (n - 1)
    @test basis[:, (n * 3) + 1] == ct.data.A
    lasso, β, β0, nz = fit_glmnet(basis, y::AbstractVector, Normal(); nlambda = 100, nfolds = 10)

    sections, knots = get_sections_and_knots(X, nz, term_lengths)
    @test length.(sections) == length.(knots)
    hab = ha_basis_matrix(X, sections, knots, smoothness; diff = diff)
    @test all([basis[:, nz[i]] == hab[:, i] for i in 1:length(nz)])
end

@testset "Basis creation functions" begin
    test_basis(0, false)
    test_basis(1, false)
    test_basis(2, false)

    test_basis(0, true)
    test_basis(1, true)
    test_basis(2, true)

    # Random Basis
    nfeatures = Int(round(n*log(n)))
    sections_r, knots_r = random_sections_and_knots(X, nfeatures)
    basis_r = ha_basis_matrix(X, sections_r, knots_r, 0)

    @test length(sections_r) == nfeatures
    @test length.(sections_r) == length.(knots_r)
end

#@testset "Model Fitting" begin

    cttest = rand(scm, n)
    Xtest = responseparents(cttest)
    ytest = vec(responsematrix(cttest))

    cttestA = intervene(cttest, treat_all)
    XAtest = responseparents(cttestA)
    yAtest = vec(responsematrix(cttestA))

    # HAL
    model = HALRegressor()
    @time hal = machine(model, X, y) |> fit!

    true_mean = conmean(scm, cttest, :Y)
    halpreds = MLJ.predict(hal, Xtest)
    halmse = mean((halpreds .- true_mean).^2)

    @test halmse < 0.1

    ### Extra diff 
    model = HALRegressor(0, true)
    @time hal_d = machine(model, X, y) |> fit!

    halpreds_d = MLJ.predict(hal_d, Xtest)
    halmse = mean((halpreds_d .- true_mean).^2)

    using Plots
    scatter(true_mean, [halpreds halpreds_d])

    ###


    # Random HAL

    model3 = RandomHALRegressor()
    
    @time rhal = machine(model3, X, y) |> fit!

    rhalpreds = MLJ.predict(rhal, Xtest)
    rhalmse = mean((rhalpreds .- true_mean).^2)

    # RMSE are bounded
    @test rhalmse < 0.1
    
    # Binary treatment
    Xbin = treatmentparents(ct)
    Xbintest = treatmentparents(cttest)
    A = vec(treatmentmatrix(ct))
    Atest = vec(treatmentmatrix(cttest))
    modelbin = HALBinaryClassifier()
    @time halbin = fit!(machine(modelbin, Xbin, A))

    halpredsbin = MLJ.predict(halbin, Xbintest)
    halbinmse = mean((halpredsbin .- conmean(scm, cttest, :A)).^2)
    
    @time rhalbin = machine(RandomHALBinaryClassifier(), Xbin, A) |> fit!
    rhalpredsbin = MLJ.predict(rhalbin, Xbintest)
    rhalbinmse = mean((rhalpredsbin .- conmean(scm, cttest, :A)).^2)

    @test halbinmse < 0.1
    @test rhalbinmse < 0.1

end
