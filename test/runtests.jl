using RandomHAL
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

n = 100
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))

@testset "Basis creation functions" begin
    basis, term_lengths = ha_basis_matrix(X, 0)

    @test typeof(basis) == BitMatrix
    @test size(basis)[1] == n
    @test size(basis)[2] == n * (2^length(X) - 1) - (n - 1)
    @test basis[:, (n * 3) + 1] == ct.data.A
    lasso, β, β0, nz = fit_glmnet(basis, y::AbstractVector, Normal(); nlambda = 100, nfolds = 10)
    @test 301 ∈ nz

    sections, knots = get_sections_and_knots(X, nz, term_lengths)
    @test [4] ∈ sections
    @test [true] ∈ knots
    @test length.(sections) == length.(knots)

    nfeatures = Int(round(n*log(n)))
    sections, knots = random_sections_and_knots(X, nfeatures)
    basis = ha_basis_matrix(X, sections, knots, 0)

    @test length(sections) == nfeatures
    @test length.(sections) == length.(knots)
end

@testset "Model Fitting" begin

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
