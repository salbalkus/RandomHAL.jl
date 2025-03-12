using RandomHAL
using Test
using CausalTables
using Distributions
using MLJ
import LogExpFunctions: logistic

@testset "Model Fitting" begin

    dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2) - 2.5))),
        Y ~ (@. Normal(A + sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4) + 5, 1.0))
    )
    scm = StructuralCausalModel(dgp, :A, :Y)

    # Generate and extract necessary data
    n = 500
    ct = rand(scm, n)
    cttest = rand(scm, n)
    X = responseparents(ct)
    Xtest = responseparents(cttest)
    y = responsematrix(ct)[:, 1]
    ytest = responsematrix(cttest)[:, 1]

    cttestA = intervene(cttest, treat_all)
    XAtest = responseparents(cttestA)
    yAtest = responsematrix(cttest)[:, 1]

    # HAL
    model = HALRegressor(0, 100, 5, [])
    model2 = HALRegressor(0, 100, 5, [:A])

    @time hal = machine(model, X, y) |> fit!
    @time hal2 = machine(model2, X, y) |> fit!

    true_mean = conmean(scm, cttest, :Y)
    halpreds = MLJ.predict(hal, Xtest)
    halmse = mean((halpreds .- true_mean).^2)

    halpreds2 = MLJ.predict(hal2, Xtest)
    halmse2 = mean((halpreds2 .- true_mean).^2)


    @test halmse < 1.5
    @test halmse2 < 1.5
    @test [4] ∈ hal2.fitresult.sections

    scatter(halpreds, conmean(scm, cttest, :Y))
    scatter(halpreds2, conmean(scm, cttest, :Y))

    model3 = RandomHALRegressor(0, round(n^(5/4)), 100, 5, [], [])
    model4 = RandomHALRegressor(0, round(n^(5/4)), 100, 5, [:A], [:A])
    
    @time rhal = machine(model3, X, y) |> fit!
    @time rhal2 = machine(model4, X, y) |> fit!

    rhalpreds = MLJ.predict(rhal, Xtest)
    rhalmse = mean((rhalpreds .- ytest).^2)

    rhalpreds2 = MLJ.predict(rhal2, Xtest)
    rhalmse2 = mean((rhalpreds2 .- ytest).^2)

    scatter(rhalpreds, conmean(scm, cttest, :Y))
    scatter(rhalpreds2, conmean(scm, cttest, :Y))

    # RMSE are bounded
    @test sqrt(rhalmse) < 1.5
    @test sqrt(rhalmse2) < 1.5

    @test [4] ∈ rhal2.fitresult.sections
    
    # Binary treatment
    Xbin = treatmentparents(ct)
    Xbintest = treatmentparents(cttest)
    A = treatmentmatrix(ct)[:, 1]
    Atest = treatmentmatrix(cttest)[:, 1]
    modelbin = HALBinaryClassifier(0)
    @time halbin = fit!(machine(modelbin, Xbin, A))

    halpredsbin = MLJ.predict(halbin, Xbintest)
    halbinmse = mean((halpredsbin .- conmean(scm, cttest, :A)).^2)
    
    scatter(conmean(scm, cttest, :A), halpredsbin)

    @time rhalbin = machine(RandomHALBinaryClassifier(0, round(n^(5/4))), Xbin, A) |> fit!
    rhalpredsbin = MLJ.predict(rhalbin, Xbintest)
    rhalbinmse = mean((rhalpredsbin .- conmean(scm, cttest, :A)).^2)

    scatter(conmean(scm, cttest, :A), rhalpredsbin)

    @test sqrt(halbinmse) < 0.5
    @test sqrt(rhalbinmse) < 0.5

end