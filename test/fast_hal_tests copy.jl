using Test
using Tables
using CausalTables
using Distributions
import LogExpFunctions: logistic
using StatsBase
using DecisionTree
using LinearAlgebra
using Random
using GLMNet
using MLJBase
using Combinatorics: combinations
using RandomHAL
using Copulas

Random.seed!(1234)

function binary_scm(d, d_first, ρ)

    dgp = @dgp(
        L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Uniform(), d))),
        g = vec(mean(sin.(2 .* pi .* L[:, 1:d_first]), dims = 2)),
        A ~ Bernoulli.(logistic.(4 .* g)),
        Y ~ Normal.((3 .+ A) .* g, 0.2)
    )

    scm = StructuralCausalModel(dgp, :A, :Y)
    return scm
end

scm = binary_scm(40, 40, 0.1)
n = 200
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
Xm = Tables.matrix(X)
y = vec(responsematrix(ct))
true_conmean = conmean(scm, ct, :Y)

Xa = Tables.Columns(treatmentparents(ct))
Xma = Tables.matrix(Xa)
A = vec(treatmentmatrix(ct))
true_pr = conmean(scm, ct, :A)

cttest = rand(scm, n)
Xtest = Tables.Columns(responseparents(cttest))
Xmtest = Tables.matrix(Xtest)
ytest = vec(responsematrix(cttest))

Xatest = Tables.Columns(treatmentparents(cttest))
Xmatest = Tables.matrix(Xatest)
Atest = vec(treatmentmatrix(cttest))
true_pr_test = conmean(scm, cttest, :A)
true_conmean_test = conmean(scm, cttest, :Y)


# Test NestedMatrix functionality
@testset "NestedMatrix" begin
    i = 1
    all_ranks = reduce(hcat, map(competerank, eachcol(Xma)))

    # Test the path sampler
    @test all_ranks[path_sample(all_ranks, [i]), 1] == 1:n

    # Construct nested matrix
    indicator = NestedIndicators(all_ranks, [i], Xma)
    eye = Matrix(I, n, n)
    # NestedMatrix
    B = NestedMatrix(indicator, Xm)

    # Construct the "true" sort
    pa = path_sample(all_ranks, [i])
    B_true = (Xm[:, i] .>= Xm[pa, i]')

    @test B_true * ones(n) == B * ones(B.ncol)
    @test all([col in collect(eachcol(B_true)) for col in eachcol(B * eye)])

    # Now check interaction
    j = 2
    # Check to make sure all sampled paths are nested within each other
    S = [i, j]
    pa = path_sample(all_ranks, S)
    @test all([all_ranks[pa[i-1], 1] < all_ranks[pa[i], 1] for i in 2:length(pa)])
    @test all([all_ranks[pa[i-1], 2] < all_ranks[pa[i], 2] for i in 2:length(pa)])

    indicator = NestedIndicators(all_ranks, [i, j], Xma)
    B = NestedMatrix(indicator, Xm)
    B_true = (Xm[:, j] .>= Xm[pa, j]') .* (Xm[:, i] .>= Xm[pa, i]')

    # Are the bins constructed correctly?
    bins = indicator.bins
    order = binary_bin_search(Xma[:, [i, j]], bins)
    @test order == [sum(all(bins .<= row', dims = 2)) for row in eachrow(Xma[:, [i, j]])]
    @test B_true * ones(size(B_true, 2)) == B * ones(B.ncol)

    matrix_cols = collect(eachcol(B_true))
    basis_cols = eachcol(B * Matrix(I, B.ncol, B.ncol))
    @test all([col in matrix_cols for col in basis_cols])

    # Test if B is a "mirrored" version of B_true
    @test Int.(B * Matrix(I, B.ncol, B.ncol)) == B_true[: , end:-1:1]

    # NestedMatrixTranspose
    Bt = transpose(B)
    v = rand(Bt.ncol)
    @test Bt * ones(n) ≈ reverse(transpose(B_true) * ones(n))
    @test Bt * v ≈ reverse(transpose(B_true) * v)

    # NestedMatrixBlocks
    S = [[i], [i, j]]
    indb = NestedIndicatorBlocks(S, Xm)
    Bb = NestedMatrixBlocks(indb, Xm)

    ## Not there's no real ground truth here since we sample a random path,
    ## plus we already tested the individual blocks, so we'll just test 
    # that the additional block components function as expected
    @test Bb.ncol == sum(Bb.blocks[i].ncol for i in 1:length(Bb.blocks))
    @test Bb.nrow == n
    @test all(sort(Bb * ones(Bb.ncol)) .< Bb.ncol)

    Bb * ones(Bb.ncol)

    # NestedMatrixBlocksTranspose
    Bbt = transpose(Bb)
    @test Bbt.ncol == n
    @test Bbt.nrow == Bb.ncol
    @test all(sort(Bbt * ones(Bbt.ncol)) .< Bbt.nrow)
end

# Test BasisMatrix functionality
@testset "BasisMatrix" begin
    all_ranks = reduce(hcat, map(competerank, eachcol(Xm)))
    smoothness = 2
    S = [2]
    indicator = Basis(all_ranks, [2], Xm, smoothness)
    eye = Matrix(I, n, n)
    # BasisMatrix
    B = BasisMatrix(indicator, Xm)
    v = ones(n)


    # Construct the "true" sort
    pa = path_sample(all_ranks, S)

    B_true = (Xm[:, 2] .>= Xm[pa, 2]') .* (Xm[:, 2].^smoothness .- (Xm[pa, 2].^smoothness)') ./ factorial(smoothness)    
    @test B_true * ones(n) ≈ B * ones(n)
    @test B * Matrix(I, B.ncol, B.ncol) == B_true[:, end:-1:1]

    
    # Test if the matrices contain the same columns
    matrix_cols = collect(eachcol(B_true))
    basis_cols = eachcol(B * Matrix(I, B.ncol, B.ncol))
    @test all([col in matrix_cols for col in basis_cols])

    # BasisMatrixTranspose
    Bt = transpose(B)

    @test Bt * ones(n) ≈ reverse(transpose(B_true) * ones(n))
    @test Bt * v ≈ reverse(transpose(B_true) * v)

    # BasisMatrixBlocks
    S = [[2], [2, 3]]
    indb = BasisBlocks(S, Xm, 1)
    Bb = BasisMatrixBlocks(indb, Xm)

    ## Not there's no real ground truth here since we sample a random path,
    ## plus we already tested the individual blocks, so we'll just test 
    # that the additional block components function as expected
    @test Bb.ncol == sum(Bb.blocks[i].F.ncol for i in 1:length(Bb.blocks))
    @test Bb.nrow == n
    @test all(sort(Bb * ones(Bb.ncol)) .< Bb.ncol)

    # BasisMatrixBlocksTranspose
    Bbt = transpose(Bb)
    @test Bbt.ncol == n
    @test Bbt.nrow == Bb.ncol
    @test all(sort(Bbt * ones(Bbt.ncol)) .< Bbt.nrow)

end

#@testset "Coordinate descent" begin

    # Set up inputs
    smoothness = 1
    ycs = (y .- mean(y)) ./ sqrt(var(y, corrected=false))
    S = collect(combinations([1,2,3]))[2:end]
    #S = [[1], [2], [1,2]]
    d = size(Xm, 2)
    indb = BasisBlocks(S, Xm, smoothness)
    B = BasisMatrixBlocks(indb, Xm)

    μ = colmeans(B)

    F = BasisMatrixBlocks(BasisBlocks(S, Xm, 0), Xm)
    findall(transpose(F) * ones(F.nrow) .> sqrt(n))


    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)

    σ2

    σ2[σ2 .< 0.0] .= 0.0
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0 

    # Test the scaling
    B2 = (B * Matrix(I, B.ncol, B.ncol))

    μ_true = vec(mean(B2, dims=1))


    @test μ_true ≈ μ

    σ2_true = vec(var(B2, corrected=false, dims=1))
    @test σ2_true ≈ σ2

    BT = (transpose(B) * Matrix(I, B.nrow, B.nrow))
    @test transpose(B2) ≈ BT

    # Run the algorithm
    λ_range = [0.1, 0.01, 0.001, 0.0001]
    @time path, β0 = coord_descent(B, ycs, μ, invσ, σ2, λ_range; outer_max_iters = 1000, inner_max_iters = 1000)
    preds = B * path .+ β0

    # Make sure we get close to a reasonable solution
    mse = [mean((preds[:, i] .- ycs).^2) for i in 1:size(path, 2)]
    @test all(mse .< 0.5)
    @test mse[2] < mse[1]
    @test mse[3] < mse[2]
    @test mse[3] < 0.1

    # How close are we to GLMNet?
    @time glmnet_fit = glmnet(B2, ycs, lambda = λ_range, intercept = true)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)

    glmnet_mse = [mean((GLMNet.predict(glmnet_fit, B2)[:, i] .- ycs).^2) for i in 1:length(λ_range)]

    abs_diff = abs.(glmnet_mse .- mse)
    @test all(abs_diff .< 0.01)
end

#@testset "Cross-validated model" begin
    # Make sure to center y to make comparison with GLMNet feasible
    #ycs = (y .- mean(y)) ./ sqrt(var(y, corrected=false))

    # Set up model parameters
    S = collect(combinations(1:40))[2:end]
    #S = [[1], [2], [2,3]]
    min_λ_ε = 0.001
    n_λ = 100
    smoothness = 0

    max_block_size = n
    @time model = fast_fit_cv_randomhal(S, Xm, y; max_block_size = max_block_size, smoothness = smoothness, K = 10, min_λ_ε = min_λ_ε, n_λ = n_λ) 
    
    preds = predict_randomhal(model, Xm)
    mse = mean((true_conmean .- preds).^2)
    @test mse < 0.1

    # How does this compare to glmnet?
    # Instantiate full basis
    indb = subsample(BasisBlocks(S, Xm, smoothness), max_block_size)
    B = BasisMatrixBlocks(indb, Xm)
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    
    # Set up grid so that glmnet is consistent with our method
    μ = colmeans(B)
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    σ2[σ2 .< 0.0] .= 0.0
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0
    corrs = ((transpose(B)*ycs) .- (μ .* sum(ycs))) .* invσ
    λ_max = maximum(abs.(corrs)) / n
    λ_min = min_λ_ε * λ_max    
    λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = n_λ)))
    
    @time glmnet_fit = glmnetcv(B2, y; lambda = λ_range)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)
    glmnet_mse = mean((true_conmean .- glmnet_preds).^2)

    @test abs(mse - glmnet_mse) < 0.01
    @test abs(model.best_λ - glmnet_fit.lambda[argmin(glmnet_fit.meanloss)]) .< 0.01
    
    #= using Plots
    Btest = BasisMatrixBlocks(indb, Xmtest)
    B2test = Btest * Matrix(I, B.ncol, B.ncol)
    preds_test = predict_randomhal(model, Xmtest)

    scatter(true_conmean_test, ytest)
    scatter!(true_conmean_test, preds_test)
    scatter!(true_conmean_test, GLMNet.predict(glmnet_fit, B2test))

    function pred(model, ind)
        β_alt = copy(model.β)
        β_alt[Not(ind)] .= 0.0
        β0 = model.β0
        (Btest * β_alt) .+ β0
    end

    scatter(Xmtest[:, 1], pred(model, 1:400))
    scatter(Xmtest[:, 2], pred(model, 401:800)) =#


    


    
end

#@testset "MLJ Interface" begin
    # Instantiate an MLJ model with mostly default parameters
    Random.seed!(123)

    model = RandomHALRegressor(smoothness = 1, max_block_size = n ÷ 4)

    mach = machine(model, X, y) |> MLJBase.fit!

    B = BasisMatrixBlocks(mach.fitresult.params.indblocks, Xm)
    B2 = B * Matrix(I, B.ncol, B.ncol)

    using MLJLinearModels
    basis = B2[:, mach.fitresult.params.β .!= 0.0]
    linmod = LinearRegressor()
    linmach = machine(linmod, Tables.table(basis), y) |> MLJBase.fit!

    linpreds = MLJBase.predict(linmach, basis)


    # Make sure our predictions work well
    preds = MLJBase.predict(mach, X)
    mse = mean((true_conmean .- preds).^2)
    @test mse < 0.01

    scatter(true_conmean, y)
    scatter!(true_conmean, preds)
    scatter!(true_conmean, linpreds)


    preds_test = MLJBase.predict(mach, Xtest)
    mse_test = mean((true_conmean_test .- preds_test).^2)
    @test mse_test < 0.01

    B_test = BasisMatrixBlocks(mach.fitresult.params.indblocks, Xmtest)
    B2_test = B_test * Matrix(I, B.ncol, B.ncol)
    basis_test = B2_test[:, mach.fitresult.params.β .!= 0.0]
    linpreds_test = MLJBase.predict(linmach, basis_test)



    scatter(true_conmean_test, ytest)
    scatter!(true_conmean_test, preds_test)
    scatter!(true_conmean_test, linpreds_test)

    indb = mach.fitresult.params.indblocks
    indb.blocks[1].indicators.path

end

@testset "Logistic regression coordinate descent" begin
    smoothness = 0
    #S = collect(combinations([1,2,3]))[2:end]
    S = [[1], [2], [1,2]]
    indb = BasisBlocks(S, Xm, smoothness)
    B = BasisMatrixBlocks(indb, Xm)
    B2 = (B * Matrix(I, B.ncol, B.ncol))

    μ = colmeans(B)
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    σ2[σ2 .< 0.0] .= 0.0
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0 

    # Test the weighted variance and mean for IRLS
    w = rand(n)
    true_w_mean =  vec(sum(B2 .* w, dims=1))
    @test true_w_mean ≈ colmeans(B, w)
    true_w_squares = vec(sum(w .* (B2.^2), dims=1))
    w_squares = squares(transpose(B), w)
    @test true_w_squares ≈ w_squares
    @test vec(sum(B2 .* 0.25, dims=1)) ≈ colmeans(B) .* (n * 0.25)
    @test vec(sum(0.25 .* (B2.^2), dims=1)) ≈ squares(transpose(B)) .* 0.25

    λ_range = [0.1, 0.05, 0.01, 0.001]
    @time path, β0 = coord_descent_binom(B, A, μ, invσ, σ2, λ_range)

    lin_preds = (B * path) .+ β0
    preds = 1 ./ (1 .+ exp.(-lin_preds))

    mse = [mean((preds[:, i] .- true_pr).^2) for i in 1:size(path, 2)]

    # How close are we to GLMNet?
    @time glmnet_fit = glmnet(B2, float.([.!(A) A]), Binomial(), lambda = λ_range)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2, outtype = :prob)
    glmnet_mse = [mean((GLMNet.predict(glmnet_fit, B2, outtype = :prob)[:, i] .- true_pr).^2) for i in 1:length(glmnet_fit.lambda)]

    abs_diff = abs.(glmnet_mse .- mse)
    @test all(abs_diff .< 0.01)

    #scatter(Xm[:, 1], true_pr)
    #scatter!(Xm[:, 1], preds[:, 3])
    #scatter!(Xm[:, 1], glmnet_preds[:, 3])

    #scatter(true_pr, preds[:, 2])
    #scatter!(true_pr, glmnet_preds[:, 2])

end

#@testset "Cross-validated logistic regression" begin#
    # Set up model parameters
    S = collect(combinations([1,2]))[2:end]
    Random.seed!(1234)

    #S = [[1], [2], [1,2]]
    min_λ_ε = 0.001
    n_λ = 100
    smoothness = 1
    max_block_size = n
    @time model = fast_fit_cv_randomhal(S, Xma, Float64.(A); family = Binomial(), max_block_size = max_block_size, smoothness = smoothness, K = 10, min_λ_ε = min_λ_ε, n_λ = n_λ) 
    
    preds = predict_randomhal(model, Xma)
    mse = mean((true_pr .- preds).^2)
    @test mse < 0.01

    scatter(true_pr, preds)


    # How does this compare to glmnet?
    # Instantiate full basis
    indb = model.indblocks
    B = BasisMatrixBlocks(indb, Xma)
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    
    # Set up grid so that glmnet is consistent with our method
    μ = colmeans(B)
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    σ2[σ2 .< 0.0] .= 0.0
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0
    corrs = ((transpose(B)*A) .- (μ .* sum(A))) .* invσ
    λ_max = maximum(abs.(corrs)) / n
    λ_min = min_λ_ε * λ_max    
    λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = n_λ)))
    
    @time glmnet_fit = glmnetcv(B2, float.([.!(A) A]), Binomial(); lambda = λ_range)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2, outtype = :prob)
    glmnet_mse = mean((true_pr .- glmnet_preds).^2)

    @test abs(mse - glmnet_mse) < 0.1

    scatter!(true_pr, glmnet_preds, color = "black")

    preds_test = predict_randomhal(model, Xmatest)

    scatter(true_pr_test, preds_test)


    Btest = BasisMatrixBlocks(indb, Xmatest)
    B2test = Btest * Matrix(I, Btest.ncol, Btest.ncol)
    glmnet_preds_test = GLMNet.predict(glmnet_fit, B2test, outtype = :prob)

    mse_test = mean((true_pr_test .- preds_test).^2)
    @test mse_test < 0.01

    pred_path = 1 ./ (1 .+ exp.(-(Btest * model.β_path .+ model.β0_path)))

    anim = @animate for i in 1:100
        scatter(true_pr_test, pred_path[:, i])
    end

    gif(anim, "anim_fps15.gif", fps = 15)


    #scatter!(true_pr_test, glmnet_preds_test)

end

#@testset "MLJ Interface Logistic Regression" begin
    # Instantiate an MLJ model with mostly default parameters
    Random.seed!(1234)

    model = RandomHALClassifier(smoothness = 1, max_block_size = n)
    mach = machine(model, Xa, Float64.(A)) |> MLJBase.fit!

    # Make sure our predictions work well
    preds = MLJBase.predict(mach, Xa)
    mse = mean((true_pr .- preds).^2)
    @test mse < 0.01

    preds_test = MLJBase.predict(mach, Xatest)
    mse_test = mean((true_pr_test .- preds_test).^2)
    @test mse < 0.01

    scatter(true_pr, preds)
    scatter(true_pr_test, preds_test)

end