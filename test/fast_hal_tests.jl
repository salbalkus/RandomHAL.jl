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


Random.seed!(1234)

dgp = @dgp(
        #X1 ~ Bernoulli(0.5),
        X2 ~ Beta(1, 1),
        X3 ~ Beta(1, 1),
        X4 ~ Beta(1, 1),
        #X5 ~ Normal.(X2, 0.01),
        #X6 ~ Normal.(X3 .* (1 .- (2 .*X1)), 0.001),
        #X7 ~ Normal.(sin.(2*pi*X2), 0.0001),
        #X8 ~ Normal.(1 .- cos.(2*pi*X2), 0.0001),
        #X9 ~ Normal.((X3 .- 0.5) .* ((X3 .> 0.5) - (X3 .< 0.5)), 0.0),

        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        #Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.01))
        Y ~ (@. Normal(sin.(2*pi * X2) + sin(2*pi*X3) + sin(2*pi*X4), 0.1))
    )
scm = StructuralCausalModel(dgp, :A, :Y)
n = 100
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
Xm = Tables.matrix(X)
y = vec(responsematrix(ct))

# Test NestedMatrix functionality
@testset "NestedMatrix" begin
    all_ranks = reduce(hcat, map(competerank, eachcol(Xm)))
    indicator = NestedIndicators(all_ranks, [2], Xm)
    eye = Matrix(I, n, n)
    # NestedMatrix
    B = NestedMatrix(indicator, Xm)

    # Construct the "true" sort
    perm = reverse(sortperm(Xm[:, 2]))
    B_true = Xm[:, 2] .>= Xm[perm, 2]'

    @test B_true * ones(n) == B * ones(n)
    
    v = randn(n)
    @test B * v ≈ B_true * v
    @test B * eye == B_true

    # NestedMatrixTranspose
    Bt = transpose(B)
    @test Bt * ones(n) ≈ transpose(B_true) * ones(n)
    @test Bt * v ≈ transpose(B_true) * v

    # NestedMatrixBlocks
    S = [[2], [2, 3]]
    indb = NestedIndicatorBlocks(S, Xm)
    Bb = NestedMatrixBlocks(indb, Xm)

    ## Not there's no real ground truth here since we sample a random path,
    ## plus we already tested the individual blocks, so we'll just test 
    # that the additional block components function as expected
    @test Bb.ncol == sum(Bb.blocks[i].ncol for i in 1:length(Bb.blocks))
    @test Bb.nrow == n
    @test all(sort(Bb * ones(Bb.ncol)) .< Bb.ncol)

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
    indicator = Basis(all_ranks, [2], Xm, smoothness)
    eye = Matrix(I, n, n)
    # BasisMatrix
    B = BasisMatrix(indicator, Xm)
    v = ones(n)

    # Construct the "true" sort
    perm = reverse(sortperm(Xm[:, 2]))
    B_true = (Xm[:, 2] .>= Xm[perm, 2]') .* (Xm[:, 2].^smoothness .- (Xm[perm, 2].^smoothness)') ./ factorial(smoothness)    
    @test B_true * ones(n) ≈ B * ones(n)
    
    v = randn(n)
    @test B * v ≈ B_true * v
    @test B * eye == B_true

    # BasisMatrixTranspose
    Bt = transpose(B)
    @test Bt * ones(n) ≈ transpose(B_true) * ones(n)
    @test Bt * v ≈ transpose(B_true) * v

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

    # NestedMatrixBlocksTranspose
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
    #S = [[1]]
    indb = BasisBlocks(S, Xm, smoothness)
    B = BasisMatrixBlocks(indb, Xm)
    μ = colmeans(B)
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
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
    λ_range = [0.1, 0.01, 0.001, 0.000001]
    # WORKS FOR all 0th-order and 1D 1st-order smoothness, but nothing higher
    # ANOTHER PROBLEM: Higher-order smoothness requires increasingly lower and lower tolerance to match the glmnet
    # Not sure why that is; maybe because we're using a "percentage change" metric for convergence?
    path = coord_descent(B, ycs, μ_true, σ2_true, λ_range; outer_max_iters = 1000, inner_max_iters = 1000, tol = 10e-14)
    # Make sure we get close to a reasonable solution
    
    path_scaled = path .* invσ
    preds = B * path_scaled .- (reshape(μ, 1, B.ncol) * path_scaled)# .+ mean(y)

    mse = [mean((preds[:, i] .- ycs).^2) for i in 1:size(path, 2)]
    @test all(mse .< 0.5)
    @test mse[2] < mse[1]
    @test mse[3] < mse[2]
    @test mse[3] < 0.05

    # How close are we to GLMNet?
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    glmnet_fit = glmnet(B2, ycs, lambda = λ_range, intercept = true)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)

    glmnet_mse = [mean((GLMNet.predict(glmnet_fit, B2)[:, i] .- ycs).^2) for i in 1:length(λ_range)]

    abs_diff = abs.(glmnet_mse .- mse)
    @test all(abs_diff .< 0.01)

    scatter(Xm[:, 1], ycs)
    scatter!(Xm[:, 1], preds[:, 3])
    scatter!(Xm[:, 1], glmnet_preds[:, 3])

end


@testset "Cross-validated model" begin
    S = collect(combinations([2,3,4]))[2:end]
    min_λ_ε = 0.01
    λ_grid_length = 100

    @time model = fast_fit_cv_randomhal(S, Xm, y; K = 5, min_λ_ε = min_λ_ε, λ_grid_length = λ_grid_length) 
    
    preds = predict_randomhal(model, Xm)
    mse = mean((y .- preds).^2)
    @test mse < 0.01

    # How does this compare to glmnet?
    # Instantiate full basis
    indb = NestedIndicatorBlocks(S, Xm)
    B = NestedMatrixBlocks(indb, Xm)
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    
    # Set up grid so that glmnet is consistent with our method
    λ_max = maximum(abs.(transpose(B)*y_cs)) / n
    λ_min = min_λ_ε * λ_max    
    λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    
    glmnet_fit = glmnetcv(B2, y; lambda = λ_range)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)
    glmnet_mse = mean((y .- glmnet_preds).^2)

    @test abs(mse - glmnet_mse) < 0.01
end

@testset "MLJ Interface" begin
    model = RandomHALRegressor()


    mach = machine(model, X, y) |> MLJBase.fit!

    preds = MLJBase.predict(mach, X)
    mse = mean((y .- preds).^2)
    @test mse < 0.01
end
