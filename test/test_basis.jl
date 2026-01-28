using Test
using Tables
using CausalTables
using Distributions
import LogExpFunctions: logistic
using StatsBase
using DecisionTree
using Plots
using LinearAlgebra
using Random
using GLMNet

Random.seed!(1234)

dgp = @dgp(
        X1 ~ Bernoulli(0.5),
        X2 ~ Beta(1, 1),
        X3 ~ Beta(1, 1),
        X4 ~ Beta(1, 1),
        X5 ~ Normal.(X2, 0.01),
        X6 ~ Normal.(X3 .* (1 .- (2 .*X1)), 0.001),
        X7 ~ Normal.(sin.(2*pi*X2), 0.0001),
        X8 ~ Normal.(1 .- cos.(2*pi*X2), 0.0001),
        X9 ~ Normal.((X3 .- 0.5) .* ((X3 .> 0.5) - (X3 .< 0.5)), 0.0),

        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        #Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.01))
        Y ~ (@. Normal(sin.(2*pi * X2), 0.1))
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

@testset "Coordinate descent" begin
    # Set up inputs
    ycs = (y .- mean(y)) ./ sqrt(var(y, corrected=false))
    S = [[2]]
    indb = NestedIndicatorBlocks(S, Xm)
    B = NestedMatrixBlocks(indb, Xm)
    μ = (transpose(B) * ones(B.nrow)) ./ n
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0 

    # Test the scaling
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    B2c = (B2 .- reshape(μ, 1, B.nrow)) .* reshape(invσ, 1, B.nrow)
    @test vec(mean(B2c .* B2c, dims=1)) ≈ vcat(ones(B.ncol-1), [0])

    # Run the algorithm
    λ_range = [0.1, 0.01, 0.001, 0.0001]
    path = coord_descent(B, ycs, μ, σ2, λ_range; outer_max_iters = 1000, inner_max_iters = 1000, tol = 10e-7, warm_β = zeros(100))
    # Make sure we get close to a reasonable solution
    
    path_scaled = path .* invσ
    preds = B * path_scaled .- (reshape(μ, 1, n) * path_scaled)# .+ mean(y)
    mse = [mean((preds[:, i] .- ycs).^2) for i in 1:size(path, 2)]
    @test all(mse .< 0.5)
    @test mse[2] < mse[1]
    @test mse[3] < mse[2]
    @test mse[3] < 0.05

    # How close are we to GLMNet?
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    B2 = B2[:,1:(size(B2,2)-1)]
    glmnet_fit = glmnet(B2, ycs, lambda = λ_range, intercept = false)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)

    glmnet_mse = [mean((GLMNet.predict(glmnet_fit, B2)[:, i] .- ycs).^2) for i in 1:length(λ_range)]

    abs_diff = abs.(glmnet_mse .- mse)
    @test all(abs_diff .< 0.01)


    #l = 3
    #scatter(preds[:, l], glmnet_preds[:, l])
    #scatter(glmnet_preds[:, l], ycs)
    #scatter!(preds[:, l], ycs)

    #scatter(Xm[:, 2], ycs)
    #scatter!(Xm[:, 2], preds[:, l])
    #scatter!(Xm[:, 2], glmnet_preds[:, l])
end

@testset "Cross-validated model" begin
    S = [[2]]
    @time model = fast_fit_cv_randomhal(S, Xm, y)
    
    preds = predict_randomhal(model, Xm)
    mse = mean((y .- preds).^2)
    @test mse < 0.01

    # How does this compare to glmnet?
    indb = NestedIndicatorBlocks(S, Xm)
    B = NestedMatrixBlocks(indb, Xm)
    B2 = (B * Matrix(I, B.ncol, B.ncol))
    glmnet_fit = glmnetcv(B2, y)
    glmnet_preds = GLMNet.predict(glmnet_fit, B2)
    glmnet_mse = mean((y .- glmnet_preds).^2)

    @test abs(glmnet_mse - mse) < 0.001
    
    # Slight difference between us and glmnet...
    # But this may be due to the randomness of the CV procedure
    scatter(preds, glmnet_preds)
    scatter(glmnet_preds, y)
    scatter!(preds, y)

    scatter(Xm[:, 2], y)
    scatter!(Xm[:, 2], preds)
    scatter!(Xm[:, 2], glmnet_preds)
end
