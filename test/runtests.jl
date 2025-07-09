using RandomHAL
using Random
using Test
using CausalTables
using Distributions
using MLJ
using StatsBase
using LinearAlgebra
import LogExpFunctions: logistic

using Tables
import Combinatorics: powerset

Random.seed!(1234)
dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.1))
    )
scm = StructuralCausalModel(dgp, :A, :Y)

dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic(-(X2 + X3 + 6 * X4)./8))),
        Y ~ (@. Normal(A + X2 + X3 + X4, 0.1))
    )
scm = StructuralCausalModel(dgp, :A, :Y)

n = 1000
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))

soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)

function update_coef(i, β, X, mean_shift, λ, α, n)
    # Get predictions without the contribution of the ith feature
    held_out_preds = X[:, Not(i)] * β[Not(i)]

    # Compute the squared penalty of the Riesz representer
    square_penalty = dot(X[:, i], held_out_preds) ./ n

    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    β_next = soft_threshold(square_penalty .+ mean_shift[i], α * λ) / (1 + (1-α) * λ)
    return β_next
end

function coord_descent(X, Χ_shift, λ, α; max_iters = 1000, tol = 1e-4)
    # Initialize variables
    n, d = size(X)
    β = zeros(d)
    β_next = zeros(d)
    iteration = 0
    mean_shift = vec(mean(Χ_shift, dims = 1) .- mean(Χ, dims = 1))

    # Descend until convergence or max iterations
    while (iteration < max_iters) || (norm(β .- β_next) > tol)
        println("Iteration: ", iteration)
        # Iterate through each coordinate
        for i in 1:d
            β_next[i] = update_coef(i, β_next, X, mean_shift, λ, α, n)
            println(β_next)
        end

        # Update variables for this cycle
        β = copy(β_next)
        iteration += 1
    end

    return β
end

X = Tables.matrix(X)
X_shift = copy(X)
X_shift[:, 4] .= 1.0

t = fit(ZScoreTransform, X; dims=1)
Z = StatsBase.transform(t, X)
Z_shift = StatsBase.transform(t, X_shift)
Z = hcat(ones(n, 1), Z)
Z_shift = hcat(ones(n, 1), Z_shift)

n, d = size(Z)
β = zeros(d)
β_next = zeros(d)
norm_next = Inf
iteration = 0
mean_shift = vec(mean(Z_shift, dims = 1))

# Descend until convergence or max iterations
max_iters = 100
λ = 0.0
α = 1.0
tol = 1e-4

function update_coef(i, β, X, mean_shift, λ, α, n)
    # Get predictions without the contribution of the ith feature
    held_out_preds = X[:, Not(i)] * β[Not(i)]

    # Compute the squared penalty of the Riesz representer
    square_penalty = dot(X[:, i], held_out_preds) ./ n

    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    β_next = soft_threshold(mean_shift[i] - square_penalty, α * λ) / (1 + (1-α) * λ)
    return β_next
end

while (iteration < max_iters) && (norm_next > tol)
    println("Iteration: ", iteration)
    # Iterate through each coordinate
    for i in 1:d
        β_next[i] = update_coef(i, β_next, Z, mean_shift, λ, α, n)
        println(β_next)
    end

    # Update variables for this cycle
    iteration += 1
    norm_next = norm(β .- β_next)
    println("Norm: ", norm_next)
    β = copy(β_next)
end

w = (Z * β)
mean(w .* y) # IPW should match the true


# The Riesz weights should match the linear regression weights
reg_w = vec(sum(Z_shift * inv(Z'Z) * Z', dims = 1))
scatter(w, reg_w)
scatter(w[ct.data.A .== 0], reg_w[ct.data.A .== 0])
scatter(w[ct.data.A .== 1], reg_w[ct.data.A .== 1])

mean(conmean(scm, intervene(ct, treat_all), :Y))

scatter(w[ct.data.A .== 0], 1 ./ propensity(scm, ct, :A)[ct.data.A .== 0])

mean(Z * β)







function test_basis(smoothness)
    basis, all_sections, term_lengths = ha_basis_matrix(X, smoothness)#; basis_type = basis_type)

    @test size(basis)[1] == n
    #@test size(basis)[2] == n * (2^length(X) - 1) - (n - 1)
    @test basis[:, (n * 7) + 1] == ct.data.A
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
    model = HALRegressor(0)
    @time hal = machine(model, X, y) |> fit!

    halpreds = MLJ.predict(hal, Xtest)
    halmse = mean((halpreds .- true_mean).^2)

    @test halmse < 0.03

    # Random HAL
    n_samples = Int(round(n * log(n)))
    model3 = RandomHALRegressor(0, n, 5, n_samples, (guaranteed_sections = [[4]], interaction_order_weights = Weights([8, 4, 2, 0])))
    
    @time rhal = machine(model3, X, y) |> fit!

    rhalpreds = MLJ.predict(rhal, Xtest)
    rhalmse = mean((rhalpreds .- true_mean).^2)

    # RMSE are bounded
    @test rhalmse < 0.03
    
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

    @test halbinmse < 0.04
    @test rhalbinmse < 0.03

end
