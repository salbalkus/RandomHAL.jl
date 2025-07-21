using Test
using RandomHAL
using Random
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
        A ~ (@. Bernoulli(logistic(-(X2 + X3 + 6 * X4 .+ 0.1)./8))),
        Y ~ (@. Normal(A + X2 + X3 + X4 .+ 1, 0.1))
    )
scm = StructuralCausalModel(dgp, :A, :Y)

n = 200
ct = rand(scm, n)
X = Tables.Columns(responseparents(ct))
y = vec(responsematrix(ct))

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

#@testset "Model Fitting" begin

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

riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector) = mean((X * β).^2) - 2 * dot(mean_shift, β)

# Cross-fitting functions
function evenly_spaced_grid(n, nfolds)
    leftover = n % nfolds
    output = fill(n ÷ nfolds, nfolds)
    output[1:leftover] .+= 1
    return cumsum(output)
end

function train_test_folds(input_indices, nfolds)
    n = length(input_indices)
    ind = shuffle(input_indices)
    end_points  = evenly_spaced_grid(n, nfolds)
    start_points = vcat([1], end_points[1:(nfolds-1)] .+ 1)
    test_folds = [ind[start_points[i]:end_points[i]] for i in 1:nfolds]
    train_folds = [reduce(vcat, test_folds[Not(i)]) for i in 1:nfolds]
    return train_folds, test_folds
end

model = HALRiesz()
X = Tables.Columns(responseparents(ct))
X_shift = Tables.Columns(intervene(responseparents(ct), treat_all))

basis, all_sections, term_lengths = ha_basis_matrix(X, model.smoothness)
basis_shift, all_sections_shift, term_lengths_shift = ha_basis_matrix(X_shift, X, model.smoothness)

X = basis
X_shift = basis_shift

soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)

function update_coef(β::AbstractVector, col::AbstractVector, mean_shift::Real, λ::Float64, α::Float64)
    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    return soft_threshold(mean_shift - dot(col, β), α * λ) / (1 + (1 - α) * λ)
end

function cycle_coord(β_next, cols, mean_shift, λ, α)
    for (i, c) in enumerate(cols)
        β_next[i] = update_coef(β_next, c, mean_shift[i], λ, α)
    end
end

function coord_descent(X, X_shift; λ = nothing, α = 1.0, min_λ_ε = 0.01, λ_grid_length = 100, outer_max_iters = 20, inner_max_iters = 20, tol = 0.01)
    # Initialize variables
    n, d = size(X)

    # Get components to standardize data
    means = mean(X, dims = 1)
    invsds = 1 ./ std(X, dims = 1)

    # Set up safeguard for variables with 0 variance
    invsds[isinf.(invsds)] .= 0

    # Standardize the data
    Z = (X .- means) .* invsds
    Z_shift = (X_shift .- means) .* invsds
    mean_shift = vec(mean(Z_shift, dims = 1))

    ZZbyn = transpose(Z) * Z ./ n
    ZZbyn[diagind(ZZbyn)] .= 0
    cols = eachcol(ZZbyn)

    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    λ = nothing
    if isnothing(λ)
        λ_max = maximum(abs.(mean_shift)) / α
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = reverse(λ)
    end

    # Set up storage for coefficients
    λ_length = length(λ_range)
    β = fill(zeros(d), λ_length)
    β_next = zeros(d)

    # We loop through λ in the outer loop to take advantage of warm starts
    for (λ_index, λ) in enumerate(λ_range)
        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 
        active_set = []
        norm_next = tol .+ 1.0
        outer_iteration = 1
        # Run an initial update
        cycle_coord(β_next, cols, mean_shift, λ, α)
        prev_riesz_loss = riesz_loss(Z, mean_shift, β[λ_index])

        while (outer_iteration < outer_max_iters)
            # Initial full set iteration. Iterate through each coordinate cyclically
            cycle_coord(β_next, cols, mean_shift, λ, α)

            # Update the active set
            next_active_set = findall(β_next .!= 0)
            
            # Update the norm to track convergence
            next_riesz_loss = riesz_loss(Z, mean_shift, β_next)
            norm_next = abs((next_riesz_loss - prev_riesz_loss) / prev_riesz_loss)
            prev_riesz_loss = next_riesz_loss

            # If the active set has not changed, then we're done. Otherwise, keep going
            active_set == next_active_set && break
            active_set = next_active_set

            # Update active set until convergence
            inner_iteration = 1
            while (inner_iteration < inner_max_iters) && (norm_next > tol)

                # Repeat initial loop twice
                cycle_coord(β_next, cols, mean_shift, λ, α)
                
                # Update the norm to track convergence
                next_riesz_loss = riesz_loss(Z, mean_shift, β_next)
                norm_next = abs((next_riesz_loss - prev_riesz_loss) / prev_riesz_loss)
                prev_riesz_loss = next_riesz_loss

                β[λ_index] = copy(β_next)
                inner_iteration += 1
            end
            outer_iteration += 1
        end
    end

    # Reconstruct coefficients to be on the original scale
    β_orig = reduce(hcat, β)
    β_orig = β_orig .* transpose(invsds)

    # Finally, add the intercept.
    # This intercept, when scaled by y, accounts for the fact 
    # that when the β are rescaled, they no longer sum to 1. 
    β0 = 1 .- mean(X * β_orig, dims = 1)

    return β_orig, β0
end

@time coord_descent(X, X_shift)

β_orig, β0 = coord_descent(X, X_shift)

    ipws = mean((X * β_orig .+ β0).*y, dims = 1)
    truth = cfmean(scm, treat_all)

    using Plots
    plot(log.(λ_range), vec(ipws))
    xflip!(true)
    hline!([truth.μ])

    riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β::AbstractMatrix, β0::AbstractMatrix) = mean((X * β .+ β0).^2, dims = 1) .- mean(2 .* ((X_shift * β) .+ β0), dims = 1)
    losses = vec(riesz_loss(X, X_shift, reduce(hcat, β), β0))
    plot(log.(λ_range), log.(losses .- minimum(losses) .+ 0.001))
    xflip!(true)

    plot(vec(ipws), log.(losses .- minimum(losses) .+ 0.001))
    xflip!(true)


