using Test
using RandomHAL
using Random
using CausalTables
using Distributions
using MLJ
using StatsBase
using LinearAlgebra
import LogExpFunctions: logistic

using Profile

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

n = 900
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

    @test halmse < 0.05

    # Random HAL
    n_samples = Int(round(n * log(n)))
    model3 = RandomHALRegressor(0, n, 5, n_samples, (guaranteed_sections = [[4]], interaction_order_weights = Weights([8, 4, 2, 0])))
    
    @time rhal = machine(model3, X, y) |> fit!

    rhalpreds = MLJ.predict(rhal, Xtest)
    rhalmse = mean((rhalpreds .- true_mean).^2)

    # RMSE are bounded
    @test rhalmse < 0.05
    
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

    @test halbinmse < 0.05
    @test rhalbinmse < 0.05

end

@testset "Riesz Models" begin

    X_shift = intervene(responseparents(ct), treat_all)
    truth = cfmean(scm, treat_all)


    rmodel1 = HALRiesz()
    @time rmach1 = machine(rmodel1, X, X_shift) |> fit!
    w1 = MLJ.predict(rmach1) 
    ipw1 = mean(w1 .* y)

    @test abs(ipw1 - truth.μ) < 0.05

    rmodel2 = RandomHALRiesz()
    @time rmach2 = machine(rmodel2, X, X_shift) |> fit!
    w2 = MLJ.predict(rmach2)
    ipw2 = mean(w2 .* y)

    @test abs(ipw2 - truth.μ) < 0.05

end



# TODO: FIX LOSSES SO THEY ALSO ARE FAST AND ITERATE THROUGH COLUMNS INSTEAD OF ROWS
riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β, β0) = mean((X * β .+ β0).^2, dims = 1) .- mean(2 .* ((X_shift * β) .+ β0), dims = 1)

# Losses for checking individual updates
reg_riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector, λ) = mean((X * β).^2) - 2 * sum(mean_shift .* β) + 2*λ * sum(abs.(β))

function riesz_loss(X::AbstractMatrix, mean_shift::Vector{Float64}, β::Vector{Float64})
    pred = X * β
    return dot(pred, pred)/size(X, 1) - 2 * dot(mean_shift, β)
end

@time riesz_loss(X, mean_shift, β1)
@time riesz_loss_active(X, mean_shift, β1, active_set)


soft_threshold(z::Float64, λ::Float64) = sign(z) * max(0, abs(z) - λ)

pct_change(next_loss::Float64, prev_loss::Float64) = abs(next_loss - prev_loss) / prev_loss

function cycle_coord!(β::Vector{Float64}, cols, mean_shift::Vector{Float64}, λ::Float64)
    @inbounds for (i, col) in enumerate(cols)
        β[i] = soft_threshold(mean_shift[i] - dot(col, β), λ)
    end
end

function cycle_coord_inactive!(β::Vector{Float64}, cols, mean_shift::Vector{Float64}, λ::Float64, inactive_set)
    @inbounds for i in inactive_set
        β[i] = soft_threshold(mean_shift[i] - dot(cols[i], β), λ)
    end
end

function cycle_coord_active!(β::Vector{Float64}, cols, mean_shift::Vector{Float64}, λ::Float64, active_set)
    @inbounds for i in active_set
        penalty = mean_shift[i]
        cur_col = cols[i]
        @simd for j in active_set
            penalty -= cur_col[j] * β[j]
        end
        β[i] = soft_threshold(penalty, λ)
    end    
end

β1 = zeros(size(X, 2))
β2 = zeros(size(X, 2))
β3 = zeros(size(X, 2))

cols = eachcol(ZZbyn)
active_set = sample(1:d, 500)

@time cycle_coord!(β1, cols, mean_shift, 0.01)
@time cycle_coord_active!(β2, cols, mean_shift, 0.01, active_set)
@time cycle_coord_inactive!(β3, cols, mean_shift, 0.01, active_set)
@time cycle_coord_active_arr!(β3, ZZbyn, mean_shift, 0.01, active_set)




function coord_descent(X::AbstractMatrix, X_shift::AbstractMatrix; λ = nothing, min_λ_ε = 0.001, λ_grid_length = 20, outer_max_iters = 20, inner_max_iters = 50, tol = 0.0)
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
    #λ = nothing
    #λ_grid_length = 100
    #min_λ_ε = 0.001
    if isnothing(λ)
        λ_max = maximum(abs.(mean_shift))# / α
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = reverse(λ)
    end

    # Set up storage for coefficients
    λ_length = length(λ_range)
    β = Matrix(undef, d, λ_length)
    β_next = zeros(d)

    # We loop through λ in the outer loop to take advantage of warm starts
    for (λ_index, λ) in enumerate(λ_range)

        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 
        active_set = []
        inactive_set = 1:d
        norm_next = tol .+ 1.0
        outer_iteration = 1
        # Run an initial update
        cycle_coord!(β_next, cols, mean_shift, λ)
        prev_riesz_loss = riesz_loss(Z, mean_shift, β_next)

        while (outer_iteration < outer_max_iters)
            # Initial full set iteration. Iterate through each coordinate cyclically
            cycle_coord_inactive!(β_next, cols, mean_shift, λ, inactive_set)

            # Update the active set
            next_active_set = findall(β_next .!= 0)

            # If the active set has not changed, then we're done. Otherwise, keep going
            active_set == next_active_set && break
            active_set = next_active_set
            inactive_set = findall(β_next .== 0)

            # Update the norm to track convergence
            next_riesz_loss = riesz_loss(Z, mean_shift, β_next)
            norm_next = pct_change(next_riesz_loss, prev_riesz_loss)
            prev_riesz_loss = next_riesz_loss

            # Update active set until convergence
            inner_iteration = 1
            while (inner_iteration < inner_max_iters) && (norm_next > tol)
                cycle_coord_active!(β_next, cols, mean_shift, λ, active_set)

                # Update the norm to track convergence
                next_riesz_loss = riesz_loss(Z, mean_shift, β_next)
                norm_next = pct_change(next_riesz_loss, prev_riesz_loss)
                prev_riesz_loss = next_riesz_loss
            end
            outer_iteration += 1
        end
        β[:, λ_index] = β_next
    end

    # Reconstruct coefficients to be on the original scale
    β = β .* transpose(invsds)

    # Finally, add the intercept.
    # This intercept, when scaled by y, accounts for the fact 
    # that when the β are rescaled, they no longer sum to 1. 
    β0 = 1 .- vec(mean(X, dims = 1) * β)

    return β, β0
end

# NEED TO OPTIMIZE THIS FUNCTION STILL
function cross_coord_descent(X, X_shift; nfolds = 5, λ = nothing, min_λ_ε = 0.01, λ_grid_length = 100, outer_max_iters = 20, inner_max_iters = 20, tol = 0.01, only_refit_best = true)
    # Get components to standardize data to select initial lambda grid
    n, d = size(X)
    means = mean(X, dims = 1)
    invsds = 1 ./ std(X, dims = 1)

    # Set up safeguard for variables with 0 variance
    invsds[isinf.(invsds)] .= 0

    # Standardize the data
    Z_shift = (X_shift .- means) .* invsds
    mean_shift = mean(Z_shift, dims = 1)

    # Compute search grid of lambda
    if isnothing(λ)
        λ_max = maximum(abs.(mean_shift))
        λ_min = min_λ_ε * λ_max    
        λ_range = exp.(range(log(λ_min), log(λ_max), length = λ_grid_length))
    else
        λ_range = λ
    end

    ### Run coordinate descent cross-validated ###
    # Set up initial variables and storage
    train, test = train_test_folds(1:n, nfolds)
    βs = Vector{AbstractMatrix}(undef, nfolds)
    β0s = Vector{AbstractVector}(undef, nfolds)
    loss = Vector{AbstractVector}(undef, nfolds)


    # Cross-validate coordinate descent
    for fold in 1:nfolds
        X_train = view(X, train[fold], :)
        X_shift_train = view(X_shift, train[fold], :)
        βs[fold], β0s[fold] = coord_descent(X_train, X_shift_train; λ = λ_range, min_λ_ε = min_λ_ε, λ_grid_length = λ_grid_length, outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol)
        
        X_test = view(X, test[fold], :)
        X_shift_test = view(X_shift, test[fold], :)
        loss[fold] = vec(riesz_loss(X_test, X_shift_test, βs[fold], reshape(β0s[fold], 1, length(λ_range))))
    end

    mean_loss = mean(reduce(hcat, loss), dims = 2)
    best_λ_index = argmin(vec(mean_loss))

    total_λ_range = only_refit_best ? λ_range[best_λ_index] : λ_range
    β, β0 = coord_descent(X, X_shift; λ = [total_λ_range], outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol)

    return β, β0, total_λ_range
end

#Profile.clear()
@time β, β0 = coord_descent(X, X_shift; λ_grid_length = 100)
#Profile.print(format=:flat)

@time βc, β0c, λc = cross_coord_descent(X, X_shift; λ_grid_length = 20)

    mean((X * βc .+ β0c) .* y)

    ipws = mean((X * β .+ transpose(β0)).*y, dims = 1)
    truth = cfmean(scm, treat_all)

    λ_range[argmin((vec(ipws) .- truth.μ).^2)]
    λc

    λ = nothing
    α = 1.0
    min_λ_ε = 0.01
    λ_grid_length = 20
    outer_max_iters = 20
    inner_max_iters = 20
    tol = 0.0
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

    using Plots
    plot(log.(λ_range), vec(ipws))
    xflip!(true)
    hline!([truth.μ])

    riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β::AbstractMatrix, β0::AbstractMatrix) = mean((X * β .+ β0).^2, dims = 1) .- mean(2 .* ((X_shift * β) .+ β0), dims = 1)
    losses = vec(riesz_loss(X, X_shift, β, reshape(β0, 1, length(λ_range))))
    plot(log.(λ_range), log.(losses .- minimum(losses) .+ 0.001))
    xflip!(true)

    plot(vec(ipws), log.(losses .- minimum(losses) .+ 0.001))
    xflip!(true)


mean((X_test * βs[fold] .+ β0s[fold]).^2, dims = 1) .- mean(2 .* ((X_shift_test * βs[fold]) .+ β0s[fold]), dims = 1)