riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β::AbstractMatrix, β0::AbstractMatrix) = mean((X * β .+ β0).^2, dims = 1) .- mean(2 .* ((X_shift * β) .+ β0), dims = 1)

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

soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)

# TODO: Speed this up by precomputing the entire covariance matrix Z'Z
# so that covariance betweeen Z values is not re-computed multiple times
# Perform a single update of one coordinate in the descent algorithm
function update_coef(i, β, Z, mean_shift, λ_range, α, n)
    # Get predictions without the contribution of the ith feature
    held_out_preds = Z[:, Not(i)] * β[Not(i), :]

    # Compute the squared penalty of the Riesz representer
    square_penalty = held_out_preds' * Z[:, i] ./ n

    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    β_next = soft_threshold.(mean_shift[i] .- square_penalty, α .* λ_range) ./ (1 .+ (1 - α) .* λ_range)
    return β_next
end

# TODO: Make this algorithm faster using active set
# Should only need to iterate on nonzero coefficients somehow
# Fit the minimum-distance Lasso or ElasticNet for an arbitrary basis X using coordinate descent
# Xβ is then the Riesz representer for X_shift

function coord_descent(X, X_shift; λ = nothing, α = 1.0, min_λ_ε = 0.001, λ_grid_length = 100, max_iters = 1000, tol = 0.0)
    # Initialize variables
    n, d = size(X)
    iteration = 1
    norm_next = tol .+ 1.0

    # Get components to standardize data
    means = mean(X, dims = 1)
    sds = std(X, dims = 1)

    # Standardize the data
    Z = (X .- means) ./ sds
    Z_shift = (X_shift .- means) ./ sds
    mean_shift = mean(Z_shift, dims = 1)

    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    if isnothing(λ)
        λ_max = maximum(abs.(mean_shift)) / α
        λ_min = min_λ_ε * λ_max    
        λ_range = exp.(range(log(λ_min), log(λ_max), length = λ_grid_length))
    else
        λ_range = λ
    end

    # Set up storage for coefficients
    λ_length = length(λ_range)
    β = zeros(d, λ_length)
    β_next = zeros(d, λ_length)

    # Descend active set until convergence or max iterations
    while (iteration < max_iters) && any(norm_next .> tol)
        # Iterate through each coordinate cyclically
        for i in 1:(d)
            β_next[i, :] = update_coef(i, β_next, Z, mean_shift, λ_range, α, n)
        end

        # Update variables for this cycle
        iteration += 1
        norm_next = map(i -> norm(β[i, :] .- β_next[i, :]), 1:d)
        β = copy(β_next)
    end

    # Reconstruct coefficients to be on the original scale
    β_orig = copy(β)
    β_orig = β ./ vec(sds)

    # Finally, add the intercept.
    # This intercept, when scaled by y, accounts for the fact 
    # that when the β are rescaled, they no longer sum to 1. 
    β0 = 1 .- mean(X * β_orig, dims = 1)

    return β_orig, β0
end

function cross_coord_descent(X, X_shift; λ = nothing, α = 1.0, nfolds = 5, min_λ_ε = 0.001, λ_grid_length = 100, max_iters = 1000, tol = 0.0, only_refit_best = true)
    # Get components to standardize data to select initial lambda grid
    n, d = size(X)
    means = mean(X, dims = 1)
    sds = std(X, dims = 1)

    # Standardize the data
    Z_shift = (X_shift .- means) ./ sds
    mean_shift = mean(Z_shift, dims = 1)

    # Compute search grid of lambda
    if isnothing(λ)
        λ_max = maximum(abs.(mean_shift)) / α
        λ_min = min_λ_ε * λ_max    
        λ_range = exp.(range(log(λ_min), log(λ_max), length = λ_grid_length))
    else
        λ_range = λ
    end

    ### Run coordinate descent cross-validated ###
    # Set up initial variables and storage
    train, test = train_test_folds(1:n, nfolds)
    βs = Vector{AbstractMatrix}(undef, nfolds)
    β0s = Vector{AbstractMatrix}(undef, nfolds)
    mse = Vector{AbstractMatrix}(undef, nfolds)


    # Cross-validate coordinate descent
    for fold in 1:nfolds
        X_train = view(X, train[fold], :)
        X_shift_train = view(X_shift, train[fold], :)
        βs[fold], β0s[fold] = coord_descent(X_train, X_shift_train; λ = λ_range, α = α, max_iters = max_iters, tol = tol)
        
        X_test = view(X, test[fold], :)
        X_shift_test = view(X_shift, test[fold], :)
        mse[fold] = riesz_loss(X_test, X_shift_test, βs[fold], β0s[fold])
    end

    mean_mse = mean(reduce(vcat, mse), dims = 1)
    best_λ_index = argmin(vec(mean_mse))

    total_λ_range = only_refit_best ? λ_range[best_λ_index] : λ_range
    β, β0 = coord_descent(X, X_shift; λ = total_λ_range, α = α, max_iters = max_iters, tol = tol)

    return β, β0, total_λ_range
end