
# TODO: FIX LOSSES SO THEY ALSO ARE FAST AND ITERATE THROUGH COLUMNS INSTEAD OF ROWS
riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β::AbstractMatrix, β0::AbstractMatrix) = mean((X * β .+ β0).^2, dims = 1) .- mean(2 .* ((X_shift * β) .+ β0), dims = 1)

# Losses for checking individual updates
reg_riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector, λ) = mean((X * β).^2) - 2 * sum(mean_shift .* β) + 2*λ * sum(abs.(β))

# Optimized Riesz Loss
function riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector)
    pred = X * β
    return dot(pred, pred)/size(X, 1) - 2 * dot(mean_shift, β)
end
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

function cross_coord_descent(X, X_shift; nfolds = 5, λ = nothing, α = 1.0, min_λ_ε = 0.01, λ_grid_length = 100, outer_max_iters = 20, inner_max_iters = 20, tol = 0.01, only_refit_best = true)
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
        βs[fold], β0s[fold] = coord_descent(X_train, X_shift_train; λ = λ_range, α = α, outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol)
        
        X_test = view(X, test[fold], :)
        X_shift_test = view(X_shift, test[fold], :)
        mse[fold] = riesz_loss(X_test, X_shift_test, βs[fold], β0s[fold])
    end

    mean_mse = mean(reduce(vcat, mse), dims = 1)
    best_λ_index = argmin(vec(mean_mse))

    total_λ_range = only_refit_best ? λ_range[best_λ_index] : λ_range
    β, β0 = coord_descent(X, X_shift; λ = total_λ_range, α = α, outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol)

    return β, β0, total_λ_range
end

# Get predictions from a set of fitted HAL parameters and new data
function predict_rieszhal(params::HALParameters, Xnew)
    x = Tables.Columns(Xnew)
    if(length(params.β) > 0)
        basis = ha_basis_matrix(x, params.sections, params.knots, params.smoothness)
        return (basis * params.β) .+ params.β0
    else
        return params.β0 .* ones(nrow(x))
    end
end