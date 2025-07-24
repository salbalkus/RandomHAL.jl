riesz_loss(X::AbstractMatrix, X_shift::AbstractMatrix, β::AbstractMatrix, β0::AbstractMatrix) = mean((X * β .+ β0).^2, dims = 1) .- 2 .* mean(((X_shift * β) .+ β0), dims = 1)


# Losses for checking individual updates
reg_riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector, λ) = mean((X * β).^2) - 2 * sum(mean_shift .* β) + 2*λ * sum(abs.(β))

# Optimized Riesz Loss
function riesz_loss(X::AbstractMatrix, mean_shift::AbstractVector, β::AbstractVector)
    pred = X * β
    return dot(pred, pred)/size(X, 1) - 2 * dot(mean_shift, β)
end

pct_change(next_loss::Float64, prev_loss::Float64) = abs(next_loss - prev_loss) / prev_loss

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

function coord_descent(X::AbstractMatrix, X_shift::AbstractMatrix; λ = nothing, min_λ_ε = 0.001, λ_grid_length = 20, outer_max_iters = 20, inner_max_iters = 50, tol = 0.0001)
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

function cross_coord_descent(X, X_shift; nfolds = 5, λ = nothing, min_λ_ε = 0.001, λ_grid_length = 100, outer_max_iters = 20, inner_max_iters = 20, tol = 0.01, only_refit_best = true)
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
    if only_refit_best
        β0 = β0[1]
    end
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