soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)

function update_coef(i, β, Z, mean_shift, λ, α, n)
    # Get predictions without the contribution of the ith feature
    held_out_preds = Z[:, Not(i)] * β[Not(i)]

    # Compute the squared penalty of the Riesz representer
    square_penalty = dot(Z[:, i], held_out_preds) ./ n

    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    β_next = soft_threshold(mean_shift[i] - square_penalty, α * λ) / (1 + (1-α) * λ)
    return β_next
end

# TODO: Assumes no intercept in the design matrix
function coord_descent(X, X_shift, λ, α; max_iters = 1000, tol = 1e-4)
    # Initialize variables
    n, d = size(X)
    β = zeroes(d + 1)
    β_next = zeroes(d + 1)
    iteration = 0

    # Get components to standardize data
    means = mean(X, dims = 2)
    sds = std.(X, dims = 2)

    # Standardize the data
    Z = (X .- means) ./ sds
    Z_shift = (X_shift .- means) ./ sds
    mean_shift = mean(Z_shift, dims = 2)

    # Add an intercept to the design matrix
    Z = hcat(ones(n, 1), Z)
    Z_shift = hcat(ones(n, 1), Z)
    mean_shift = vcat([0], mean_shift)

    # Descend until convergence or max iterations
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

    # Reconstruct coefficients to be on the original scale
    β_orig = copy(β)
    β_orig[2:end] = β[2:end] ./ sds

    return β
end