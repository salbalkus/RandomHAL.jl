soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)

function update_coef(i, β, X, mean_shift, λ, α, n)
    # Get predictions without the contribution of the ith feature
    held_out_preds = X[:, Not(i)] * β[Not(i)]

    # Compute the squared penalty of the Riesz representer
    square_penalty = dot(X[:, i], held_out_preds) ./ n

    # Update ith coefficient using closed-form lasso coordinate for Riesz representer
    β_next = soft_threshold(mean_shift[i] - square_penalty, α * λ) / (1 + (1-α) * λ)
    return β_next
end

# TODO: Assumes X and mean_shift originate from standardized data
# Need to automatically standardize and return rescaled coefficients in the future
function coord_descent(X, mean_shift, λ, α; max_iters = 1000, tol = 1e-4)
    # Initialize variables
    n, d = size(X)
    β = zeroes(d)
    β_next = zeroes(d)
    iteration = 0

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

    return β
end