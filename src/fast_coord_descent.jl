soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)
loss(β::AbstractVector{Float64}, X::NestedMatrixBlocks, y::AbstractVector{Float64}) = mean((y .- X * β).^2)
pct_change(next_loss::Float64, prev_loss::Float64) = abs(next_loss - prev_loss) / prev_loss



function cycle_coord!(active::AbstractVector, β::AbstractVector{Float64}, X::NestedMatrixBlocks, y::AbstractVector{Float64}, σ2::AbstractVector{Float64}, α::Float64, λ::Float64)
    # Compute residuals for entire basis.
    # Fast nesting structure requires we sum over all of the coefficients anyways,
    # so we don't skip inactive set for the high-level residual computation.
    # (could technically exclude inactive tails, but this introduces its own nontrivial overhead)
    r = y - X * β
    cur_ind  = 1
    for XB in X.blocks
        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unpenalized = (transpose(XB) * r) .+ (σ2 .* β[indices])

        # Set up variable to track change in loss update
        Δ = 0

        # Sequentially update residuals within the union of the current block and the active set and soft-threshold
        for k in indices
            if active[k]
                β_prev = β[k]
                β[k] = soft_threshold(β_unpenalized[k] - Δ, XB.nrow*λ*α) / ((1 + (1 - α)*λ) * σ2[k])
                Δ += (β[k] - β_prev) * σ2[k]
            end
        end

        # Update residuals for the given block
        r .-= XB * β[indices]

        # Update indices to the next block
        cur_ind += XB.ncol
    end
end

function coord_descent(X::NestedMatrixBlocks, XT::NestedMatrixBlocksTranspose, y::AbstractVector{Float64}, σ2::AbstractVector{Float64}; λ = nothing, min_λ_ε = 0.001, λ_grid_length = 20, outer_max_iters = 20, inner_max_iters = 50, tol = 0.0001, α=1.0)
    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    n = X.nrow
    d = X.ncol

    if isnothing(λ)
        λ_max = maximum((XT*y)) / n
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = reverse(λ)
    end

    # Set up storage for coefficients
    λ_length = length(λ_range)
    β = Matrix(undef, d, λ_length)
    β_next = zeros(d)

    for (λ_index, λ) in enumerate(λ_range)
        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 
        active = falses(d)
        norm_next = tol .+ 1.0
        outer_iteration = 1

        # Run an initial update
        cycle_coord!(trues(d), β_next, X, XT, y, σ2, α, λ)
        prev_loss = loss(β_next, X, y)

        # Begin iterative descent
        while (outer_iteration < outer_max_iters)
            # Initial full set iteration. Iterate through each coordinate cyclically
            cycle_coord!(trues(d), β_next, X, y, σ2, α, λ)
            
            # Update the active set
            next_active_set = β_next .!= 0
            
            # If the active set has not changed, then we're done. Otherwise, keep going
            active_set == next_active_set && break
            active_set = next_active_set

            # Update the norm to track convergence
            next_loss = loss(β_next, X, y)
            norm_next = pct_change(next_loss, prev_loss)
            prev_loss = next_loss

            # Update active set until convergence
            inner_iteration = 1
            while (inner_iteration < inner_max_iters) && (norm_next > tol)
                cycle_coord!(active, β_next, X, y, σ2, α, λ)

                # Update the norm to track convergence
                next_loss = riesz_loss(Z, mean_shift, β_next)
                norm_next = pct_change(next_loss, prev_loss)
                prev_loss = next_loss
            end
            outer_iteration += 1
        end
        β[:, λ_index] = β_next
    end
    return β
end