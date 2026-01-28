# This file defines a custom coordinate descent algorithm
# that takes advantage of the fast HAL basis structure

soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)
conv_crit(β_prev::AbstractVector{Float64}, β_next::AbstractVector{Float64}, σ2::AbstractVector{Float64}) = maximum(σ2 .* (β_next .- β_prev).^2)
pct_change(next_loss::Float64, prev_loss::Float64) = abs(next_loss - prev_loss) / prev_loss

function cycle_coord!(active::AbstractVector, β::AbstractVector{Float64}, X::NestedMatrixBlocks, y::AbstractVector{Float64}, μ::AbstractVector{Float64}, invσ::AbstractVector{Float64}, α::Float64, λ::Float64, n)
    # Compute residuals for entire basis.
    # Fast nesting structure requires we sum over all of the coefficients anyways,
    # so we don't skip inactive set for the high-level residual computation.
    # (could technically exclude inactive tails, but this introduces its own nontrivial overhead)
    β_scaled = β .* invσ
    r = (y - X * β_scaled) .+ dot(μ, β_scaled)
    rsum = sum(r)
    cur_ind  = 1

    for XB in X.blocks
        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unpenalized = ((((transpose(XB) * r) .- (μ[indices].*rsum)) .* invσ[indices])./ XB.nrow) .+ β[indices]

        # Set up variable to track cumulative change in residual update
        Δ_r = 0

        # Sequentially update residuals within the union of the current block and the active set and soft-threshold
        for k in indices
            if active[k]
                # These variables help track the sequential change in residuals
                # for fast O(n) computation
                β_prev = β[k]
                Δ = (1 - μ[k])*Δ_r*invσ[k]

                # Apply the lasso thresholding
                β[k] = soft_threshold(β_unpenalized[k] - Δ, λ*α) / (1 + (1 - α)*λ)

                # Update the change in residuals to avoid recomputing every subsequent coefficient
                Δ_r += μ[k] * (β[k] - β_prev) * invσ[k]
            end
        end

        # Update residuals for the given block
        β_scaled = β .* invσ
        r = (y - X * β_scaled) .+ dot(μ, β_scaled)
        rsum = sum(r)

        # Update indices to the next block
        cur_ind += XB.ncol
    end
end

function coord_descent(X::NestedMatrixBlocks, y::AbstractVector{Float64}, μ::AbstractVector{Float64}, σ2::AbstractVector{Float64}, λ_range::AbstractVector{Float64}; outer_max_iters = 20, inner_max_iters = 50, tol = 0.0001, α=1.0, warm_β = nothing)

    # Check input
    n = X.nrow
    d = X.ncol
    n == length(y) || error("Number of rows in X must match length of y")

    # Compute inverse standard deviation for scaling
    invσ = 1 ./ sqrt.(σ2) # This is right
    invσ[isinf.(invσ)] .= 0.0  # Handle zero-variance basis functions

    # Set up storage for coefficients
    β = Matrix(undef, d, length(λ_range))
    β_prev = isnothing(warm_β) ? zeros(d) : warm_β
    β_next = β_prev

    for (λ_index, λ) in enumerate(λ_range)
        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 
        active = trues(d)
        norm_next = tol .+ 1.0
        outer_iteration = 1

        # Run an initial update
        cycle_coord!(trues(d), β_next, X, y, μ, invσ, α, λ, n)

        # Begin iterative descent
        while (outer_iteration < outer_max_iters)
            # Initial full set iteration. Iterate through each coordinate cyclically
            cycle_coord!(trues(d), β_next, X, y, μ, invσ, α, λ, n)

            # Update active set until convergence
            inner_iteration = 1
            while (inner_iteration < inner_max_iters) && (norm_next > tol)
                cycle_coord!(active, β_next, X, y, μ, invσ, α, λ, n)

                # Track convergence
                norm_next = conv_crit(β_prev, β_next, σ2)
                #println("Norm next: ", norm_next)

                β_prev .= β_next
                inner_iteration += 1
            end
            #println("Inner ", inner_iteration)

                        # Update the active set
            next_active = β_next .!= 0
            
            # If the active set has not changed, then we're done. Otherwise, keep going
            #println("Active set size: ", sum(next_active))
            active == next_active && break
            active = next_active

            outer_iteration += 1
        end
        #println("Outer: ", outer_iteration)
        β[:, λ_index] = β_next
    end
    return β
end





