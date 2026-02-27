# This function currently produces a lot of allocations. 
# May be able to reduce these with clever programming tricks
function update_coefficients_binom!(indices, active::BitVector, β, β_unp, β_prev, l_sum, l_squares, r_shift, w_sum, nz_sum, w_squares, w_inv_squares, μ, invσ, lasso_penalty::Float64, ridge_penalty::Float64, cur_ind::Int64, n)
    
    # These variables help track the sequential change in residuals
    # for fast O(n) computation
    Δ = 0
    Δ_intercept_part = 0
    Δ_scaled_part = 0
    #Δ_μ1 = 0 # Δ_μ1 and Δ_μ2 cancel each other out; keeping them
    #Δ_μ2 = 0 #  here to understand what the update is mathematically
    Δ_μ3 = 0

    # Sequentially update residuals within the union of the current block and the active set and soft-threshold
    for k in indices
        if active[k]

            # Compute components needed to update the next β_unp
            # that do not get cumulatively summed
            ΔA = (Δ_intercept_part + (r_shift[k] * Δ_scaled_part)) 
            ΔB = Δ_μ3*μ[k]# - Δ_μ2*μ[k] + Δ_μ1*(l_sum[k] - nz_sum[k]*r_shift[k])
            Δ = invσ[k]*(ΔA - ΔB)

            # Apply the lasso thresholding
            β[k] = soft_threshold(β_unp[k - cur_ind + 1] - Δ, lasso_penalty*w_inv_squares[k]) / ridge_penalty

            # Compute components needed to update the next β_unp
            # that involve the kth entries of vectors
            invσΔβ            =  invσ[k] * (β[k] - β_prev[k]) * w_inv_squares[k]
            Δ_intercept_part +=  invσΔβ * (l_squares[k] - r_shift[k]*l_sum[k])
            Δ_scaled_part    +=  invσΔβ * (nz_sum[k]*r_shift[k] - l_sum[k])
            #Δ_μ1 += invσΔβ * μ[k]
            #Δ_μ2 += invσΔβ * (l_sum[k] - nz_sum[k]*r_shift[k])
            Δ_μ3 += invσΔβ * w_sum * μ[k]

        end
    end
end

function cycle_coord_binom!(active::BitVector, β, β_prev, X::BasisMatrixBlocks, 
                      w, w_sum, w_squares, w_inv_squares, res, 
                      l_sum, l_squares, r_shift, nz_sum, μ, invσ,
                      lasso_penalty::Float64, ridge_penalty::Float64)

    cur_ind  = 1
    for XB in X.blocks

        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unp = view(β, indices) .+ ((((transpose(XB) * res)  .- (view(μ, indices).*sum(res))) .* view(invσ, indices)) .* w_inv_squares[indices])

        # Update coefficients sequentially
        update_coefficients_binom!(indices, active, β, β_unp, β_prev, l_sum, l_squares, r_shift, w_sum, nz_sum, w_squares, w_inv_squares, μ, invσ, lasso_penalty, ridge_penalty, cur_ind, X.nrow)

        # Update residuals in-place to avoid excessive allocations
        dif = (view(β, indices)  - view(β_prev, indices)) .* view(invσ, indices)
        res .-= w .* ((XB * dif) .- sum(view(μ, indices) .* dif))

        # Update indices to the next block
        cur_ind += XB.ncol
    end
end


function coord_descent_binom(X::BasisMatrixBlocks, y::Vector, μ::Vector{Float64}, σ2::Vector{Float64}, λ_range::Vector{Float64}; newton_max_iters::Int64 = 100, outer_max_iters::Int64 = 1000, inner_max_iters::Int64 = 1000, tol::Float64 = 1e-7, α::Float64 = 1.0)

    # Check input
    n = X.nrow
    d = X.ncol
    n == length(y) || error("Number of rows in X must match length of y")

    # Compute inverse standard deviation for scaling
    invσ = 1 ./ sqrt.(σ2) # This is right
    invσ[isinf.(invσ)] .= 0.0  # Handle zero-variance basis functions
    invσ2 = invσ.^2
    μ2 = μ.^2

    # Initialize probability, weights, and working response for Netwon desscent
    pr = fill(0.5, X.nrow)
    w = pr .* (1 .- pr)
    res = y .- pr

    # Precompute some quantities for cycling
    l_sum = left_sum(transpose(X), w)
    l_squares = left_squares(transpose(X), w)
    r_shift = reduce(vcat, XB.r for XB in X.blocks)
    nz_sum = nonzero_sum(transpose(X), w)
    w_sum = sum(w)
    w_squares = wls_reweight(transpose(X), w, w_sum, μ, μ2, invσ2)
    w_inv_squares = 1 ./ w_squares
    w_inv_squares[isinf.(w_inv_squares)] .= 0.0


    # Set up storage for coefficients
    β = Matrix{Float64}(undef, d, length(λ_range))
    β_prev = zeros(d)
    β_newton_prev = zeros(d)
    β_next = zeros(d)

    for (λ_index, λ) in enumerate(λ_range)

        # Compute penalties
        lasso_penalty = λ*α
        ridge_penalty = 1 - (1 - α)*λ

        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 

        # Below is some code to implement screening used by GLMNet
        # Commented out to disable it for testing 
        # The active set is defined initially by the sequential strong rule (Tibshirani et al. 2010)
        #if λ_index == 1
        #    active = abs.(transpose(X) * y) .< λ
        #else
        #    active = abs.(transpose(X) * (y - X * β[:,λ_index-1])) .< ((2*λ) - λ_range[λ_index - 1])
        #end     

        newton_iteration = 1
        while (newton_iteration < newton_max_iters)

            # Initialize active set and iteration counter
            active = trues(d)
            next_active = copy(active)   

            # Run an initial update
            cycle_coord_binom!(trues(d), β_next, β_prev, X, w, w_sum, w_squares, w_inv_squares, res, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty, ridge_penalty)
            β_prev .= β_next

            # Begin iterative descent
            outer_iteration = 1
            while (outer_iteration < outer_max_iters)
                # Update the active set and norm for next sub-cycle
                norm_next = tol .+ 1.0

                # Update active set until convergence
                inner_iteration = 1
                while (inner_iteration < inner_max_iters) && (norm_next > tol)
                    cycle_coord_binom!(active, β_next, β_prev, X, w, w_sum, w_squares, w_inv_squares, res, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty, ridge_penalty)

                    # Track convergence
                    norm_next = conv_crit(β_prev, β_next, σ2)
                    β_prev .= β_next
                    inner_iteration += 1
                end

                # One more cycle over all variables to assess if active set changes
                cycle_coord_binom!(trues(d), β_next, β_prev, X, w, w_sum, w_squares, w_inv_squares, res, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty, ridge_penalty)
                next_active .= β_next .!= 0
                
                # If the active set has not changed, then we're done. Otherwise, keep going
                active == next_active && break
                active .= next_active
                β_prev .= β_next

                outer_iteration += 1
            end
            newton_iteration += 1
            
            ### Perform Newton update ###
            # Has Newton's method converged? If so, we're done.
            conv_crit(β_newton_prev, β_next, σ2) < tol && break
            β_newton_prev .= β_next

            # Otherwise, update the IRLS weights
            pr .= 1 ./ (1 .+ exp.(-(X * β_next)))
            w .= pr .* (1 .- pr)

            # Deal with possible divergence issues
            lower_diverging_pr = pr .< 10e-5
            upper_diverging_pr = (1 .- pr).< 10e-5
            pr[lower_diverging_pr] .= 10e-5
            pr[upper_diverging_pr] .= 1 - 10e-5
            w[lower_diverging_pr .|| upper_diverging_pr] .= 10e-5

            w_sum = sum(w)
            res .= y .- pr

            # Use the IRLS weights to update the intermediary variables for fast coefficient updates
            l_sum = left_sum(transpose(X), w)
            l_squares = left_squares(transpose(X), w)
            nz_sum = nonzero_sum(transpose(X), w)
            w_sum = sum(w)
            w_squares = wls_reweight(transpose(X), w, w_sum, μ, μ2, invσ2)
            w_inv_squares = 1 ./ w_squares
            w_inv_squares[isinf.(w_inv_squares)] .= 0.0
        end

        # Store final output
        β[:, λ_index] = β_next
    end
    return β
end
