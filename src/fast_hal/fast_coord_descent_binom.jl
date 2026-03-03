# This function currently produces a lot of allocations. 
# May be able to reduce these with clever programming tricks

function update_coefficients_binom!(indices, active::BitVector, β, β_unp, β_prev, l_sum, l_squares, r_shift, hessian_bound, nz_sum, μ, invσ, lasso_penalty::Float64, cur_ind::Int64, n)
    
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
            β[k] = soft_threshold(β_unp[k - cur_ind + 1] - Δ, lasso_penalty) / hessian_bound

            # Compute components needed to update the next β_unp
            # that involve the kth entries of vectors

            invσΔβ            =  invσ[k] * (β[k] - β_prev[k]) / n
            Δ_intercept_part +=  invσΔβ * (l_squares[k] - r_shift[k]*l_sum[k])
            Δ_scaled_part    +=  invσΔβ * (nz_sum[k]*r_shift[k] - l_sum[k])
            #Δ_μ1 += invσΔβ * μ[k]
            #Δ_μ2 += invσΔβ * (l_sum[k] - nz_sum[k]*r_shift[k])
            Δ_μ3 += invσΔβ * hessian_bound * n * μ[k]

        end
    end
end

function cycle_coord_binom!(active::BitVector, β, β_prev, X::BasisMatrixBlocks, 
                      hessian_bound, z, lin_preds,
                      l_sum, l_squares, r_shift, nz_sum, μ, invσ,
                      lasso_penalty::Float64)    

    # Iterate through each block of the basis
    cur_ind  = 1
    res = z .- lin_preds
    for XB in X.blocks

        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unp = hessian_bound .* (view(β, indices) .+ (((transpose(XB) * res) .- (view(μ, indices).*sum(res))) .* view(invσ, indices)) ./ X.nrow)

        # Update coefficients sequentially
        update_coefficients_binom!(indices, active, β, β_unp, β_prev, l_sum, l_squares, r_shift, hessian_bound, nz_sum, μ, invσ, lasso_penalty, cur_ind, X.nrow)

        # Update residuals in-place to avoid excessive allocations
        dif = (view(β, indices)  - view(β_prev, indices)) .* view(invσ, indices)
        lin_preds .+= ((XB * dif) .- sum(view(μ, indices) .* dif))
        res .= z .- lin_preds

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
    σ2[σ2 .< 0.0] .= 0.0
    invσ = 1 ./ sqrt.(σ2) # This is right
    invσ[isinf.(invσ)] .= 0.0  # Handle zero-variance basis functions

    # Initialize probability, weights, and working response for Newton descent
    # We use the Hessian bound of 0.25 suggested in Friedman et al. 2010 to avoid divergence issues with the true IRLS weights.
    pr = fill(0.5, n)
    hessian_bound = 0.25
    z = ((y .- pr) ./ hessian_bound) 
    lin_preds = zeros(n)

    # Precompute some quantities for cycling
    l_sum = hessian_bound .* left_sum(transpose(X))
    l_squares = left_squares(transpose(X)) .* hessian_bound
    r_shift = reduce(vcat, XB.r for XB in X.blocks)
    nz_sum = nonzero_count(transpose(X)) .* hessian_bound

    # Set up storage for coefficients
    β = Matrix{Float64}(undef, d, length(λ_range))
    β_prev = zeros(d)
    β_newton_prev = zeros(d)
    β_next = zeros(d)
    β0 = zeros(length(λ_range))
    β0_next = 0
    β0_prev = 0
    lin_preds = zeros(n)

    for (λ_index, λ) in enumerate(λ_range)

        # Compute penalties
        lasso_penalty = λ*α

        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 

        newton_iteration = 1
        while (newton_iteration < newton_max_iters)

            # Initialize active set and iteration counter
            active = trues(d)
            next_active = copy(active)   

            # Run an initial update
            # Note the reason we update the intercept separately is because it is a Float not passed by reference,
            # so we need to update β0_prev outside of a function call
            β0_next = mean(z .- lin_preds .+ β0_next)
            lin_preds .+= β0_next .- β0_prev
            cycle_coord_binom!(trues(d), β_next, β_prev, X, hessian_bound, z, lin_preds, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty)
            β_prev .= β_next
            β0_prev = β0_next


            # Begin iterative descent
            outer_iteration = 1
            while (outer_iteration < outer_max_iters)
                # Update the active set and norm for next sub-cycle
                norm_next = tol .+ 1.0

                # Update active set until convergence
                inner_iteration = 1
                while (inner_iteration < inner_max_iters) && (norm_next > tol)
                    β0_next = mean(z .- lin_preds .+ β0_next)
                    lin_preds .+= β0_next .- β0_prev
                    cycle_coord_binom!(active, β_next, β_prev, X, hessian_bound, z, lin_preds, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty)

                    # Track convergence
                    norm_next = conv_crit(β_prev, β_next, σ2)
                    β_prev .= β_next
                    β0_prev = β0_next
                    inner_iteration += 1
                end

                # One more cycle over all variables to assess if active set changes
                β0_next = mean(z .- lin_preds .+ β0_next)
                lin_preds .+= β0_next .- β0_prev
                cycle_coord_binom!(trues(d), β_next, β_prev, X, hessian_bound, z, lin_preds, l_sum, l_squares, r_shift, nz_sum, μ, invσ, lasso_penalty)
                next_active .= β_next .!= 0
                β_prev .= β_next
                β0_prev = β0_next
                
                # If the active set has not changed, then we're done. Otherwise, keep going
                active == next_active && break
                active .= next_active

                outer_iteration += 1
            end
            newton_iteration += 1
            
            ### Perform Newton update ###
            # Has Newton's method converged? If so, we're done.
            conv_crit(β_newton_prev, β_next, σ2) < tol && break
            β_newton_prev .= β_next

            # Otherwise, update the IRLS weights
            β_scaled = (β_next .* invσ)
            lin_preds = (X * β_scaled) .- sum(μ .* β_scaled) .+ β0_next
            pr .= 1 ./ (1 .+ exp.(-(lin_preds)))

            # Deal with possible divergence issues
            lower_diverging_pr = pr .< 10e-5
            upper_diverging_pr = (1 .- pr).< 10e-5
            pr[lower_diverging_pr] .= 0
            pr[upper_diverging_pr] .= 1

            # Update working response for next Newton iteration
            z .= ((y .- pr) ./ hessian_bound) .+ lin_preds
        end

        # Store final output
        β[:, λ_index] = β_next
        β0[λ_index] = β0_next
    end
    return β, β0
end