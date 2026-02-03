# This file defines a custom coordinate descent algorithm
# that takes advantage of the fast HAL basis structure

soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)
conv_crit(β_prev::Vector{Float64}, β_next::Vector{Float64}, σ2::Vector{Float64}) = maximum(σ2 .* (β_next .- β_prev).^2)
pct_change(next_loss::Float64, prev_loss::Float64) = abs(next_loss - prev_loss) / prev_loss

# This function currently produces a lot of allocations. 
# May be able to reduce these with clever programming tricks

function cycle_coord_julia!(active::BitVector, β::Vector{Float64}, X::NestedMatrixBlocks, y::Vector{Float64}, 
                      μ::Vector{Float64}, invσ::Vector{Float64}, μinvσ::Vector{Float64}, μinvσdif::Vector{Float64}, 
                      lasso_penalty::Float64, ridge_penalty::Float64)
    
    cur_ind  = 1
    for XB in X.blocks
        # Compute residuals for entire basis.
        # Fast nesting structure requires we sum over all of the coefficients anyways,
        # so we don't skip inactive set for the high-level residual computation.
        # (could technically exclude inactive tails, but this introduces its own nontrivial overhead)        β_scaled = β .* invσ
        β_scaled = β .* invσ
        r = (y .- (X * β_scaled)) .+ dot(μ, β_scaled)

        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unpenalized = ((((transpose(XB) * r) .- (μ[indices].*sum(r))) .* invσ[indices])./ XB.nrow) .+ β[indices]

        # Set up variable to track cumulative change in residual update
        Δ_r = 0

        # Sequentially update residuals within the union of the current block and the active set and soft-threshold
        # WARNING: This loop produces most of the allocations in this function, because it's about 20 allocations per coefficient
        for k in indices
            if active[k]
                # These variables help track the sequential change in residuals
                # for fast O(n) computation
                β_prev = β[k]
                #Δ = (1 - μ[k])*invσ[k]*Δ_r
                Δ = μinvσdif[k]*Δ_r

                # Apply the lasso thresholding
                β[k] = soft_threshold(β_unpenalized[k] - Δ, lasso_penalty) / ridge_penalty

                # Update the change in residuals to avoid recomputing every subsequent coefficient
                #Δ_r += μ[k] * invσ[k] * (β[k] - β_prev)
                Δ_r += μinvσ[k] * (β[k] - β_prev)   
            end
        end

        # Update indices to the next block
        cur_ind += XB.ncol
    end
end

function update_coefficients!(indices, active::Vector{Int8}, β, β_unp, μinvσdif, μinvσ, λ_l, λ_r)

    # Convert indices
    indices_c = Int64.(indices .- 1)
    
    # Call the C function
    ccall((:coordinate_descent_update, LIB_PATH), 
        Cvoid, 
        (Ptr{Int64}, Int32, Ptr{Int8}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64, Float64),
        indices_c, length(indices_c), active, β, β_unp, μinvσdif, μinvσ, λ_l, λ_r)
    
    return nothing
end

function cycle_coord!(active::Vector{Int8}, β::Vector{Float64}, X::NestedMatrixBlocks, y::Vector{Float64}, 
                      μ::Vector{Float64}, invσ::Vector{Float64}, μinvσ::Vector{Float64}, μinvσdif::Vector{Float64}, 
                      lasso_penalty::Float64, ridge_penalty::Float64)
    
    cur_ind  = 1
    for XB in X.blocks
        # Compute residuals for entire basis.
        # Fast nesting structure requires we sum over all of the coefficients anyways,
        # so we don't skip inactive set for the high-level residual computation.
        # (could technically exclude inactive tails, but this introduces its own nontrivial overhead)        β_scaled = β .* invσ
        β_scaled = β .* invσ
        r = (y .- (X * β_scaled)) .+ dot(μ, β_scaled)

        # Get the coefficient indices for the current block
        indices = cur_ind:(cur_ind + XB.ncol - 1)

        # Compute unpenalized coefficient update for entire block
        β_unpenalized = ((((transpose(XB) * r) .- (μ[indices].*sum(r))) .* invσ[indices])./ XB.nrow) .+ β[indices]

        # Set up variable to track cumulative change in residual update
        Δ_r = 0

        # Sequentially update residuals within the union of the current block and the active set and soft-threshold
        # WARNING: This loop produces most of the allocations in this function, because it's about 20 allocations per coefficient
        update_coefficients!(indices, active, β, β_unpenalized, μinvσdif, μinvσ, lasso_penalty, ridge_penalty)

        # Update indices to the next block
        cur_ind += XB.ncol
    end
end

function coord_descent(X::NestedMatrixBlocks, y::Vector{Float64}, μ::Vector{Float64}, σ2::Vector{Float64}, λ_range::Vector{Float64}; outer_max_iters::Int64 = 1000, inner_max_iters::Int64 = 1000, tol::Float64 = 1e-6, α::Float64 = 1.0)

    # Check input
    n = X.nrow
    d = X.ncol
    n == length(y) || error("Number of rows in X must match length of y")

    # Compute inverse standard deviation for scaling
    invσ = 1 ./ sqrt.(σ2) # This is right
    invσ[isinf.(invσ)] .= 0.0  # Handle zero-variance basis functions

    # Precompute some quantities for cycling
    μinvσ = μ .* invσ
    μinvσdif = invσ .- μinvσ

    # Set up storage for coefficients
    β = Matrix{Float64}(undef, d, length(λ_range))
    
    #warm_path = false
    #if typeof(warm_β) <: Matrix
    #    warm_path = true
    #    β_prev = Vector{Float64}(undef, d)
    #elseif isnothing(warm_β)
    #    β_prev = zeros(d)
    #else
    #    β_prev = warm_β
    #end
    β_prev = zeros(d)
    β_next = copy(β_prev)
    β_unpenalized = copy(β_prev)

    # Change behavior if we've provided an entire path of warm starts
    outer_total_tracker = 0
    inner_total_tracker = 0

    for (λ_index, λ) in enumerate(λ_range)
        # Compute penalties
        lasso_penalty = λ*α
        ridge_penalty = 1 - (1 - α)*λ

        # Before we begin, update β vectors with pre-existing guesses if using a warm path
        #if warm_path
        #    β_prev = warm_β[:, λ_index]
        #    β_next .= β_prev
        #end

        # First, cycle through all variables to determine the active set
        # Then, iterate on the active set until convergence
        # Finally, repeat on the entire set of variables. If nothing changes, done!
        # Otherwise, update the active set and repeat 

        # The active set is defined initially by the sequential strong rule (Tibshirani et al. 2010)
        #if λ_index == 1
        #    active = abs.(transpose(X) * y) .< λ
        #else
        #    active = abs.(transpose(X) * (y - X * β[:,λ_index-1])) .< ((2*λ) - λ_range[λ_index - 1])
        #end
        active = ones(Int8, d)
        next_active = copy(active)
        outer_iteration = 1

        # Run an initial update
        cycle_coord!(ones(Int8, d), β_next, X, y, μ, invσ, μinvσ, μinvσdif, lasso_penalty, ridge_penalty)

        # Begin iterative descent
        while (outer_iteration < outer_max_iters)
            # Update the active set and norm for next sub-cycle
            norm_next = tol .+ 1.0

            # Update active set until convergence
            inner_iteration = 1
            while (inner_iteration < inner_max_iters) && (norm_next > tol)
                cycle_coord!(ones(Int8, d), β_next, X, y, μ, invσ, μinvσ, μinvσdif, lasso_penalty, ridge_penalty)

                # Track convergence
                norm_next = conv_crit(β_prev, β_next, σ2)
                #println("Norm next: ", norm_next)

                β_prev .= β_next
                inner_iteration += 1
            end
            inner_total_tracker += inner_iteration

            # One more cycle over all variables to assess if active set changes
            cycle_coord!(ones(Int8, d), β_next, X, y, μ, invσ, μinvσ, μinvσdif, lasso_penalty, ridge_penalty)
            next_active .= Int8.(β_next .!= 0)
            
            # If the active set has not changed, then we're done. Otherwise, keep going
            #println("Active set size: ", sum(next_active))
            active == next_active && break
            active .= next_active
            β_prev .= β_next

            outer_iteration += 1
        end
        #println("Outer: ", outer_iteration)
        outer_total_tracker += outer_iteration
        β[:, λ_index] = β_next
    end
    println("Outer: ", outer_total_tracker, " Inner: ", inner_total_tracker)

    return β
end










