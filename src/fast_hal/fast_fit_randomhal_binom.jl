function fast_fit_cv_randomhal_binom(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix, y::AbstractVector{Float64}; 
    smoothness::Int64 = 0, K::Int64 = 10, outer_max_iters::Int64 = 1000, inner_max_iters::Int64 = 1000, 
    λ = nothing, λ_grid_length::Int64 = 100, min_λ_ε::Float64 = 1e-3, tol::Float64 = 1e-7, α::Float64 = 1.0)

    # Construct the indicators to produce a basis
    indblocks = BasisBlocks(sections, X, smoothness)

    # Construct the basis and variance estimates for the training data
    B = BasisMatrixBlocks(indblocks, X)
    n = B.nrow

    # Fit the initial set of coefficients over the entire dataset
    μ = colmeans(B)
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    σ2[σ2 .< 0.0] .= 0.0 # Handle numerical issues with negative variance estimates
    invσ = 1 ./ sqrt.(σ2)
    invσ[isinf.(invσ)] .= 0.0  # Handle zero-variance basis functions

    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    if isnothing(λ)
        corrs = ((transpose(B)*y) .- (μ .* sum(y))) .* invσ
        λ_max = maximum(abs.(corrs)) / n
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = sort(λ)
    end

    β, β0 = coord_descent(B, y, μ, invσ, σ2, λ_range; outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol, α = α)

    # Split the data into K folds
    folds = split_folds(sample(1:n, n), n, K)

    # Cross-fit the highly adaptive lasso to select best lambda
    dev = Vector{Matrix{Float64}}(undef, K)
    for k in 1:K
        # Split the data into training and testing folds
        train = reduce(vcat, folds[Not(k)])
        val = folds[k]

        Bt = B[train]
        yt = y[train]

        # Compute variables to center and scale each training column implicitly in the coordinate descent algorithm
        μt = (transpose(Bt) * ones(Bt.nrow)) ./ Bt.nrow
        σ2t = (squares(transpose(Bt)) ./ Bt.nrow) .- (μt.^2)
        σ2t[σ2t .< 0.0] .= 0.0 # Handle numerical issues with negative variance estimates
        invσt = 1 ./ sqrt.(σ2t)
        invσt[isinf.(invσt)] .= 0.0

        βt, β0t = coord_descent(Bt, yt, μt, invσt, σ2t, λ_range; outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol, α = α)

        # Evaluate mean-squared error on validation set
        Bv = B[val]
        predv = expit.((Bv * βt) .+ β0t)

        yv = y[val]
        dev[k] = -mean(yv .* log.(predv) .+ (1 .- yv) .* log.(1 .- predv), dims=1)
    end

    # Compute which λ value was best over the cross-validated folds
    dev_matrix = vcat(dev...)
    test_dev = vec(mean(dev_matrix, dims = 1))
    best_index = argmin(test_dev)

    # Obtain coefficients corresponding to best λ value
    β_final = β[:, best_index]
    β0_final = β0[best_index]

    return RandomHALParameters(indblocks, vec(β_final), β0_final, λ_range[best_index])
end

function predict_randomhal_binom(model::RandomHALParameters, X::AbstractMatrix)
    B = BasisMatrixBlocks(model.indblocks, X)
    return expit.((B * model.β) .+ model.β0)
end