
using InvertedIndices

split_folds(v, n, K) = [v[collect(i:K:n)] for i in 1:K]

# Define a data structure to store the fitted HAL components
mutable struct RandomHALParameters
    indblocks::NestedIndicatorBlocks
    β::AbstractVector{Float64}
    β0::Float64
end

function fast_fit_cv_randomhal(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix, y::AbstractVector{Float64}; K = 10, outer_max_iters = 1000, inner_max_iters = 1000, λ = nothing, λ_grid_length = 100, min_λ_ε = 0.0001, tol = 1e-7, α = 1.0)

    # Preprocess outcome variable
    σ_y = sqrt(var(y, corrected=false))
    μ_y = mean(y)
    y_cs = (y .- μ_y) ./ σ_y
    n = length(y_cs)

    # Construct the indicators to produce a basis
    indblocks = NestedIndicatorBlocks(sections, X)

    # Construct the basis and variance estimates for the training data
    B = NestedMatrixBlocks(indblocks, X)

    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    if isnothing(λ)
        λ_max = maximum(abs.(transpose(B)*y_cs)) / n
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = reverse(λ)
    end

    # Split the data into K folds
    folds = split_folds(sample(1:n, n), n, K)

    # Cross-fit the highly adaptive lasso to select best lambda
    mse = Vector{Matrix{Float64}}(undef, K)
    β = Vector{Matrix{Float64}}(undef, K)
    for k in 1:K
        # Split the data into training and testing folds
        train = reduce(vcat, folds[Not(k)])
        val = folds[k]

        Bt = B[train]
        yt = y_cs[train]

        # Compute variables to center and scale each training column implicitly in the coordinate descent algorithm
        μt = (transpose(Bt) * ones(Bt.nrow)) ./ Bt.nrow
        σ2t = (squares(transpose(Bt)) ./ Bt.nrow) .- (μt.^2)

        # Fit the coefficients
        β[k] = coord_descent(Bt, yt, μt, σ2t, λ_range; outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol, α = α)
        
        # Evaluate mean-squared error on validation set
        Bv = B[val]
        yv = y_cs[val]
        mse[k] = mean((yv .- Bv * β[k]).^2, dims=1)
    end

    # Compute which λ value was best over the cross-validated folds
    mse_matrix = vcat(mse...)
    test_mse = vec(mean(mse_matrix, dims = 1))
    best_index = argmin(test_mse)
    λ_best = λ_range[best_index]

    # Extract the "best" set of coefficients in order to warm-start the final fitting
    best_fold = argmin(mse_matrix[:,best_index])
    β_best = β[best_fold][:, best_index]

    # Fit the final set of coefficients over the entire dataset
    μ = (transpose(B) * ones(B.nrow)) ./ B.nrow
    σ2 = (squares(transpose(B)) ./ B.nrow) .- (μ.^2)
    β = coord_descent(B, y_cs, μ, σ2, [λ_best]; outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, tol = tol, α = α, warm_β = β_best)
    
    # Rescale the fitted coefficients to be on the original scale
    β_final = (β ./ sqrt.(σ2)) .* σ_y
    β_final[isnan.(β_final)] .= 0.0

    # Compute the intercept
    β0 = μ_y - (reshape(μ, 1, n) * β_final)[1,1]
    print("Best λ: ", λ_best)

    return RandomHALParameters(indblocks, vec(β_final), β0)
end

function predict_randomhal(model::RandomHALParameters, X::AbstractMatrix)
    B = NestedMatrixBlocks(model.indblocks, X)
    return (B * model.β) .+ model.β0
end