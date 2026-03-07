
# Utility losses for each family
loss(family::Normal, y::Vector{Float64}, pred::AbstractMatrix{Float64}) = mean((y .- pred).^2, dims = 1)
loss(family::Binomial, y::Vector{Float64}, pred::AbstractMatrix{Float64}) = -mean((y .* log.(pred)) .+ ((1 .- y) .* log.(1 .- pred)), dims = 1)

# Prediction functions that apply proper link depending on the model family
get_predictions(family::Normal, B::AbstractNestedMatrix, β, β0) = (B * β) .+ β0
get_predictions(family::Binomial, B::AbstractNestedMatrix, β, β0) = expit.((B * β) .+ β0)

# Utility functions to map family to call correct coordinate descent algorithm
function coordinate_descent(family::Normal, B::AbstractNestedMatrix, y::AbstractVector{Float64}, 
                            μ::AbstractVector{Float64}, invσ::AbstractVector{Float64}, σ2::AbstractVector{Float64}, λ_range::AbstractVector{Float64};
                            kwargs...)
    kwargs = values(kwargs)
    return coord_descent(B, y, μ, invσ, σ2, λ_range; 
        outer_max_iters = kwargs.outer_max_iters, 
        inner_max_iters = kwargs.inner_max_iters, 
        tol = kwargs.tol, α = kwargs.α)
end

function coordinate_descent(family::Binomial, B::RandomHAL.AbstractNestedMatrix, y::AbstractVector{Float64}, 
                            μ::AbstractVector{Float64}, invσ::AbstractVector{Float64}, σ2::AbstractVector{Float64}, λ_range::AbstractVector{Float64};
                            kwargs...)
    kwargs = values(kwargs)
    return coord_descent_binom(B, y, μ, invσ, σ2, λ_range; 
        newton_max_iters = kwargs.newton_max_iters, 
        outer_max_iters = kwargs.outer_max_iters, 
        inner_max_iters = kwargs.inner_max_iters, 
        tol = kwargs.tol, α = kwargs.α)
end


# Define a data structure to store the fitted HAL components
mutable struct RandomHALParameters{T <:Distribution}
    indblocks::BasisBlocks
    β::AbstractVector{Float64}
    β0::Float64
    best_λ::Float64
    family::T
end

function fast_fit_cv_randomhal(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix, y::AbstractVector{Float64}; 
    smoothness::Int64 = 0, family::Distribution = Normal(), max_block_size::Int = length(y),
    K::Int64 = 10, outer_max_iters::Int64 = 100, inner_max_iters::Int64 = 100, newton_max_iters::Int64 = 25,
    λ = nothing, n_λ::Int64 = 100, min_λ_ε::Float64 = 1e-3, tol::Float64 = 1e-7, α::Float64 = 1.0)

    # Preprocess outcome variable
    if isa(family, Normal)
        σ_y = sqrt(var(y, corrected=false))
        μ_y = mean(y)
        y_cs = (y .- μ_y) ./ σ_y
    else
        y_cs = y
    end
    n = length(y_cs)

    # Construct the indicators to produce a basis
    # TODO: Could change this to just take a vector of integers as a subset
    # That way users can sample after the full basis has been constructed,
    # i.e. based on number of entries.
    indblocks = subsample(BasisBlocks(sections, X, smoothness), max_block_size)

    # Construct the basis and variance estimates for the training data
    # CHECK THAT LINE 278 of fastbasis.jl IS USING THE RIGHT length(indicators.path) IN THE SUBTRACTION
    B = BasisMatrixBlocks(indblocks, X)

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
        corrs = ((transpose(B)*y_cs) .- (μ .* sum(y_cs))) .* invσ
        λ_max = maximum(abs.(corrs)) / n
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = n_λ)))
    else
        λ_range = sort(λ)
    end

    # Run an initial fit
    β, β0 = coordinate_descent(family, B, y_cs, μ, invσ, σ2, λ_range; 
                               outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, 
                               newton_max_iters = newton_max_iters, tol = tol, α = α)

    # Split the data into K folds
    folds = split_folds(shuffle(1:n), n, K)

    # Cross-fit the highly adaptive lasso to select best lambda
    loss_vec = Vector{Matrix{Float64}}(undef, K)
    for k in 1:K
        # Split the data into training and testing folds
        train = reduce(vcat, folds[Not(k)])
        val = folds[k]

        Bt = BasisMatrixBlocks(indblocks, X[train, :])
        yt = y_cs[train]

        # Compute variables to center and scale each training column implicitly in the coordinate descent algorithm
        μt = colmeans(Bt)
        σ2t = (squares(transpose(Bt)) ./ Bt.nrow) .- (μt.^2)
        σ2t[σ2t .< 0.0] .= 0.0 # Handle numerical issues with negative variance estimates
        invσt = 1 ./ sqrt.(σ2t)
        invσt[isinf.(invσt)] .= 0.0

        βt, β0t = coordinate_descent(family, Bt, yt, μt, invσt, σ2t, λ_range; 
                                     outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, 
                                     newton_max_iters = newton_max_iters, tol = tol, α = α)

        # Evaluate loss on validation set
        Bv = BasisMatrixBlocks(indblocks, X[val, :])
        loss_vec[k] = loss(family, y_cs[val], get_predictions(family, Bv, βt, β0t))
    end

    # Compute which λ value was best over the cross-validated folds
    loss_matrix = reduce(vcat, loss_vec)
    test_loss = vec(sum((length.(folds) ./ n) .* loss_matrix, dims = 1))
    best_index = argmin(test_loss)

    if isa(family, Normal)
        # Rescale the fitted coefficients to be on the original scale of y and add mean of y
        RandomHALParameters(indblocks, vec(β[:, best_index] .* σ_y), (β0[best_index]* σ_y) + μ_y, 
                            λ_range[best_index], family)
    else
        RandomHALParameters(indblocks, vec(β[:, best_index]), β0[best_index], 
                            λ_range[best_index], family)
    end
end

function predict_randomhal(model::RandomHALParameters, X::AbstractMatrix)
    # Construct basis matrix from data
    B = BasisMatrixBlocks(model.indblocks, X)
    
    # Apply appropriate prediction function for the model's family via dispatch
    return get_predictions(model.family, B, model.β, model.β0)
end