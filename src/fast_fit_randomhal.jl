
split_folds(v, n, K) = [v[collect(i:K:n)] for i in 1:K]

# Define a data structure to store the fitted HAL components
mutable struct RandomHALParameters
    sections::AbstractVector{AbstractVector{Int}}
    knots::AbstractVector{AbstractVector{Real}}
    β::AbstractVector{Real}
    smoothness::Int
end

function fast_fit_cv_randomhal(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix, y::AbstractVector{Float64}; K = 5, outer_max_iters = 20, inner_max_iters = 20, λ = nothing, λ_grid_length = 20, min_λ_ε = 0.001)

    # Preprocess outcome variable
    σ_y = sqrt(var(y, corrected=false))
    μ_y = mean(y)
    y_cs = (y .- μ_y) ./ σ_y

    # Construct the indicators to produce a basis
    indblocks = NestedIndicatorBlocks(sections, X)

    # Construct the basis and variance estimates for the training data
    B = NestedMatrixBlocks(indblocks, X)
    σ2 = transpose(B) * ones(B.nrow)

    # If λ is unspecified, automatically construct a grid.
    # We choose λ_max as the smallest value of λ that will guarantee 
    # all coefficients remain 0 after updating for the first time.
    # β will not change from 0 if λ_max > |mean_shift| / α
    if isnothing(λ)
        λ_max = maximum((transpose(B)*y_cs)) / n
        λ_min = min_λ_ε * λ_max    
        λ_range = reverse(exp.(range(log(λ_min), log(λ_max), length = λ_grid_length)))
    else
        λ_range = reverse(λ)
    end

    # Split the data into K folds
    n = length(y_cs)
    folds = split_folds(sample(1:n, n), n, K)

    mse = Vector{Float64}(undef, K)
    for k in 1:K
        train = reduce(vcat, folds[Not(k)])
        test = folds[k]

        Bt = B[train, :]
        yt = y_cs[train]

        β = fast_fit_randomhal(sections, Bt, yt; outer_max_iters = outer_max_iters, inner_max_iters = inner_max_iters, λ_range = λ_range)

        mse[k] = mean((y_cs .- Bt * β).^2)
    end
    
end

fold_indices = [findall(x -> x == k, Iterators.partition(1:n, K)) for k in 1:K]


n = length(y)
sample(1:n,n)
K = 6

roundn / K