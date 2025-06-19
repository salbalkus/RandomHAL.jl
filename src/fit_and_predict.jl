
# Define a data structure to store the fitted HAL components
mutable struct HALParameters
    sections::AbstractVector{AbstractVector{Int}}
    knots::AbstractVector{AbstractVector{Real}}
    β::AbstractVector{Real}
    β0::Real
    smoothness::Int
    basis_type::String
end

# Fit a "vanilla" HAL model
function fit_hal(X, y, family, basis_type::String, smoothness::Int, weights::AbstractVector{<:Real}; glmnet_args...)

    # Convert any Table into a common type
    x = Tables.Columns(X)

    # Construct the basis matrix from the data
    basis, term_lengths = ha_basis_matrix(x, smoothness; basis_type = basis_type)

    # Fit the LASSO model (remember in Julia rightmost keyword arguments take precedence)
    lasso, β, β0, nz = fit_glmnet(basis, y, family; glmnet_args..., weights = weights)

    # Extract the sections and knots representing basis functions selected by LASSO
    sections, knots = get_sections_and_knots(x, nz, term_lengths)

    return HALParameters(sections, knots, β, β0, smoothness, basis_type), lasso
end

# If no weights are provided, assume equal weights
fit_hal(X, y, family, basis_type::String, smoothness::Int, weights::Nothing; glmnet_args...) = fit_hal(X, y, family, basis_type, smoothness, ones(nrow(X)); glmnet_args...)

# Fit a randomized approximation to the full HAL model for computational efficiency
function fit_random_hal(X, y, family, basis_type::String, smoothness::Int, nfeatures::Int, p::Float64, weights::AbstractVector{<:Real}; glmnet_args...)

    # Convert any Table into a common type
    x = Tables.Columns(X)

    # Construct the basis matrix from the data
    sections, knots = random_sections_and_knots(x, nfeatures; p = p)
    basis = ha_basis_matrix(x, sections, knots, smoothness; basis_type = basis_type)

    # Fit the LASSO model
    lasso, β, β0, nz = fit_glmnet(basis, y, family; glmnet_args..., weights = weights)

    # Extract the sections and knots representing basis functions selected by LASSO
    sections = sections[nz]
    knots = knots[nz]

    return HALParameters(sections, knots, β, β0, smoothness, basis_type), lasso
end

# Fit RandomHAL with the asymptotically optimal number of basis functions 
# and theoretically-proven uniform sampling
# TODO: Make the function look at both "family" and "alpha" in glmnet_args
# to automatically detect whether to use the faster log(n) or slower sqrt(n) sampling multiplier
function fit_random_hal(X, y, family, basis_type::String, smoothness::Int, weights::AbstractVector{<:Real}; glmnet_args...)
    # Automatically compute the "asymptotically optimal" number of basis functions
    # for strongly convex loss functions
    nfeatures = Int(round( 0.5 * nrow(X) * log(nrow(X)) ))

    # Run random HAL with uniform sampling
    return fit_random_hal(X, y, family, basis_type, smoothness, nfeatures, 0.5, weights; glmnet_args...)
end

# If no weights are provided, assume equal weights
fit_random_hal(X, y, family, basis_type::String, smoothness::Int, nfeatures::Union{Int, Nothing}, p::Union{Float64, Nothing}, weights::Nothing; glmnet_args...) = fit_random_hal(X, y, family, basis_type, smoothness, nfeatures, p, ones(nrow(X)); glmnet_args...)

# If no nfeatures or p are provided, call function assuming the asymptotically optimal number of basis functions
fit_random_hal(X, y, family, basis_type::String, smoothness::Int, nfeatures::Nothing, p::Nothing, weights::AbstractVector{<:Real}; glmnet_args...) = fit_random_hal(X, y, family, basis_type, smoothness, weights; glmnet_args...)

# Get predictions from a set of fitted HAL parameters and new data
function predict_hal(params::HALParameters, Xnew)
    x = Tables.Columns(Xnew)
    if(length(params.β) > 0)
        basis = ha_basis_matrix(x, params.sections, params.knots, params.smoothness; basis_type = params.basis_type)
        return (basis * params.β) .+ params.β0
    else
        return params.β0 .* ones(nrow(x))
    end
end