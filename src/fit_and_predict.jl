
# Define a data structure to store the fitted HAL components
mutable struct HALParameters
    sections::AbstractVector{AbstractVector{Int}}
    knots::AbstractVector{AbstractVector{Real}}
    β::AbstractVector{Real}
    β0::Real
    smoothness::Int
end

# Fit a "vanilla" HAL model
function fit_hal(X, y, family, smoothness::Int, weights::AbstractVector{<:Real}; glmnet_args...)

    # Convert any Table into a common type
    x = Tables.Columns(X)

    # Construct the basis matrix from the data
    basis, all_sections, term_lengths = ha_basis_matrix(x, smoothness)

    # Fit the LASSO model (remember in Julia rightmost keyword arguments take precedence)
    lasso, β, β0, nz = fit_glmnet(basis, y, family; glmnet_args..., weights = weights)

    # Extract the sections and knots representing basis functions selected by LASSO
    sections, knots = get_sections_and_knots(x, nz, all_sections, term_lengths)

    return HALParameters(sections, knots, β, β0, smoothness), lasso
end

# If no weights are provided, assume equal weights
fit_hal(X, y, family, smoothness::Int, weights::Nothing; glmnet_args...) = fit_hal(X, y, family, smoothness, ones(nrow(X)); glmnet_args...)

# Fit a randomized approximation to the full HAL model for computational efficiency
function fit_random_hal(X, y, family, smoothness::Int, nfeatures::Int, sampler_params::NamedTuple, weights::AbstractVector{<:Real}; glmnet_args...)

    # Convert any Table into a common type
    x = Tables.Columns(X)

    # Construct the basis matrix from the data
    sections, knots = random_sections_and_knots(x, nfeatures; sampler_params...)
    basis = ha_basis_matrix(x, sections, knots, smoothness)

    # Fit the LASSO model
    lasso, β, β0, nz = fit_glmnet(basis, y, family; glmnet_args..., weights = weights)

    # Extract the sections and knots representing basis functions selected by LASSO
    sections = sections[nz]
    knots = knots[nz]

    return HALParameters(sections, knots, β, β0, smoothness), lasso
end

# If no weights are provided, assume equal weights
fit_random_hal(X, y, family, smoothness::Int, nfeatures::Union{Int, Nothing}, sampler_params::NamedTuple, weights::Nothing; glmnet_args...) = fit_random_hal(X, y, family, smoothness, nfeatures, sampler_params::NamedTuple, ones(nrow(X)); glmnet_args...)

# If no nfeatures are provided, call function assuming the asymptotically optimal number of basis functions
fit_random_hal(X, y, family, smoothness::Int, nfeatures::Nothing, sampler_params::NamedTuple, weights::AbstractVector{<:Real}; glmnet_args...) = fit_random_hal(X, y, family, smoothness, Int(round(length(y) * log(length(y)))), sampler_params, weights; glmnet_args...)

# Get predictions from a set of fitted HAL parameters and new data
function predict_hal(params::HALParameters, Xnew)
    x = Tables.Columns(Xnew)
    if(length(params.β) > 0)
        basis = ha_basis_matrix(x, params.sections, params.knots, params.smoothness)
        return (basis * params.β) .+ params.β0
    else
        return params.β0 .* ones(nrow(x))
    end
end