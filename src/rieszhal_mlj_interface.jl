const HALRIESZ_DEFAULT_NLAMBDA = 20
const HALRIESZ_DEFAULT_NFOLDS = 5
const HALRIESZ_OUTER_MAX_ITERS = 20
const HALRIESZ_INNER_MAX_ITERS = 50
const HALRIESZ_LAMBDA_REDUCTION = 0.001
const HALRIESZ_TOL = 0.0

mutable struct HALRiesz <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    outer_max_iters::Int
    inner_max_iters::Int
    min_λ_ε::Float64
    tol::Float64
end

HALRiesz() = HALRiesz(0, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)
HALRiesz(smoothness) = HALRiesz(smoothness, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)
HALRiesz(smoothness, nlambda) = HALRiesz(smoothness, nlambda, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)

function MLJBase.fit(model::HALRiesz, verbosity, X, X_shift, w = nothing)
    
    x = Tables.Columns(X)
    x_shift = Tables.Columns(X_shift)
    basis, all_sections, term_lengths = ha_basis_matrix(x, model.smoothness)
    basis_shift, all_sections_shift, term_lengths_shift = ha_basis_matrix(x_shift, x, model.smoothness)

    β, β0, λ_best = cross_coord_descent(basis, basis_shift; nfolds = model.nfolds, λ = nothing, min_λ_ε = model.min_λ_ε, λ_grid_length = model.nlambda, outer_max_iters = model.outer_max_iters, inner_max_iters = model.inner_max_iters, tol = model.tol, only_refit_best = true)
    nz = [i for i in 1:length(β) if β[i] != 0]
    sections, knots = get_sections_and_knots(X, nz, all_sections, term_lengths)

    fitresult = (params = HALParameters(sections, knots, β[nz], β0, model.smoothness), )
    cache = nothing
    report = (λ_best=λ_best,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALRiesz, fitresult, X) = predict_rieszhal(fitresult.params, X)

mutable struct RandomHALRiesz <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    outer_max_iters::Int
    inner_max_iters::Int
    min_λ_ε::Float64
    tol::Float64
    nfeatures::Union{Int, Nothing}
    sampler_params::NamedTuple
end

RandomHALRiesz() = RandomHALRiesz(0, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL, nothing, NamedTuple())
RandomHALRiesz(smoothness) = RandomHALRiesz(smoothness, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL, nothing, NamedTuple())
RandomHALRiesz(smoothness, nlambda) = RandomHALRiesz(smoothness, nlambda, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL, nothing, NamedTuple())
RandomHALRiesz(smoothness, nlambda, nfeatures, sampler_params) = RandomHALRiesz(smoothness, nlambda, HAL_DEFAULT_NFOLDS, HALRIESZ_OUTER_MAX_ITERS, HALRIESZ_INNER_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL, nfeatures, sampler_params)

function MLJBase.fit(model::RandomHALRiesz, verbosity, X, X_shift, w = nothing)
    
    x = Tables.Columns(X)
    x_shift = Tables.Columns(X_shift)

    if isnothing(model.nfeatures)
        n = nrow(x)
        nfeatures = Int(round(n * log(n)))
    else
        nfeatures = model.nfeatures
    end

    sections, knots = random_sections_and_knots(x, nfeatures; model.sampler_params...)
    basis = ha_basis_matrix(x, sections, knots, model.smoothness)
    basis_shift = ha_basis_matrix(x_shift, sections, knots, model.smoothness)

    β, β0, λ_best = cross_coord_descent(basis, basis_shift; nfolds = model.nfolds, λ = nothing, min_λ_ε = model.min_λ_ε, λ_grid_length = model.nlambda, outer_max_iters = model.outer_max_iters, inner_max_iters = model.inner_max_iters, tol = model.tol, only_refit_best = true)
    nz = [i for i in 1:length(β) if β[i] != 0]
    sections = sections[nz]
    knots = knots[nz]

    fitresult = (params = HALParameters(sections, knots, β[nz], β0, model.smoothness), )
    cache = nothing
    report = (λ_best=λ_best,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALRiesz, fitresult, X) = predict_rieszhal(fitresult.params, X)