const HALRIESZ_DEFAULT_NLAMBDA = 100
const HALRIESZ_DEFAULT_NFOLDS = 5
const HALRIESZ_MAX_ITERS = 1000
const HALRIESZ_LAMBDA_REDUCTION = 0.001
const HALRIESZ_TOL = 0.0

### Continuous Data ###
mutable struct HALRiesz <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    max_iters::Int
    min_λ_ε::Float64
    tol::Float64
end

HALRiesz() = HALRiesz(0, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)
HALRiesz(smoothness) = HALRiesz(smoothness, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS, HALRIESZ_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)
HALRiesz(smoothness, nlambda, nfolds) = HALRiesz(smoothness, nlambda, nfolds, HALRIESZ_MAX_ITERS, HALRIESZ_LAMBDA_REDUCTION, HALRIESZ_TOL)


function MLJBase.fit(model::HALRiesz, verbosity, X, X_shift, w = nothing)
    
    basis, all_sections, term_lengths = ha_basis_matrix(X, model.smoothness)
    basis_shift, all_sections_shift, term_lengths_shift = ha_basis_matrix(X_shift, X, model.smoothness)

    β, β0, total_λ_range = cross_coord_descent(basis, basis_shift, λ = nothing, α = 1.0, min_λ_ε = model.min_λ_ε, λ_grid_length = model.nlambda, max_iters = model.max_iters, tol = model.tol)

    fitresult = (β = β, β0 = β0)
    cache = nothing
    report = (total_λ_range=total_λ_range,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALRiesz, fitresult, Χ) = (Tables.matrix(Χ) * fitresult.β) .+ β0