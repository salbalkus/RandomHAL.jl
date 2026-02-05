const RHAL_DEFAULT_NLAMBDA = 100
const RHAL_DEFAULT_NFOLDS = 5

### Continuous Data ###
mutable struct SlowRandomHALRegressor <: MMI.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    sampler_params::NamedTuple
end

SlowRandomHALRegressor() = SlowRandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, NamedTuple())
SlowRandomHALRegressor(nfeatures) = SlowRandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, NamedTuple())
SlowRandomHALRegressor(smoothness, nfeatures) = SlowRandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, NamedTuple())
SlowRandomHALRegressor(smoothness, nfeatures, sampler_params) = SlowRandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, sampler_params)

function MLJBase.fit(model::SlowRandomHALRegressor, verbosity, X, y, w = nothing)
    n = length(y)
    alpha = 1.0# - (1.0/sqrt(n))
    params, lasso = fit_random_hal(X, y, Normal(), model.smoothness, model.nfeatures, model.sampler_params, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds, alpha = alpha)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::SlowRandomHALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct SlowRandomHALBinaryClassifier <: MMI.Probabilistic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    sampler_params::NamedTuple
end

SlowRandomHALBinaryClassifier() = SlowRandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, NamedTuple())
SlowRandomHALBinaryClassifier(nfeatures) = SlowRandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, NamedTuple())
SlowRandomHALBinaryClassifier(smoothness, nfeatures) = SlowRandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, NamedTuple())
SlowRandomHALBinaryClassifier(smoothness, nfeatures, sample_balance) = SlowRandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, sampler_params)

function MLJBase.fit(model::SlowRandomHALBinaryClassifier, verbosity, X, y::Array{Bool, 1}, w = nothing)
    n = length(y)
    alpha = 1.0 - (1.0/sqrt(n))
    params, lasso = fit_random_hal(X, [.!(y) y], Binomial(), model.smoothness, model.nfeatures, model.sampler_params, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds, alpha = alpha)
    fitresult = (params = params,)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::SlowRandomHALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))
