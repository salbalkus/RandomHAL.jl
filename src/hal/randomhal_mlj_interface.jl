const RHAL_DEFAULT_NLAMBDA = 100
const RHAL_DEFAULT_NFOLDS = 5
const RHAL_DEFAULT_NONZEROS = 0

### Continuous Data ###
mutable struct RandomHALRegressor <: MMI.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    min_nonzeros::Union{Int, Nothing}
    sampler_params::NamedTuple
end

RandomHALRegressor() = RandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALRegressor(nfeatures) = RandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALRegressor(smoothness, nfeatures) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALRegressor(smoothness, nfeatures, min_nonzeros) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, min_nonzeros, NamedTuple())
RandomHALRegressor(smoothness, nfeatures, min_nonzeros, sampler_params) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, min_nonzeros, sampler_params)

function MLJBase.fit(model::RandomHALRegressor, verbosity, X, y, w = nothing)
    n = length(y)
    alpha = 1.0# - (1.0/sqrt(n))
    params, lasso = fit_random_hal(X, y, Normal(), model.smoothness, model.nfeatures, model.min_nonzeros, model.sampler_params, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds, alpha = alpha)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct RandomHALBinaryClassifier <: MMI.Probabilistic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    min_nonzeros::Union{Int, Nothing}
    sampler_params::NamedTuple
end

RandomHALBinaryClassifier() = RandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALBinaryClassifier(nfeatures) = RandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALBinaryClassifier(smoothness, nfeatures) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, RHAL_DEFAULT_NONZEROS, NamedTuple())
RandomHALBinaryClassifier(smoothness, nfeatures, min_nonzeros) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, min_nonzeros, NamedTuple())
RandomHALBinaryClassifier(smoothness, nfeatures, min_nonzeros, sampler_params) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, min_nonzeros, sampler_params)

function MLJBase.fit(model::RandomHALBinaryClassifier, verbosity, X, y::Array{Bool, 1}, w = nothing)
    n = length(y)
    alpha = 1.0# - (1.0/sqrt(n))
    params, lasso = fit_random_hal(X, [.!(y) y], Binomial(), model.smoothness, model.nfeatures, model.min_nonzeros, model.sampler_params, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds, alpha = alpha)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))
