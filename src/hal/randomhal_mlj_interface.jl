const RHAL_DEFAULT_SMOOTHNESS = 0
const RHAL_DEFAULT_NFEATURES = nothing
const RHAL_DEFAULT_SAMPLER = NamedTuple()
const RHAL_DEFAULT_KWARGS = (standardize = false, nlambda = 100, nfolds = 5)

### Continuous Data ###
mutable struct RandomHALRegressor <: MMI.Deterministic
    smoothness::Int
    nfeatures::Union{Int, Nothing}
    sampler_params::NamedTuple
    glmnet_kwargs::NamedTuple
end

RandomHALRegressor() = RandomHALRegressor(RHAL_DEFAULT_SMOOTHNESS, RHAL_DEFAULT_NFEATURES, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALRegressor(smoothness) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NFEATURES, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALRegressor(smoothness, nfeatures) = RandomHALRegressor(smoothness, nfeatures, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALRegressor(smoothness, nfeatures, sampler_params) = RandomHALRegressor(smoothness, nfeatures, sampler_params, RHAL_DEFAULT_KWARGS)

function MLJBase.fit(model::RandomHALRegressor, verbosity, X, y, w = nothing)
    n = length(y)
    params, lasso = fit_random_hal(X, y, Normal(), model.smoothness, model.nfeatures, model.sampler_params, w; model.glmnet_kwargs...)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct RandomHALBinaryClassifier <: MMI.Probabilistic
    smoothness::Int
    nfeatures::Union{Int, Nothing}
    sampler_params::NamedTuple
    glmnet_kwargs::NamedTuple
end

RandomHALBinaryClassifier() = RandomHALBinaryClassifier(RHAL_DEFAULT_SMOOTHNESS, RHAL_DEFAULT_NFEATURES, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALBinaryClassifier(smoothness) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NFEATURES, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALBinaryClassifier(smoothness, nfeatures) = RandomHALBinaryClassifier(smoothness, nfeatures, RHAL_DEFAULT_SAMPLER, RHAL_DEFAULT_KWARGS)
RandomHALBinaryClassifier(smoothness, nfeatures, sampler_params) = RandomHALBinaryClassifier(smoothness, nfeatures, sampler_params, RHAL_DEFAULT_KWARGS)

function MLJBase.fit(model::RandomHALBinaryClassifier, verbosity, X, y::Array{Bool, 1}, w = nothing)
    n = length(y)
    params, lasso = fit_random_hal(X, [.!(y) y], Binomial(), model.smoothness, model.nfeatures, model.sampler_params, w; model.glmnet_kwargs...)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))
