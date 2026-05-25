const HAL_DEFAULT_MIN_NONZERO = 0
const HAL_DEFAULT_SMOOTHNESS = 0
const HAL_DEFAULT_KWARGS = (standardize = false, nlambda = 100, nfolds = 5)

### Continuous Data ###
mutable struct HALRegressor <: MMI.Deterministic
    smoothness::Int
    min_nonzero::Int
    glmnet_kwargs::NamedTuple
end

HALRegressor() = HALRegressor(HAL_DEFAULT_SMOOTHNESS, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_KWARGS)
HALRegressor(smoothness) = HALRegressor(smoothness, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_KWARGS)
HALRegressor(smoothness, min_nonzero) = HALRegressor(smoothness, min_nonzero, HAL_DEFAULT_KWARGS)


function MLJBase.fit(model::HALRegressor, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, y, Normal(), model.smoothness, model.min_nonzero, w; model.glmnet_kwargs...)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct HALBinaryClassifier <: MMI.Probabilistic
    smoothness::Int
    min_nonzero::Int
    glmnet_kwargs::NamedTuple
end

HALBinaryClassifier() = HALBinaryClassifier(HAL_DEFAULT_SMOOTHNESS, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_KWARGS)
HALBinaryClassifier(smoothness) = HALBinaryClassifier(smoothness, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_KWARGS)
HALBinaryClassifier(smoothness, min_nonzero) = HALBinaryClassifier(smoothness, min_nonzero, HAL_DEFAULT_KWARGS)

function MLJBase.fit(model::HALBinaryClassifier, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, [.!(y) y], Binomial(), model.smoothness, model.min_nonzero, w; model.glmnet_kwargs...)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))