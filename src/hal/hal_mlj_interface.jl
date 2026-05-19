const HAL_DEFAULT_NLAMBDA = 100
const HAL_DEFAULT_NFOLDS = 5
const HAL_DEFAULT_MIN_NONZERO = 0
const HAL_DEFAULT_SMOOTHNESS = 0

### Continuous Data ###
mutable struct HALRegressor <: MMI.Deterministic
    smoothness::Int
    min_nonzero::Int
    nlambda::Int
    nfolds::Int
end

HALRegressor() = HALRegressor(HAL_DEFAULT_SMOOTHNESS, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)
HALRegressor(smoothness) = HALRegressor(smoothness, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)
HALRegressor(smoothness, min_nonzero) = HALRegressor(smoothness, min_nonzero, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)


function MLJBase.fit(model::HALRegressor, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, y, Normal(), model.smoothness, model.min_nonzero, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
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
    nlambda::Int
    nfolds::Int
end

HALBinaryClassifier() = HALBinaryClassifier(HAL_DEFAULT_SMOOTHNESS, HAL_DEFAULT_MIN_NONZERO, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)
HALBinaryClassifier(smoothness, min_nonzero) = HALBinaryClassifier(smoothness, min_nonzero, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)

function MLJBase.fit(model::HALBinaryClassifier, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, [.!(y) y], Binomial(), model.smoothness, model.min_nonzero, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = (params = params,)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))