const HAL_DEFAULT_NLAMBDA = 100
const HAL_DEFAULT_NFOLDS = 5

### Continuous Data ###
mutable struct HALRegressor <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
end

HALRegressor() = HALRegressor(0, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)
HALRegressor(smoothness) = HALRegressor(smoothness, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)

function MLJBase.fit(model::HALRegressor, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, y, Normal(), model.smoothness, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct HALBinaryClassifier <: MLJBase.Probabilistic
    smoothness::Int
    nlambda::Int
    nfolds::Int
end

HALBinaryClassifier() = HALBinaryClassifier(0, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)
HALBinaryClassifier(smoothness) = HALBinaryClassifier(smoothness, HAL_DEFAULT_NLAMBDA, HAL_DEFAULT_NFOLDS)

function MLJBase.fit(model::HALBinaryClassifier, verbosity, X, y, w = nothing)
    params, lasso = fit_hal(X, [.!(y) y], Binomial(), model.smoothness, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = (params = params,)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::HALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))