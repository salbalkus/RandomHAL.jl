const RHAL_DEFAULT_NLAMBDA = 100
const RHAL_DEFAULT_NFOLDS = 5

### Continuous Data ###
mutable struct RandomHALRegressor <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    sample_balance::Union{Float64, Nothing}
end

RandomHALRegressor() = RandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, nothing)
RandomHALRegressor(nfeatures) = RandomHALRegressor(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, nothing)
RandomHALRegressor(smoothness, nfeatures) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, nothing)
RandomHALRegressor(smoothness, nfeatures, sample_balance) = RandomHALRegressor(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, sample_balance)

function MLJBase.fit(model::RandomHALRegressor, verbosity, X, y, w = nothing)
    params, lasso = fit_random_hal(X, y, Normal(), model.smoothness, model.nfeatures, model.sample_balance, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = (params = params,)
    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALRegressor, fitresult, Xnew) = predict_hal(fitresult.params, Xnew)

### Binary Data ###
mutable struct RandomHALBinaryClassifier <: MLJBase.Probabilistic
    smoothness::Int
    nlambda::Int
    nfolds::Int
    nfeatures::Union{Int, Nothing}
    sample_balance::Union{Float64, Nothing}
end

RandomHALBinaryClassifier() = RandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nothing, nothing)
RandomHALBinaryClassifier(nfeatures) = RandomHALBinaryClassifier(0, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, nothing)
RandomHALBinaryClassifier(smoothness, nfeatures) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, nothing)
RandomHALBinaryClassifier(smoothness, nfeatures, sample_balance) = RandomHALBinaryClassifier(smoothness, RHAL_DEFAULT_NLAMBDA, RHAL_DEFAULT_NFOLDS, nfeatures, sample_balance)

function MLJBase.fit(model::RandomHALBinaryClassifier, verbosity, X, y::Array{Bool, 1}, w = nothing)
    params, lasso = fit_random_hal(X, [.!(y) y], Binomial(), model.smoothness, model.nfeatures, model.sample_balance, w; standardize = true, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = (params = params,)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALBinaryClassifier, fitresult, Xnew) = logistic.(predict_hal(fitresult.params, Xnew))
