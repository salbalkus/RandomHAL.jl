# Continuous Data
mutable struct HALRegressor <: MLJBase.Deterministic
    smoothness::Int
    nlambda::Int
    nfolds::Int
end

HALRegressor() = HALRegressor(0, 100, 5)
HALRegressor(smoothness) = HALRegressor(smoothness, 100, 5)

function MLJBase.fit(model::HALRegressor, verbosity, X, y, w = nothing)
    # Construct matrix representing the full HAL matrix on the data
    x = MLJBase.matrix(X)
    basis = highly_adaptive_basis(x, model.smoothness)

    # Set up default parameters
    if isnothing(w)
        w = ones(size(x, 1))
    end

    # Fit the LASSO model
    lasso = glmnetcv(basis, y, Normal(); standardize = false, weights = w, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = highly_adaptive_parameters(x, lasso)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

function MLJBase.predict(model::HALRegressor, fitresult, Xnew) 
    x = MLJBase.matrix(Xnew) 
    if(length(fitresult.sections) > 0)
        basis = highly_adaptive_basis(x, fitresult.sections, fitresult.knots, model.smoothness)
        return (basis * fitresult.β) .+ fitresult.β0
    else
        return fitresult.β0 .* ones(size(x)[1])
    end
end

# Binary Data

mutable struct HALBinaryClassifier <: MLJBase.Probabilistic
    smoothness::Int
    nlambda::Int
    nfolds::Int
end

HALBinaryClassifier() = HALBinaryClassifier(0, 100, 5)
HALBinaryClassifier(smoothness) = HALBinaryClassifier(smoothness, 100, 5)

function MLJBase.fit(model::HALBinaryClassifier, verbosity, X, y, w = nothing)
    # Construct matrix representing the full HAL matrix on the data
    x = MLJBase.matrix(X)
    basis = highly_adaptive_basis(x, model.smoothness)

    # Set up default parameters
    if isnothing(w)
        w = ones(size(x, 1))
    end

    # Fit the LASSO model
    lasso = glmnetcv(basis, [.!(y) y], Binomial(); standardize = false, weights = w, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = highly_adaptive_parameters(x, lasso)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

function MLJBase.predict(model::HALBinaryClassifier, fitresult, Xnew) 
    x = MLJBase.matrix(Xnew) 
    if(length(fitresult.sections) > 0)
        basis = highly_adaptive_basis(x, fitresult.sections, fitresult.knots, model.smoothness)
        return logistic.((basis * fitresult.β) .+ fitresult.β0)
    else
        return logistic.(fitresult.β0) .* ones(size(x)[1])
    end
end