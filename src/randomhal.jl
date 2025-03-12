# Continuous Data

mutable struct RandomHALRegressor <: MLJBase.Deterministic
    smoothness::Int
    nfeatures::Int
    nlambda::Int
    nfolds::Int
end

RandomHALRegressor(nfeatures) = RandomHALRegressor(0, nfeatures, 100, 5)
RandomHALRegressor(smoothness, nfeatures) = RandomHALRegressor(smoothness, nfeatures, 100, 5)

# TODO: Technically sampling is not performed with replacement here like it should be
function MLJBase.fit(model::RandomHALRegressor, verbosity, X, y, w = nothing)
    # Construct matrix representing the full HAL matrix on the data
    x = MLJBase.matrix(X)

    # Construct random highly adaptive basis
    sections, knots = random_sections_and_knots(x, model)
    basis = highly_adaptive_basis(x, sections, knots, model.smoothness)

    # Set up default parameters
    if isnothing(w)
        w = ones(size(x, 1))
    end

    # Fit the LASSO model
    lasso = glmnetcv(basis, y, Normal(); standardize = false, weights = w, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = highly_adaptive_parameters(sections, knots, lasso)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

function MLJBase.predict(model::RandomHALRegressor, fitresult, Xnew) 
    x = MLJBase.matrix(Xnew) 
    if(length(fitresult.sections) > 0)
        basis = highly_adaptive_basis(x, fitresult.sections, fitresult.knots, model.smoothness)
        return (basis * fitresult.β) .+ fitresult.β0
    else
        return fitresult.β0 .* ones(size(x)[1])
    end
end

# Binary Data

mutable struct RandomHALBinaryClassifier<: MLJBase.Probabilistic
    smoothness::Int
    nfeatures::Int
    nlambda::Int
    nfolds::Int
end

RandomHALBinaryClassifier(nfeatures) = RandomHALRegressor(0, nfeatures, 100, 5)
RandomHALBinaryClassifier(smoothness, nfeatures) = RandomHALRegressor(smoothness, nfeatures, 100, 5)

function MLJBase.fit(model::RandomHALBinaryClassifier, verbosity, X, y::Array{Bool, 1}, w = nothing)
    # Construct matrix representing the full HAL matrix on the data
    x = MLJBase.matrix(X)

    # Construct random highly adaptive basis
    sections, knots = random_sections_and_knots(x, model)
    basis = highly_adaptive_basis(x, sections, knots, model.smoothness)

    # Set up default parameters
    if isnothing(w)
        w = ones(size(x, 1))
    end

    # Fit the LASSO model
    lasso = glmnetcv(basis, [.!(y) y], Binomial(); standardize = false, weights = w, nlambda = model.nlambda, nfolds = model.nfolds)
    fitresult = highly_adaptive_parameters(sections, knots, lasso)

    cache = nothing
    report = (lasso=lasso,)
    return fitresult, cache, report
end

function MLJBase.predict(model::RandomHALBinaryClassifier, fitresult, Xnew) 
    x = MLJBase.matrix(Xnew) 
    if(length(fitresult.sections) > 0)
        basis = highly_adaptive_basis(x, fitresult.sections, fitresult.knots, model.smoothness)
        return logistic.((basis * fitresult.β) .+ fitresult.β0)
    else
        return logistic.(fitresult.β0) .* ones(size(x)[1])
    end
end