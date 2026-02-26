

### Continuous Data ###
@mlj_model mutable struct RandomHALRegressor <: MMI.Deterministic
    smoothness::Int64 = 0::(_ >= 0)
    nlambda::Int64 = 100::(_ > 0)
    nfolds::Int64 = 5::(_ > 0)
    outer_max_iters::Int64 = 1000::(_ > 0)
    inner_max_iters::Int64 = 1000::(_ > 0)
    λ_grid_length::Int64 = 100::(_ > 0)
    min_λ_ε::Float64 = 1e-3::(_ > 0 && _ < 1.0)
    tol::Float64 = 1e-7::(_ > 0)
    α::Float64 = 1.0::(_ >= 0.0 && _ <= 1.0)
end

function MLJBase.fit(model::RandomHALRegressor, verbosity, X, y)
    sections = collect(combinations(1:DataAPI.ncol(X)))[2:end]
    Xm = Tables.matrix(X)

    params = fast_fit_cv_randomhal(sections, Xm, y; 
                K = model.nfolds, outer_max_iters = model.outer_max_iters, 
                inner_max_iters = model.inner_max_iters, λ = nothing, 
                λ_grid_length = model.λ_grid_length, min_λ_ε = model.min_λ_ε, 
                tol = model.tol, α = model.α)
    
    fitresult = (params = params,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MLJBase.predict(model::RandomHALRegressor, fitresult, Xnew) = predict_randomhal(fitresult.params, Tables.matrix(Xnew))