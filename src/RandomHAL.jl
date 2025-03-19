module RandomHAL
    import Combinatorics: powerset
    import DataAPI: ncol, nrow
    import LogExpFunctions: logistic

    using Tables
    using Distributions
    using GLMNet
    using MLJBase

    include("basis.jl")
    export ha_basis_matrix

    include("sections_and_knots.jl")
    export fit_glmnet, get_sections_and_knots, random_sections_and_knots

    include("fit_and_predict.jl")
    export HALParameters, fit_hal, fit_random_hal, predict_hal

    include("hal_mlj_interface.jl")
    export HALRegressor, HALBinaryClassifier

    include("randomhal_mlj_interface.jl")
    export RandomHALRegressor, RandomHALBinaryClassifier

    export fit, predict
end