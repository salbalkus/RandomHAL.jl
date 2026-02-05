module RandomHAL
    import Combinatorics: powerset
    import DataAPI: ncol, nrow
    import LogExpFunctions: logistic
    import Base: *, getindex, size, transpose

    using StatsBase
    using Tables
    using Distributions
    using GLMNet
    using MLJBase
    using InvertedIndices
    using LinearAlgebra

    ###################################################
    ### Code to implement classical HAL, RandomHAL, ###
    ### and RieszHAL without fast nested scheme     ###
    ###################################################

    include("hal/basis.jl")
    export ha_basis_matrix

    include("hal/sections_and_knots.jl")
    export fit_glmnet, get_sections_and_knots, random_sections_and_knots

    include("hal/fit_and_predict.jl")
    export HALParameters, fit_hal, fit_random_hal, predict_hal

    include("hal/hal_mlj_interface.jl")
    export HALRegressor, HALBinaryClassifier

    include("hal/randomhal_mlj_interface.jl")
    export RandomHALRegressor, RandomHALBinaryClassifier

    include("hal/riesz.jl")
    export coord_descent, cross_coord_descent, predict_rieszhal
    
    include("hal/rieszhal_mlj_interface.jl")
    export HALRiesz, RandomHALRiesz
    export fit, predict

    #############################################
    ### Code to implement RandomHAL with fast ###
    ### nested matrix multiplication scheme   ###
    #############################################

    # New fast basis stuff
    include("fast_basis.jl")
    export NestedIndicators, NestedIndicatorBlocks, NestedMatrix, NestedMatrixBlocks, transpose, squares, *, mul, mul!

    include("fast_coord_descent2.jl")
    export coord_descent

    include("fast_fit_randomhal.jl")
    export fast_fit_cv_randomhal, predict_randomhal

end