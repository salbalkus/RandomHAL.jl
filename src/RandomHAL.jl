module RandomHAL
    import Combinatorics: powerset, combinations
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
    using MLJModelInterface
    const MMI = MLJModelInterface


    ############################################
    ### Code to implement classical HAL and  ###
    ### RandomHAL without fast nested scheme ###
    ############################################

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

    #############################################
    ### Code to implement RandomHAL with fast ###
    ### nested matrix multiplication scheme   ###
    #############################################

    # New fast basis stuff
    include("fast_hal/fast_basis.jl")
    export NestedIndicators, NestedIndicatorBlocks, NestedMatrix, NestedMatrixBlocks 
    export Basis, BasisBlocks, BasisMatrix, BasisMatrixBlocks
    export transpose, colmeans, squares, left_sum, left_squares, nonzero_count, *, mul, mul!

    include("fast_hal/fast_coord_descent.jl")
    export coord_descent

    include("fast_hal/fast_fit_randomhal.jl")
    export fast_fit_cv_randomhal, predict_randomhal

    include("fast_hal/fast_hal_mlj_interface.jl")
    export RandomHALRegressor, fit, predict

end