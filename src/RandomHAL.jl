module RandomHAL

    #using Random
    #using StatsBase
    #using Combinatorics
    #using LinearAlgebra
    #using GLMNet
    #using MLJBase
    #using DataAPI
    #using Tables

    #using Distributions
    #using SparseArrays

    #import LogExpFunctions: logistic


    using Tables
    import Combinatorics: powerset
    import SpecialFunctions: logfactorial
    import DataAPI: ncol, nrow

    include("basis.jl")
    include("parameters.jl")
    #include("hal.jl")
    #include("randomhal.jl")

    export ha_basis_matrix, fit_glmnet, get_sections_and_knots, random_sections_and_knots
    #export HALRegressor, RandomHALRegressor, HALBinaryClassifier, RandomHALBinaryClassifier
    #export fit, predict

end