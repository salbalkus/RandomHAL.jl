module RandomHAL

    using Random
    using Distributions
    using StatsBase
    using SparseArrays
    using Combinatorics
    using LinearAlgebra
    using GLMNet
    using MLJBase
    using DataAPI
    using Tables
    import LogExpFunctions: logistic

    include("basis.jl")
    include("hal.jl")
    include("randomhal.jl")

    export HALRegressor, RandomHALRegressor, HALBinaryClassifier, RandomHALBinaryClassifier
    export fit, predict

end