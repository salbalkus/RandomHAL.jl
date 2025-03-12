# This module provides two strategies for defining a HAL basis. 

# The first is to initialize the matrix of basis function values from the data directly.
# This is more computationally efficient when *fitting* a full HAL model on training data,
# because storing knots and sections requires more memory and cannot exploit vectorized matrix
# operations by assuming the knot points are the same as the underlying data. 

# The second is to define an object that stores a collection of knots and sections. 
# This is necessary when *predicting* on new data using knot points selected from a training set. 
# Additionally, it is more computationally efficient for outputting predictions, since only the
# subset of basis functions with nonzero coefficients (selected by lasso) need to be evaluated.
# Furthermore, it allows RandomHAL to sample random subsets of possible knots and sections
# without needing to enumerate the entire basis matrix of the full HAL model first. 

"""
    highly_adaptive_basis(X::Matrix, smoothness::Int)

Generate a basis matrix for the Highly Adaptive Lasso from data `X` assuming each combination of existing data point and section constitutes a basis function.

# Arguments
- `X::Matrix`: A matrix of input data where each row represents an observation and each column represents a feature.
- `smoothness::Int`: An integer specifying the order of the spline fit over each section of the data.

# Returns
- A matrix where each column represents a basis function. Each basis function is computed by checking if the elements in the specified sections of `X` are greater than the corresponding knots.

"""
function highly_adaptive_basis(X::Matrix, smoothness::Int)
    n, d = size(X)

    # Generate basis without interactions
    #main_terms = reduce(hcat, (transpose(Xcol) .<= Xcol) .* (Xcol.^smoothness) for Xcol in eachcol(X))
    main_terms = reduce(hcat, (transpose(Xcol) .<= Xcol) for Xcol in eachcol(X))

    # Generate indices representing rank transform of each feature
    feature_indices = [(n*(j-1) + 1):(n*j) for j in 1:d]
    
    # Generate subsets of rank-transformed features representing interaction terms
    feature_subsets = Combinatorics.powerset(feature_indices, 2)
    
    if d > 1
        interactions = reduce(hcat, map(fs -> reduce(.*, map(x -> main_terms[:, x], fs)), feature_subsets))
        return hcat(main_terms, interactions)
    else
        return main_terms
    end
end

"""
    highly_adaptive_parameters(X::Matrix, lasso::GLMNetCrossValidation)

Extract the sections, knots, and nonzero coefficients selected by a Highly Adaptive Lasso fit using GLMNet.

# Arguments
- `X::Matrix`: A matrix of input data where each row represents an observation and each column represents a feature.
- `lasso::GLMNetCrossValidation`: A GLMNetCrossValidation object representing the LASSO fit.

# Returns
- `sections`: A collection of indices specifying which columns of `X` to consider for each basis function.
- `knots`: A collection of threshold values for each section. Each element in `knots` corresponds to a section and contains the threshold values for that section.
- `β`: A vector of coefficients corresponding to the nonzero coefficients selected by the LASSO fit.
- `β0`: The intercept term selected by the LASSO fit.

"""
function highly_adaptive_parameters(X::Matrix, lasso::GLMNetCrossValidation)

    # Get initial data dimensions
    n, d = size(X)

    # Extract selected nonzero coefficients from lasso fit
    best = argmin(vec(mean(reduce(hcat, lasso.losses[length.(lasso.losses) .> 0]), dims = 2)))
    β0 = lasso.path.a0[best]
    βvec = lasso.path.betas[:, best]
    nz = [i for i in 1:length(βvec) if βvec[i] != 0]
    β = βvec[nz]
    
    # Reverse engineer the unit-feature combinations selected by lasso
    unit_dom = (nz .% n) # observation to dominate
    unit_dom[unit_dom .== 0] .= n # fix indexing when first observation is chosen
    feature_dom = ((nz .- 1) .÷ n) .+ 1 # variable combination number to dominate
    feature_combinations = collect(Combinatorics.powerset(1:d, 1)) # generate powerset
    
    # Construct the vector of sections selected by LASSO
    sections = [feature_combinations[fd] for fd in feature_dom]  

    # Construct the set of knot-points for each section selected by LASSO
    knots = [X[unit_dom[i], sections[i]] for i in 1:length(unit_dom)]

    return (sections = sections, knots = knots, β = β, β0 = β0)
end

"""
    highly_adaptive_parameters(sections, knots, lasso::GLMNetCrossValidation)

Extract the sections, knots, and nonzero coefficients selected by a Highly Adaptive Lasso fit using GLMNet.

# Arguments
- `X::Matrix`: A matrix of input data where each row represents an observation and each column represents a feature.
- `lasso::GLMNetCrossValidation`: A GLMNetCrossValidation object representing the LASSO fit.

# Returns
- `sections`: A collection of indices specifying which columns of `X` to consider for each basis function.
- `knots`: A collection of threshold values for each section. Each element in `knots` corresponds to a section and contains the threshold values for that section.
- `β`: A vector of coefficients corresponding to the nonzero coefficients selected by the LASSO fit.
- `β0`: The intercept term selected by the LASSO fit.

"""
function highly_adaptive_parameters(sections, knots, lasso::GLMNetCrossValidation)
    # Extract selected nonzero coefficients from lasso fit
    best = argmin(vec(mean(reduce(hcat, lasso.losses[length.(lasso.losses) .> 0]), dims = 2)))
    β0 = lasso.path.a0[best]
    βvec = lasso.path.betas[:, best]
    nz = [i for i in 1:length(βvec) if βvec[i] != 0]

    β = βvec[nz]
    nz = [i for i in 1:length(βvec) if βvec[i] != 0]

    # Construct the vector of sections selected by LASSO
    sections = sections[nz]

    # Construct the set of knot-points for each section selected by LASSO
    knots = knots[nz]

    return (sections = sections, knots = knots, β = β, β0 = β0)
end

"""
    highly_adaptive_basis(X::Matrix, sections, knots, smoothness::Int)

Generate a basis matrix for the Highly Adaptive Lasso from data `X` using the specified section and knot parameters

# Arguments
- `X::Matrix`: A matrix of input data where each row represents an observation and each column represents a feature.
- `sections`: A collection of indices specifying which columns of `X` to consider for each basis function.
- `knots`: A collection of threshold values for each section. Each element in `knots` corresponds to a section and contains the threshold values for that section.
- `smoothness::Int`: An integer specifying the order of the spline fit over each section of the data.

# Returns
- A matrix where each column represents a basis function. Each basis function is computed by checking if the elements in the specified sections of `X` are greater than the corresponding knots.

"""
function highly_adaptive_basis(X::Matrix, sections, knots, smoothness::Int)
    # transpose to take advantage of Julia's column-major order
    Xt = LinearAlgebra.transpose(X) 

    # Compute indicator functions over each section
    return LinearAlgebra.transpose(reduce(vcat, [prod(view(Xt, sections[j], :) .>= knots[j], dims=1) for j in 1:length(knots)]))
end

function random_sections_and_knots(X::Matrix, model)
    n, d = size(X)

    # Select sections by first selecting a random section size according to binomial law,
    # then selecting random section indices uniformly. 
    section_dist = DiscreteNonParametric(1:d, binomial.(d, 1:d) ./ (2^d - 1))
    section_sizes = rand(section_dist, model.nfeatures)
    sections = map(k -> sample(1:d, k; replace = false), section_sizes)

    # Select random knots and construct random basis
    knot_dist = DiscreteUniform(1, n)
    knots = [view(X, rand(knot_dist), s) for s in sections]

    return sections, knots
end