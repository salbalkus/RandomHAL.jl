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

struct HALBasis
    smoothness::Int
    sections::Vector{Vector{Int}}
    knots::Vector{Vector{Float64}}
end

# A collection of functions to coerce a column of data into a basis function
basis_function(Xcol::Union{Vector{Bool}, BitVector}) = Xcol
basis_function(Xcol::Union{Vector{Bool}, BitVector}, smoothness::Int) = Xcol
basis_function(Xcol::Vector{<:Real}) = (transpose(Xcol) .>= Xcol)
basis_function(Xcol::Vector{<:Real}, smoothness::Int) = basis_function(Xcol) .* (transpose(Xcol) .- Xcol).^smoothness ./ factorial(smoothness)

function ha_basis_matrix(X::Tables.Columns, smoothness::Int)
    # Generate basis without interactions
    main_terms = (smoothness == 0) ? 
        map(x -> basis_function(x), X) :
        map(x -> basis_function(x, smoothness), X)

    if length(X) == 1
        # Glue all the main terms into one matrix
        return reduce(hcat, main_terms)
    else
        # Generate all possible interaction terms
        interactions = [reduce(.*, main_term) for main_term in powerset(main_terms,2)]
        # Glue main terms and interactions into one matrix
        main_and_interactions = vcat(main_terms, interactions)
        term_lengths = size.(main_and_interactions, 2)
        return reduce(hcat, main_and_interactions), term_lengths
    end
end

basis_function(X::Matrix, section, knot, smoothness) = prod((view(X, :, section) .>= transpose(knot)) .* (view(X, :, section) .- transpose(knot)).^smoothness ./ factorial(smoothness), dims = 2)

# Basis matrix for 0-order HAL
function ha_basis_matrix(X::Tables.Columns, sections, knots)
    Xmat = Tables.matrix(X)
    Xe = view(Xmat, :, reduce(vcat, sections)) .>= transpose(reduce(vcat, knots))
    reduce(hcat, [prod(view(Xe, :, section), dims = 2) for section in sections])
end

# Basis matrix for higher-order HAL
function ha_basis_matrix(X::Tables.Columns, sections, knots, smoothness)
    if smoothness == 0
        return ha_basis_matrix(X, sections, knots)
    else
        Xmat = Tables.matrix(X)
        Xe = view(Xmat, :, reduce(vcat, sections))
        Xe = Xe .- transpose(reduce(vcat, knots))
        Xe[Xe .< 0.0] .= 0
        Xe = coef .* Xe .^ smoothness
        return reduce(hcat, [prod(view(Xe, :, section), dims = 2) for section in sections])
    end
end

