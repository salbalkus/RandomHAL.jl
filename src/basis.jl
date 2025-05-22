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

# A collection of functions to coerce a column of data into a basis function
basis_function(Xcol::AbstractVector{<:Real}) = Xcol .>= transpose(Xcol)
basis_function(Xcol::AbstractVector{<:Real}, smoothness::Int) = basis_function(Xcol) .* (Xcol .- transpose(Xcol)).^smoothness ./ factorial(smoothness)

### DIFFERENCE VERSION ###
basis_complement(Xcol::AbstractVector{<:Real}) = Xcol .< transpose(Xcol)
basis_complement(Xcol::AbstractVector{<:Real}, smoothness::Int) = basis_complement(Xcol) .* (Xcol .- transpose(Xcol)).^smoothness ./ factorial(smoothness)
###########################

clean_binary(X, main_terms) = [t == Bool ? Tables.getcolumn(X, i) : main_terms[i] for (i, t) in enumerate(Tables.schema(X).types)]

# Constructing basis matrix on training data directly
function ha_basis_matrix(X::Tables.Columns, smoothness::Int; diff = false)
    # Generate basis without interactions
    main_terms = (smoothness == 0) ? 
        map(x -> basis_function(x), X) :
        map(x -> basis_function(x, smoothness), X)
    # Generate all possible interaction terms
    interactions = ncol(X) == 1 ? [] : [reduce(.*, main_term) for main_term in powerset(main_terms,2)]
    
    ### DIFFERENCE VERSION ###
    # Subtract off the complements if we want to use the "difference" version of HAL
    if diff
        main_terms_neg = (smoothness == 0) ? 
            map(x -> basis_complement(x), X) :
            map(x -> basis_complement(x, smoothness), X)
        interactions_neg = ncol(X) == 1 ? [] : [reduce(.*, main_term) for main_term in powerset(main_terms_neg,2)]

        main_terms = main_terms .- main_terms_neg
        interactions = interactions .- interactions_neg
    end
    ##########################

    # Reduce binary basis functions to eliminate duplicates
    # We need to do this after generating interactions because the full "block" is needed
    # to compute interactions even for binary variables, but it contains many duplicate terms
    main_terms = clean_binary(X, main_terms)
    
    # Glue main terms and interactions into one matrix
    main_and_interactions = vcat(main_terms, interactions)    
    
    # Compute how many basis functions for each "term" in the linear model
    term_lengths = size.(main_and_interactions, 2)
    
    return reduce(hcat, main_and_interactions), term_lengths
end

# Basis matrix for 0-order HAL
function ha_basis_matrix_0(X::Tables.Columns, sections, knots)
    Xmat = Tables.matrix(X)
    Xe = view(Xmat, :, reduce(vcat, sections)) .>= transpose(reduce(vcat, knots))
    output = BitMatrix(undef, nrow(X), length(sections))
    # Iterate through each basis function
    i = 1
    for (j, section) in enumerate(sections)
        # Select indices of variables placed next to each other in Xe as a section
        indices = i:(i + length(section) - 1) 

        # Compute the product of the selected variables as the interaction
        # (if a main term, "indices" will inclue only one index)
        output[:, j] = prod(view(Xe, :, indices), dims = 2)
        i = i + length(section)
    end
    return output
end

# Basis matrix for higher-order HAL
function ha_basis_matrix_s(X::Tables.Columns, sections, knots, smoothness)
    if smoothness == 0
        return ha_basis_matrix_0(X, sections, knots)
    else
        Xmat = Tables.matrix(X)
        Xe = view(Xmat, :, reduce(vcat, sections))
        Xe_ind = view(Xmat, :, reduce(vcat, sections)) .>= transpose(reduce(vcat, knots))
        Xe = Xe_ind .* ((Xe .- transpose(reduce(vcat, knots))) .^ smoothness) ./ factorial(smoothness)
        
        # Iterate through each basis function
        i = 1
        output = Matrix{Float64}(undef, nrow(X), length(sections))
        for (j, section) in enumerate(sections)
            # Select indices of variables placed next to each other in Xe as a section
            indices = i:(i + length(section) - 1)
            # Compute the product of the selected variables as the interaction
            # (if a main term, "indices" will inclue only one index)
            if (length(indices) == 1) && (Tables.schema(X).types[section][1] == Bool)
                output[:, j] = Tables.getcolumn(X, section[1])
            else
                output[:, j] = prod(view(Xe, :, indices), dims = 2)
            end
            i = i + length(section)
        end
        return output
    end
end

# Difference matrix for 0-order HAL
function ha_difference_matrix_0(X::Tables.Columns, sections, knots)
    Xmat = Tables.matrix(X)
    Xe = view(Xmat, :, reduce(vcat, sections)) .>= transpose(reduce(vcat, knots))
    Xe_neg = view(Xmat, :, reduce(vcat, sections)) .< transpose(reduce(vcat, knots))
    
    i = 1
    # TODO: Update this to be some "signed bit" type
    output = Matrix{Int8}(undef, nrow(X), length(sections))
    # Iterate through each basis function
    for (j, section) in enumerate(sections)
        # Select indices of variables placed next to each other in Xe as a section
        indices = i:(i + length(section) - 1) 

        # Compute the product of the selected variables as the interaction
        # (if a main term, "indices" will inclue only one index)
        output[:, j] = prod(view(Xe, :, indices), dims = 2) .- prod(view(Xe_neg, :, indices), dims = 2)
        # Subtract off the complement
        i = i + length(section)
    end
    return output
end

# Difference matrix for higher-order HAL
function ha_difference_matrix_s(X::Tables.Columns, sections, knots, smoothness)
    if smoothness == 0
        return ha_difference_matrix_0(X, sections, knots)
    else
        Xmat = Tables.matrix(X)
        Xe = view(Xmat, :, reduce(vcat, sections))

        # Compute the positive (lower) function for each section
        Xe_pos_ind = view(Xmat, :, reduce(vcat, sections)) .>= transpose(reduce(vcat, knots))
        Xe_pos = Xe_pos_ind .* ((Xe .- transpose(reduce(vcat, knots))) .^ smoothness) ./ factorial(smoothness)

        # Compute the negative (upper) function for each section
        Xe_neg_ind = view(Xmat, :, reduce(vcat, sections)) .< transpose(reduce(vcat, knots))
        Xe_neg = Xe_neg_ind .* ((Xe .- transpose(reduce(vcat, knots))) .^ smoothness) ./ factorial(smoothness)

        # Iterate through each basis function
        i = 1
        output = Matrix{Float64}(undef, nrow(X), length(sections))
        for (j, section) in enumerate(sections)
            # Select indices of variables placed next to each other in Xe as a section
            indices = i:(i + length(section) - 1)
            # Compute the product of the selected variables as the interaction
            # (if a main term, "indices" will inclue only one index)
            if (length(indices) == 1) && (Tables.schema(X).types[section][1] == Bool)
                output[:, j] = Tables.getcolumn(X, section[1])
            else
                output[:, j] = prod(view(Xe_pos, :, indices), dims = 2) .- prod(view(Xe_neg, :, indices), dims = 2)
            end
            i = i + length(section)
        end
        return output
    end
end

function ha_basis_matrix(X::Tables.Columns, sections, knots, smoothness; diff = false)
    if diff
        return ha_difference_matrix_s(X, sections, knots, smoothness)
    else
        return ha_basis_matrix_s(X, sections, knots, smoothness)
    end 
end

