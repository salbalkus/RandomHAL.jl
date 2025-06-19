basis_function(Xcol::AbstractVector{<:Real}) = Xcol .>= transpose(Xcol)
basis_function(Xcol::AbstractVector{<:Real}, smoothness::Int) = basis_function(Xcol) .* (Xcol .- transpose(Xcol)).^smoothness ./ factorial(smoothness)

function interact(x_next::T, x::Vector{T}, next_section::Int, sections::Vector{Vector{Int}}, limit::Int) where T <: AbstractMatrix
    # Store the interactions and their orders, so we know when to stop
    interactions = Vector{T}()
    interaction_sections = Vector{Vector{Int}}()

    # Iteratively multiply each main term with all of the previous terms
    for i in 1:length(x)
        if length(sections[i]) < limit
            x_next_i = x_next .* x[i]
            interaction_sections_i = vcat(sections[i], next_section)
            push!(interactions, x_next_i)
            push!(interaction_sections, interaction_sections_i)
        end
    end

    # Concatenate main terms and interactions
    x_output = vcat(x, [x_next], interactions)
    sections_output = vcat(sections, [[next_section]], interaction_sections)
    return x_output, sections_output
end

function all_interactions(main_terms::Vector{T}, limit::Int) where T <: AbstractMatrix
    combined_interactions_and_main = [main_terms[1]]
    sections = [[1]]
    for i in 2:length(main_terms)
        combined_interactions_and_main, sections = interact(main_terms[i], combined_interactions_and_main, i, sections, limit)
    end
    return combined_interactions_and_main, sections
end

function ha_basis_matrix(X::Tables.Columns, smoothness::Int; interaction_limit::Int = nothing)
    if isnothing(interaction_limit)
        interaction_limit = length(X)
    end
    
    coltypes = Tables.schema(X).types
    main_terms = (smoothness == 0) ? 
        [coltypes[i] == Bool ? reshape(X[i], :, 1) : basis_function(X[i]) for i in 1:length(X)] :
        [coltypes[i] == Bool ? reshape(X[i], :, 1) : basis_function(X[i], smoothness) for i in 1:length(X)]
    
    main_terms_and_interactions, _ = all_interactions(main_terms, interaction_limit)
    term_lengths = size.(main_terms_and_interactions, 2)

    return reduce(hcat, main_terms_and_interactions), term_lengths
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
function ha_basis_matrix(X::Tables.Columns, sections, knots, smoothness)
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
        return X_output
    end
end

function ha_basis_matrix(X::Tables.Columns, sections, knots, smoothness)
    coltypes = Tables.schema(X).types
    Xmat = Tables.matrix(X)

    # Construct a matrix of knots
    X_knots = transpose(reduce(vcat, knots))

    # Construct a matrix of sections
    X_sections = view(Xmat, :, reduce(vcat, sections))

    # Compute the areas where the spline is active (i.e., where the sections are greater than or equal to the knots)
    X_knots_ineq = X_sections .>= X_knots

    # Set up indicator to skip discrete variables that cannot be smoothed in higher-orders
    which_to_smooth = reshape(reduce(vcat, [coltypes[s] for s in section] for section in sections) .!= Bool, 1, :)

    # It's actually faster to compute every basis function in a single pass due to vectorization
    if smoothness > 0
        coef = factorial(smoothness)
        X_basis = X_knots_ineq .* (X_sections .- X_knots) .^ (smoothness .* which_to_smooth) ./ coef
        X_output = Matrix{Float64}(undef, nrow(X), length(sections))

    else
        X_basis = X_knots_ineq
        X_output = BitMatrix(undef, nrow(X), length(sections))
    end

    # Iterate through each basis function
    i = 1
    for (j, section) in enumerate(sections)
        # Select indices of variables placed next to each other as a section
        indices = i:(i + length(section) - 1) 

        # Compute the product of the selected variables as the interaction
        # (if a main term, "indices" will inclue only one index)
        X_output[:, j] = prod(view(X_basis, :, indices), dims = 2)
        i = i + length(section)
    end
    return X_output
end








