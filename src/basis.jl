basis_function(Xcol1::AbstractVector{<:Real}, Xcol2::AbstractVector{<:Real}) = Xcol1 .>= transpose(Xcol2)
basis_function(Xcol1::AbstractVector{<:Real}, Xcol2::AbstractVector{<:Real}, smoothness::Int) = basis_function(Xcol1, Xcol2) .* (Xcol1 .- transpose(Xcol2)).^smoothness ./ factorial(smoothness)

function remove_duplicates(X1, X2, terms, coltypes, sections)
    output = Vector{AbstractMatrix}(undef, length(terms))
    # Check which columns are binary and select only the columns from above using the interaction
    # If all are binary, replace with the interaction
    for (i, section) in enumerate(sections)
        bools = [c == Bool for c in coltypes[section]]

        # If there are Bool columns, then we need to cull duplicate columns
        if any(bools)
            # In the case where every column is Bool, we only need to keep the single interaction between them
            if all(bools)
                binary_interaction = reduce(.*, X1[i] for i in section[bools])
                output[i] = reshape(binary_interaction, nrow(X1), 1)
            # If some are not Bool, the basis block will contain duplicates from lower orders, 
            # except in columns where all of the Bool columns interact
            else
                binary_interaction = reduce(.*, X2[i] for i in section[bools])
                output[i] = terms[i][:, binary_interaction]
            end
        # If the section doesn't contain any Bool, no duplicate columns need be culled
        else
            output[i] = terms[i]
        end
    end
    return output
end

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

function ha_basis_matrix(X1::Tables.Columns, X2::Tables.Columns, smoothness::Int; interaction_limit = nothing)
    # Get the type of each column in the data
    coltypes = Tables.schema(X1).types

    # Table checking
    if coltypes != Tables.schema(X2).types
        error("Tables X1 and X2 must have same number of columns and same column types")
    end

    # Set the highest order of interaction to the maximum if not specified
    if isnothing(interaction_limit)
        interaction_limit = length(X1)
    end
    
    main_terms = (smoothness == 0) ? 
        [basis_function(X1[i], X2[i]) for i in 1:length(X1)] :
        [coltypes[i] == Bool ? Matrix{Float64}(basis_function(X1[i], X2[i])) : basis_function(X1[i], X2[i], smoothness) for i in 1:length(X1)]
    
    main_terms_and_interactions, all_sections = all_interactions(main_terms, interaction_limit)
    main_terms_and_interactions = remove_duplicates(X1, X2, main_terms_and_interactions, coltypes, all_sections)
    term_lengths = size.(main_terms_and_interactions, 2)

    return reduce(hcat, main_terms_and_interactions), all_sections, term_lengths
end

ha_basis_matrix(X::Tables.Columns, smoothness::Int; interaction_limit = nothing) = ha_basis_matrix(X, X, smoothness; interaction_limit = interaction_limit)


function ha_basis_matrix(X::Tables.Columns, sections, knots, smoothness::Int)
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
        # TODO: Only smoothing non-binary variables using the power function might not be the most efficient way to do it
        X_basis = X_knots_ineq .* (((X_sections .- X_knots) .^ smoothness) ./ coef).^which_to_smooth
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








