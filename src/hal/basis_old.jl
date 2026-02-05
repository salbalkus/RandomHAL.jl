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

function ha_basis_matrix(X::Tables.Columns, smoothness::Int; interaction_limit = nothing)
    # Get the type of each column in the data
    coltypes = Tables.schema(X).types

    # Set the highest order of interaction to the maximum if not specified
    if isnothing(interaction_limit)
        interaction_limit = length(X)
    end
    
    main_terms = (smoothness == 0) ? 
        [basis_function(X[i]) for i in 1:length(X)] :
        [coltypes[i] == Bool ? Matrix{Float64}(basis_function(X[i])) : basis_function(X[i], smoothness) for i in 1:length(X)]
    
    main_terms_and_interactions, all_sections = all_interactions(main_terms, interaction_limit)
    term_lengths = size.(main_terms_and_interactions, 2)

    return reduce(hcat, main_terms_and_interactions), all_sections, term_lengths
end

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








