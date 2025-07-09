
# Helper function to extract the nonzero coefficients from a LASSO fit
function fit_glmnet(basis::AbstractMatrix, y::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, family; kwargs...)
    # Fit lasso model
    lasso = glmnetcv(basis, y, family; kwargs...)

    # Extract coefficients selected by LASSO and their indices in the basis
    best = argmin(vec(mean(reduce(hcat, lasso.losses[length.(lasso.losses) .> 0]), dims = 2)))
    β0 = lasso.path.a0[best]
    βvec = lasso.path.betas[:, best]
    nz = [i for i in 1:length(βvec) if βvec[i] != 0]
    β = βvec[nz]
    return lasso, β, β0, nz
end

# Helper function to extract the sections and knots selected by a LASSO fit on the matrix version of the basis
function get_sections_and_knots(X, nonzero_indices, all_possible_sections, term_lengths)
    coltypes = Tables.schema(X).types # Get type of each variable
    # List all possible interactions of variables
    sections = Vector{Vector{Int}}(undef, length(nonzero_indices))
    knots = Vector{Vector{Real}}(undef, length(nonzero_indices))

    # Set up iteration trackers
    cur_basis_index = 1
    prev_basis_bound = 0 
    cur_basis_bound = term_lengths[1] # Keeps track of which "block" of the basis we are in
    cur_section, state = iterate(all_possible_sections)

    # Iterate through all nonzero basis coefficients to extract sections and knots
    for (i, nz) in enumerate(nonzero_indices)
        # Iterate through basis functions until we reach the section for the current one
        while nz > cur_basis_bound
            cur_basis_index += 1
            prev_basis_bound = cur_basis_bound
            cur_basis_bound += term_lengths[cur_basis_index]
            cur_section, state = iterate(all_possible_sections, state)
        end

        # Reverse engineer the sections and knots from the basis function
        sections[i] = cur_section
        knot_index = nz - prev_basis_bound

        #knots[i] = [coltypes[s] == Bool ? true : Tables.getcolumn(X, s)[knot_index] for s in cur_section]
        knots[i] = [Tables.getcolumn(X, s)[knot_index] for s in cur_section]
    end

    return sections, knots
end

function random_sections_and_knots(X::Tables.Columns, n_sampled_features; guaranteed_sections = [], interaction_order_weights = nothing, section_weights = nothing, knot_weights = nothing)
    d = ncol(X) # Number of features
    n = nrow(X) # Number of observations
    coltypes = Tables.schema(X).types

    # First we construct the sections that we want to guarantee are in the basis
    if length(guaranteed_sections) > 0
        fixed_sections = reduce(vcat, fill.(guaranteed_sections, n))
        # TODO: Line below is relatively slow, could make it faster
        fixed_knots = reduce(vcat, [[coltypes[s] == Bool ? true : X[s][i] for s in section] for i in 1:n] for section in guaranteed_sections)
    else
        fixed_sections = Vector{Vector{Int}}()
        fixed_knots = Vector{Vector{Union{Real, Bool}}}()
    end

    random_sections = Vector{Vector{Int}}(undef, n_sampled_features)
    random_knots = Vector{Vector{Real}}(undef, n_sampled_features)

    # Next, we build a distribution over the sections and sample from it
    # We do this by specifying a distribution over the interaction orders,
    # and then a distribution over the sections for each interaction order
    if isnothing(interaction_order_weights)
        interaction_order_weights = StatsBase.Weights(fill(1/d, d)) # Uniformly sample interaction orders
    end

    random_interaction_orders = sample(1:d, interaction_order_weights, n_sampled_features; replace = true)

    if isnothing(section_weights)
        section_weights = StatsBase.Weights(fill(1/d, d)) # Uniformly sample sections
    end

    random_sections = [sample(1:d, section_weights, o; replace = false) for o in random_interaction_orders]

    # Then, we build a distribution over the knots and sample from it

    if isnothing(knot_weights)
        knot_weights = StatsBase.Weights(fill(1/n, n)) # Uniformly sample knots
    end

    random_knot_indices = sample(1:n, knot_weights, n_sampled_features; replace = true) 
    #random_knots = [[coltypes[s] == Bool ? true : X[s][i] for s in section] for (section, i) in zip(random_sections, random_knot_indices)]
    random_knots = [[X[s][i] for s in section] for (section, i) in zip(random_sections, random_knot_indices)]
    # Construct random basis
    return vcat(fixed_sections, random_sections), vcat(fixed_knots, random_knots)
end