
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
function get_sections_and_knots(X, nonzero_indices, term_lengths)
    coltypes = Tables.schema(X).types # Get type of each variable
    # List all possible interactions of variables
    all_possible_interactions = powerset(1:ncol(X), 1)
    sections = Vector{Vector{Int}}(undef, length(nonzero_indices))
    knots = Vector{Vector{Real}}(undef, length(nonzero_indices))
    # Set up iteration trackers
    cur_basis_index = 1
    prev_basis_bound = 0
    cur_basis_bound = term_lengths[1]
    cur_interaction, state = iterate(all_possible_interactions)

    # Iterate through all nonzero basis coefficients to extract sections and knots
    for (i, nz) in enumerate(nonzero_indices)
        # Iterate through basis functions until we reach the section for the current one
        while nz > cur_basis_bound
            cur_basis_index += 1
            prev_basis_bound = cur_basis_bound
            cur_basis_bound += term_lengths[cur_basis_index]
            cur_interaction, state = iterate(all_possible_interactions, state)
        end

        # Reverse engineer the sections and knots from the basis function
        sections[i] = cur_interaction
        knot_index = nz - prev_basis_bound
        knots[i] = coltypes[cur_interaction] == (Bool,) ? [true] : [Tables.getcolumn(X, s)[knot_index] for s in cur_interaction]
    end

    return sections, knots
end

# Randomly sample basis functions for RandomHAL
# p controls how much sampling is biased towards the main terms
function random_sections_and_knots(X::Tables.Columns, nfeatures; p = 0.5)
    d = ncol(X) # Number of features
    n = nrow(X) # Number of observations
    coltypes = Tables.schema(X).types

    # Decide how to sample sections and knots
    section_dist = Binomial(d-1, p)
    knot_dist = DiscreteUniform(1, n)

    sections = Vector{Vector{Int}}(undef, nfeatures)
    knots = Vector{Vector{Real}}(undef, nfeatures)
    # Construct random basis
    for i in 1:nfeatures
        # Sample number of features to include in each section
        s = 1 + rand(section_dist)
        # Sample the features to include in the section
        sections[i] = sample(1:d, s, replace = false)
        # Select random knots
        #knots[i] = coltypes[sections[i]] == (Bool,) ? [true] : [Tables.getcolumn(X, s)[rand(knot_dist)] for s in sections[i]]
        knots[i] = [coltypes[s] == Bool ? true : Tables.getcolumn(X, s)[rand(knot_dist)] for s in sections[i]]

    end

    return sections, knots
end

function random_sections_and_knots2(X::Tables.Columns, n_sampled_features)
    d = ncol(X) # Number of features
    n = nrow(X) # Number of observations
    coltypes = Tables.schema(X).types

    # Decide how to sample sections and knots
    section_dist = Binomial(d-1, p)
    knot_dist = DiscreteUniform(1, n)

    sections = Vector{Vector{Int}}(undef, n_sampled_features)
    knots = Vector{Vector{Real}}(undef, n_sampled_features)

    # First we construct the sections that we want to guarantee are in the basis

    # Next, we build a distribution over the sections and sample from it

    # Then, we build a distribution over the knots and sample from it

    # Construct random basis
    for i in 1:n_sampled_features
        # Sample number of features to include in each section
        s = 1 + rand(section_dist)
        # Sample the features to include in the section
        sections[i] = sample(1:d, s, replace = false)
        # Select random knots
        knots[i] = coltypes[sections[i]] == (Bool,) ? [true] : [Tables.getcolumn(X, s)[rand(knot_dist)] for s in sections[i]]
    end

    return sections, knots
end

function sample_section(coltypes, n_sampled_features)
    d = length(coltypes) # Number of features
end