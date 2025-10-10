import Base: *
DIM_ERRMSG = "Number of columns of NestedMatrix must match number of rows of vector being multiplied."

### Utility Functions ###
# Given a matrix of ranks and a section, construct a nested path of bins
function path_sample(all_ranks::AbstractMatrix{Int64}, S::AbstractVector{Int64}; start = 1)

    # Filter the ranks to only the sampled section
    ranks_orig = all_ranks[:, S]

    # Sort the observations by their maximum rank, so we can iterate through them in order
    max_rank_order = sortperm(vec(maximum(ranks_orig, dims = 2)))
    ranks = ranks_orig[vec(max_rank_order), :]

    # Start the path at the first observation
    path = [start]

    # Bookkeeping variables
    i = start
    k = 1
    n = size(ranks, 1)

    # Construct a path of nested ranks
    @views while (i + k <= n)
        if all(ranks[i + k, :] .>= ranks[i, :])
            append!(path, i+k)
            i = i + k
            k = 1
        else
            k += 1
        end
    end

    # Get the original indices of the knots in the path
    return max_rank_order[path]
    # This output provides the indices, *in order*, of the nested knots for the given section
end

# Given a set of nested bins and a vector of observations, output a vector of indices that labels in which bin each observation is contained.
# This will become "order" in a NestedMatrix
function binary_bin_search(X::AbstractMatrix{T}, bins::AbstractMatrix{T}) where T <: Number
    # Check input validity
    size(bins, 2) != size(X, 2) && error("In a binary bin search, the width of each bin must be the same as the width of the input matrix.")

    # Set up tracking variables to perform a binary search for each observation simultaneously
    n = size(X, 1)
    lh = hcat(fill(1, n), fill(length(bins), n))
    mid = lh[:,1] .+ (lh[:,2] .- lh[:,1]) .÷ 2

    # Keep halving the search area within nested bins until we've narrowed to a single bin
    while any(lh[:,1] .+ 1 .< lh[:,2])
        checks = vec(all(X .<= bins[mid, :], dims = 2) .+ 1)
        lh[[CartesianIndex(i, checks[i]) for i in 1:n]] .= mid
        mid = lh[:,1] .+ (lh[:,2] .- lh[:,1]) .÷ 2
    end
    # For the output, we can always take the highest, since the only one that violates this is
    # the lowest bin, for which we wind up with low == high anyways
    return lh[:,2]
end

### Nested Indicators Structure ###
# TODO: Currently includes a "final bin" that sums all of the extraneous stuff not in a bin
# Can we remove this to increase efficiency?
struct NestedIndicators
    section::AbstractVector{Int64}
    bins::AbstractMatrix
    function NestedIndicators(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix)
        path = path_sample(all_ranks, section; start = 1)
        bins = vcat(X[path, section], fill(Inf, length(section))')
        return new(section, bins)
    end
end

### Indicator Basis Matrix ###
struct NestedMatrix
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
    function NestedMatrix(M::NestedIndicators, X::AbstractMatrix)
        order = binary_bin_search(X[:, M.section], M.bins)
        return new(order, size(M.bins, 1), length(order))
    end
end

# Matrix-free multiplication 
function *(B::NestedMatrix, v::AbstractVector)
    length(v) != B.ncol && throw(ArgumentError(DIM_ERRMSG))
    cumsum(v)[B.order]
end

### Transpose of Indicator Basis Matrix ###
struct NestedMatrixTranspose
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
end

# Matrix-free multiplication #
# TODO: Might be able to restructure this to eke out a little more performance
function *(B::NestedMatrixTranspose, v::AbstractVector)
    B.ncol != length(v) && throw(ArgumentError(DIM_ERRMSG))
    out = zeros(B.nrow)
    for i in 1:length(v)
        out[B.order[i]] += v[i]
    end
    return cumsum(out)
end

# Transpose methods #
transpose(B::NestedMatrix) = NestedMatrixTranspose(B.order, B.nrow, B.ncol)
transpose(B::NestedMatrixTranspose) = NestedMatrix(B.order, B.nrow, B.ncol)
