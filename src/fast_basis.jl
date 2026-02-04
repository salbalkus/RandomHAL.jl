# This file defines the objects needed to construct
# and multiply fast HAL basis matrices
DIM_ERRMSG = "Number of columns of NestedMatrix must match number of rows of vector being multiplied."

abstract type AbstractNestedMatrix end
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
    lh = hcat(fill(1, n), fill(size(bins, 1), n))
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
struct NestedMatrix <: AbstractNestedMatrix
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
end

function NestedMatrix(M::NestedIndicators, X::AbstractMatrix)
    order = binary_bin_search(X[:, M.section], M.bins)
    return NestedMatrix(order, size(M.bins, 1)-1, length(order))
end

# Matrix-free multiplication #

# Multiply a coefficient vector by each indicator basis
mul(B::NestedMatrix, v::AbstractVector) = vcat(cumsum(reverse(v)),[0])[B.order] # Assumes v and B have compatible length

function mul!(out::AbstractVector, B::NestedMatrix, v::AbstractVector)
    cumsum!(out, reverse(v))
    permute!(out, B.order)
end

function Base.:*(B::NestedMatrix, v::AbstractVector)
    length(v) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::NestedMatrix, V::AbstractMatrix)
    size(V, 1) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

Base.getindex(B::NestedMatrix, inds...) = NestedMatrix(B.order[inds...], B.ncol, length(inds...))

### Transpose of Indicator Basis Matrix ###
struct NestedMatrixTranspose <: AbstractNestedMatrix
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
end

# Matrix-free multiplication #

# Take inner product of a vector of observations with each indicator basis
# TODO: Might be able to restructure this to eke out a little more performance
function mul(B::NestedMatrixTranspose, v::AbstractVector) # assumes B and v are compatible
    out = zeros(B.nrow)
    for i in 1:length(v)
        if B.order[i] > B.nrow
            continue
        end
        out[length(out) - B.order[i] + 1] += v[i]
    end
    cumsum!(out, out)
    return out
end

# inplace version
function mul!(out::AbstractVector, B::NestedMatrixTranspose, v::AbstractVector) # assumes B and v are compatible
    out .= zeros(length(out))
    for i in 1:length(v)
        if B.order[i] > B.nrow
            continue
        end
        out[length(out) - B.order[i] + 1] += v[i]
    end
    cumsum!(out, out)
end

function squares(B::NestedMatrixTranspose) # assumes B and v are compatible
    out = zeros(B.nrow)
    for i in 1:length(B.order)
        if B.order[i] > B.nrow
            continue
        end
        out[length(out) - B.order[i] + 1] += 1 # TODO: Change this to the square of the column value when higher-order implemented
    end
    return cumsum!(out, out)
end


function Base.:*(B::NestedMatrixTranspose, v::AbstractVector)
    B.ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::NestedMatrixTranspose, V::AbstractMatrix)
    size(V, 1) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

Base.getindex(B::NestedMatrixTranspose, inds...) = NestedMatrixTranspose(B.order[inds...], length(inds...), B.nrow)

# Transpose methods #
transpose(B::NestedMatrix) = NestedMatrixTranspose(B.order, B.nrow, B.ncol)
transpose(B::NestedMatrixTranspose) = NestedMatrix(B.order, B.nrow, B.ncol)

### Centered and Scaled Versions of NestedMatrix ###
struct NestedMatrixCS <: AbstractNestedMatrix
    B::NestedMatrix
    μ::AbstractVector{Float64}
    σ::AbstractVector{Float64}
end

function NestedMatrixCS(M::NestedIndicators, X::AbstractMatrix)
    B = NestedMatrix(M, X)
    μ = (transpose(B) * ones(B.nrow)) ./ B.nrow
    σ = sqrt.(squares(transpose(B)) .- B.nrow*(μ.^2))
    return NestedMatrixCS(B, μ, σ)
end

function Base.:*(B::NestedMatrixCS, v::AbstractVector)
    length(v) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    return (B.B * (v ./ B.σ)) .- dot(B.μ, v ./ B.σ)
end

struct NestedMatrixTransposeCS <: AbstractNestedMatrix
    B::NestedMatrixTranspose
    μ::AbstractVector{Float64}
    σ::AbstractVector{Float64}
end

function NestedMatrixTransposeCS(M::NestedIndicators, X::AbstractMatrix)
    B = NestedMatrix(M, X)
    μ = (transpose(B) * ones(B.nrow)) ./ B.nrow
    σ = sqrt.(squares(transpose(B)) .- B.nrow*(μ.^2))
    return NestedMatrixTransposeCS(transpose(B), μ, σ)
end

function Base.:*(B::NestedMatrixTransposeCS, v::AbstractVector)
    length(v) != B.B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    return (B.B * v) .- (B.μ .* sum(v)) ./ B.σ
end


### Blocks of NestedIndicators
struct NestedIndicatorBlocks
    blocks::AbstractVector{NestedIndicators}
    function NestedIndicatorBlocks(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix)
        all_ranks = reduce(hcat, map(competerank, eachcol(X)))
        return new([NestedIndicators(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix) for section in sections])
    end
end

struct NestedMatrixBlocks <: AbstractNestedMatrix
    blocks::AbstractVector{NestedMatrix}
    ncol::Int64
    nrow::Int64
end

function NestedMatrixBlocks(nested_indicators::NestedIndicatorBlocks, X::AbstractMatrix)
    blocks = map(block -> NestedMatrix(block, X), nested_indicators.blocks)
    ncol = sum(block.ncol for block in blocks)
    nrow = blocks[1].nrow
    NestedMatrixBlocks(blocks, ncol, nrow)
end

function mul(B::NestedMatrixBlocks, v::AbstractVector, block_col_ind) # assumes B and v are compatible
    block_starts = vcat([0], cumsum(block_col_ind))
    block_ranges = [(block_starts[i-1]+1):block_starts[i] for i in 2:length(block_starts)]
    output = zeros(B.blocks[1].nrow)
    for i in 1:length(B.blocks)
        output .+= mul(B.blocks[i], v[block_ranges[i]])
    end
    return output
end

function mul!(out::AbstractVector, B::NestedMatrixBlocks, v::AbstractVector) # assumes B and v are compatible
    block_starts = vcat([0], cumsum(map(block -> block.ncol, B.blocks)))
    block_ranges = [(block_starts[i-1]+1):block_starts[i] for i in 2:length(block_starts)]
    out .= zeros(length(out))
    tmp = Vector{Float64}(undef, B.blocks[1].nrow)
    for i in 1:length(B.blocks)
        mul!(tmp, B.blocks[i], v[block_ranges[i]])
        out .+= tmp
    end
    return out
end

function Base.:*(B::NestedMatrixBlocks, v::AbstractVector)
    block_col_ind = map(block -> block.ncol, B.blocks)
    sum(block_col_ind) != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v, block_col_ind)
end

function Base.:*(B::NestedMatrixBlocks, V::AbstractMatrix)
    block_col_ind = map(block -> block.ncol, B.blocks)
    sum(block_col_ind) != size(V, 1) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v, block_col_ind) for v in eachcol(V))
end

getindex(B::NestedMatrixBlocks, inds...) = NestedMatrixBlocks([B.blocks[i][inds...] for i in 1:length(B.blocks)], B.ncol, length(inds...))

struct NestedMatrixBlocksTranspose <: AbstractNestedMatrix
    blocks::AbstractVector{NestedMatrixTranspose}
    ncol::Int64
    nrow::Int64
end

function mul(B::NestedMatrixBlocksTranspose, v::AbstractVector) # assumes B and v are compatible
    reduce(vcat, map(block -> mul(block, v), B.blocks))
end

function Base.:*(B::NestedMatrixBlocksTranspose, v::AbstractVector)
    B.blocks[1].ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::NestedMatrixBlocksTranspose, v::AbstractMatrix)
    B.blocks[1].ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

transpose(B::NestedMatrixBlocks) = NestedMatrixBlocksTranspose(map(b -> transpose(b), B.blocks), B.nrow, B.ncol)
transpose(B::NestedMatrixBlocksTranspose) = NestedMatrixBlocks(map(b -> transpose(b), B.blocks), B.nrow, B.ncol)

getindex(B::NestedMatrixBlocksTranspose, inds...) = NestedMatrixBlocksTranspose([B.blocks[i][inds...] for i in 1:length(B.blocks)], length(inds...), B.nrow)

squares(B::NestedMatrixBlocksTranspose) = reduce(vcat, map(block -> squares(block), B.blocks))

### Centered and Scaled Versions of NestedMatrixBlocks ###
struct NestedMatrixBlocksCS <: AbstractNestedMatrix
    B::NestedMatrixBlocks
    μ::AbstractVector{Float64}
    σ::AbstractVector{Float64}
end

function NestedMatrixBlocksCS(nested_indicators::NestedIndicatorBlocks, X::AbstractMatrix)
    B = NestedMatrixBlocks(nested_indicators, X)
    μ = (transpose(B) * ones(B.nrow)) ./ B.nrow
    σ = sqrt.(squares(transpose(B)) .- B.nrow*(μ.^2))
    return NestedMatrixBlocksCS(B, μ, σ)
end

function Base.:*(B::NestedMatrixBlocksCS, v::AbstractVector)
    length(v) != B.B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    return (B.B * (v ./ B.σ)) .- dot(B.μ, v ./ B.σ)
end

struct NestedMatrixBlocksTransposeCS <: AbstractNestedMatrix
    B::NestedMatrixBlocksTranspose
    μ::AbstractVector{Float64}
    σ::AbstractVector{Float64}
end

function NestedMatrixBlocksTransposeCS(nested_indicators::NestedIndicatorBlocks, X::AbstractMatrix)
    B = NestedMatrixBlocks(nested_indicators, X)
    μ = (transpose(B) * ones(B.nrow)) ./ B.nrow
    σ = sqrt.(squares(transpose(B)) .- B.nrow*(μ.^2))
    return NestedMatrixBlocksTransposeCS(transpose(B), μ, σ)
end

function Base.:*(B::NestedMatrixBlocksTransposeCS, v::AbstractVector)
    length(v) != B.B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    return (B.B * v) .- (B.μ .* sum(v)) ./ B.σ
end