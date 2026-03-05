# This file defines the objects needed to construct
# and multiply fast HAL basis matrices
DIM_ERRMSG = "Number of columns of NestedMatrix must match number of rows of vector being multiplied."

abstract type AbstractNestedMatrix end

# Given a matrix of ranks and a section, construct a nested path of bins
function path_sample(all_ranks::AbstractMatrix{Int64}, S::AbstractVector{Int64}; start=1)

    # Filter the ranks to only the sampled section
    ranks_orig = all_ranks[:, S]

    # Sort the observations by their maximum rank, so we can iterate through them in order
    max_rank_order = sortperm(vec(maximum(ranks_orig, dims=2)))
    ranks = ranks_orig[vec(max_rank_order), :]

    # Start the path at the first observation
    path = [start]

    # Bookkeeping variables
    i = start
    k = 1
    n = size(ranks, 1)

    # Construct a path of nested ranks
    @views while (i + k <= n)
        if all(ranks[i+k, :] .>= ranks[i, :])
            append!(path, i + k)
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
function binary_bin_search(X::AbstractMatrix{T}, bins::AbstractMatrix{T}) where T<:Number
    # Check input validity
    size(bins, 2) != size(X, 2) && error("In a binary bin search, the width of each bin must be the same as the width of the input matrix.")

    # Set up tracking variables to perform a binary search for each observation simultaneously
    n = size(X, 1)
    lh = hcat(fill(0, n), fill(size(bins, 1), n))
    mid = lh[:, 1] .+ (lh[:, 2] .- lh[:, 1]) .÷ 2
    not_yet_finished = 1:n

    # Keep halving the search area within nested bins until we've narrowed to a single bin
    while any(lh[:, 1] .+ 1 .< lh[:, 2])
        checks = vec(any(X[not_yet_finished, :] .< bins[mid[not_yet_finished], :], dims=2) .+ 1)
        lh[[CartesianIndex(not_yet_finished[i], checks[i]) for i in 1:length(not_yet_finished)]] .= mid[not_yet_finished]
        mid[not_yet_finished] .= lh[not_yet_finished, 1] .+ (lh[not_yet_finished, 2] .- lh[not_yet_finished, 1]) .÷ 2

        # Track which entries have finished the search
        not_yet_finished = findall(lh[:, 1] .+ 1 .< lh[:, 2])
    end
    # For the output, take the lowest bin
    return lh[:, 1]
end

### Nested Indicators Structure ###
struct NestedIndicators
    section::AbstractVector{Int64}
    bins::AbstractMatrix
    path::AbstractVector{Int64}
end

function NestedIndicators(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix)
    path = path_sample(all_ranks, section; start=1)
    bins = vcat(X[path, section], fill(Inf, length(section))')
    return NestedIndicators(section, bins, path)
end

function subsample(indb::NestedIndicators, max_block_size::Int)
    # Subsample only m observations from the path and bins
    if max_block_size < length(indb.path)
        new_indices = sort(sample(1:length(indb.path), max_block_size, replace=false))
        return NestedIndicators(indb.section, indb.bins[vcat(new_indices, length(indb.path) + 1), :], indb.path[new_indices])
    else
        # Otherwise, if we are asked to sample more than the number of bins in the path, just return the original object
        return indb
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
    return NestedMatrix(order, size(M.bins, 1) - 1, length(order))
end

# Matrix-free multiplication #

# Multiply a coefficient vector by each indicator basis
function mul(B::NestedMatrix, v::AbstractVector) # assumes B and v are compatible
    v_sum = cumsum(v)
    out = Vector{Float64}(undef, B.nrow)
    for i in 1:B.nrow
        out[i] = B.order[i] == 0 ? 0.0 : v_sum[B.order[i]]
    end
    # Perform reverse cumulative sum
    return out
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
    # Get sum within each bin
    for i in 1:B.ncol
        if (B.order[i] <= B.ncol) && (B.order[i] > 0) # If the observation is in a bin, add its value to the sum for that bin
            out[B.order[i]] += v[i]
        end
    end
    # Cumulatively sum bins in reverse order
    for i in 1:(B.nrow-1)
        out[B.nrow - i] += out[B.nrow - i + 1]
    end
    return out
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

### Blocks of NestedIndicators
struct NestedIndicatorBlocks
    blocks::AbstractVector{NestedIndicators}
end

function NestedIndicatorBlocks(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix)
    all_ranks = reduce(hcat, map(competerank, eachcol(X)))
    return NestedIndicatorBlocks([NestedIndicators(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix) for section in sections])
end

subsample(indb::NestedIndicatorBlocks, max_block_size::Int) = NestedIndicatorBlocks([subsample(block, max_block_size) for block in indb.blocks])

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
    return reduce(+, map(i -> mul(B.blocks[i], v[block_ranges[i]]), 1:length(B.blocks)))
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

### Nested Matrix * Low-Rank Matrix Data Structures ###

struct Basis
    indicators::NestedIndicators
    smoothness::Int64
    intercept::AbstractVector{Float64}
end

function Basis(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix, smoothness::Int64)
    indicators = NestedIndicators(all_ranks, section, X)
    intercept = smoothness == 0 ? zeros(length(indicators.path)) : (vec(prod(X[indicators.path, section], dims=2) .^ smoothness) ./ factorial(smoothness))
    # Make sure the intercept is sorted because F multiplies from largest to smallest
    return Basis(indicators, smoothness, intercept)
end

function subsample(basis::Basis, max_block_size::Int)
    # Subsample only m observations from the path and bins
    if max_block_size < length(basis.indicators.path)
        new_indices = sort(sample(1:length(basis.indicators.path), max_block_size, replace=false))
        indicators = NestedIndicators(basis.indicators.section, basis.indicators.bins[vcat(new_indices, length(basis.indicators.path) + 1), :], basis.indicators.path[new_indices])
        return Basis(indicators, basis.smoothness, basis.intercept[new_indices])
    else
        # Otherwise, if we are asked to sample more than the number of bins in the path, just return the original object
        return basis
    end
end

struct BasisMatrix <: AbstractNestedMatrix
    F::NestedMatrix
    l::AbstractVector{Float64}
    r::AbstractVector{Float64}
    smoothness::Int64
    ncol::Int64
    nrow::Int64
end

function BasisMatrix(B::Basis, X::AbstractMatrix)
    F = NestedMatrix(B.indicators, X)
    l = vec(prod(X[:, B.indicators.section], dims=2) .^ B.smoothness) ./ factorial(B.smoothness)
    BasisMatrix(F, l, B.intercept, B.smoothness, F.ncol, F.nrow)
end

mul(B::BasisMatrix, v::AbstractVector) = ((B.l .* mul(B.F, v)) .- mul(B.F, B.r .* v))

#= function mul(B::BasisMatrix, v::AbstractVector) # assumes B and v are compatible
    v_sum = Vector{Float64}(undef, B.ncol)
    vr_sum = Vector{Float64}(undef, B.ncol)
    v_sum[1] = v[1]
    vr_sum[1] = B.r[1] * v[1]

    for i in 2:B.ncol
        v_sum[i] = v_sum[i-1] + v[i]
    end

    out = Vector{Float64}(undef, B.nrow)
    for i in 1:B.nrow
        out[i] = B.order[i] == 0 ? 0.0 : v_sum[B.order[i]]
    end
    return out
end =#

function Base.:*(B::BasisMatrix, v::AbstractVector)
    length(v) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::BasisMatrix, V::AbstractMatrix)
    size(V, 1) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

Base.getindex(B::BasisMatrix, inds...) = BasisMatrix(getindex(B.F, inds...), B.l[inds...], B.r, B.smoothness, B.ncol, length(inds...))

### Transpose of Indicator Basis Matrix ###
struct BasisMatrixTranspose <: AbstractNestedMatrix
    F::NestedMatrixTranspose
    l::AbstractVector{Float64}
    r::AbstractVector{Float64}
    smoothness::Int64
    ncol::Int64
    nrow::Int64
end

mul(B::BasisMatrixTranspose, v::AbstractVector) = (mul(B.F, B.l .* v) .- (B.r .* mul(B.F, v)))

function Base.:*(B::BasisMatrixTranspose, v::AbstractVector)
    B.ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::BasisMatrixTranspose, V::AbstractMatrix)
    size(V, 1) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

Base.getindex(B::BasisMatrixTranspose, inds...) = BasisMatrix(getindex(B.F, inds...), B.l[inds...], B.r, B.smoothness, length(inds...), B.nrow)
# Transpose methods #
transpose(B::BasisMatrix) = BasisMatrixTranspose(transpose(B.F), B.l, B.r, B.smoothness, B.nrow, B.ncol)
transpose(B::BasisMatrixTranspose) = BasisMatrix(transpose(B.F), B.l, B.r, B.smoothness, B.nrow, B.ncol)

struct BasisBlocks
    blocks::AbstractVector{Basis}
end

function BasisBlocks(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix, smoothness::Int64)
    all_ranks = reduce(hcat, map(competerank, eachcol(X)))
    return BasisBlocks([Basis(all_ranks, section, X, smoothness) for section in sections])
end

subsample(indb::BasisBlocks, max_block_size::Int) = BasisBlocks([subsample(block, max_block_size) for block in indb.blocks])

struct BasisMatrixBlocks <: AbstractNestedMatrix
    blocks::AbstractVector{BasisMatrix}
    ncol::Int64
    nrow::Int64
end

function BasisMatrixBlocks(basis_blocks::BasisBlocks, X::AbstractMatrix)
    blocks = map(block -> BasisMatrix(block, X), basis_blocks.blocks)
    ncol = sum(block.ncol for block in blocks)
    nrow = blocks[1].F.nrow
    BasisMatrixBlocks(blocks, ncol, nrow)
end

function mul(B::BasisMatrixBlocks, v::AbstractVector, block_col_ind)
    block_starts = vcat([0], cumsum(block_col_ind))
    block_ranges = [(block_starts[i-1]+1):block_starts[i] for i in 2:length(block_starts)]
    out = zeros(B.blocks[1].nrow)
    for (i, block) in enumerate(B.blocks)
        out .+= ((block.l .* mul(block.F, v[block_ranges[i]])) .- mul(block.F, block.r .* v[block_ranges[i]]))
    end
    return out
end

function Base.:*(B::BasisMatrixBlocks, v::AbstractVector)
    block_col_ind = map(block -> block.ncol, B.blocks)
    sum(block_col_ind) != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v, block_col_ind)
end

function Base.:*(B::BasisMatrixBlocks, V::AbstractMatrix)
    block_col_ind = map(block -> block.ncol, B.blocks)
    sum(block_col_ind) != size(V, 1) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v, block_col_ind) for v in eachcol(V))
end

getindex(B::BasisMatrixBlocks, inds...) = BasisMatrixBlocks([block[inds...] for block in B.blocks], B.ncol, length(inds...))

struct BasisMatrixBlocksTranspose <: AbstractNestedMatrix
    blocks::AbstractVector{BasisMatrixTranspose}
    ncol::Int64
    nrow::Int64
end

function mul(B::BasisMatrixBlocksTranspose, v::AbstractVector) # assumes B and v are compatible
    reduce(vcat, map(block -> mul(block, v), B.blocks))
end

function Base.:*(B::BasisMatrixBlocksTranspose, v::AbstractVector)
    B.blocks[1].F.ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

function Base.:*(B::BasisMatrixBlocksTranspose, V::AbstractMatrix)
    B.blocks[1].F.ncol != size(V, 1) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    reduce(hcat, mul(B, v) for v in eachcol(V))
end

transpose(B::BasisMatrixBlocks) = BasisMatrixBlocksTranspose(map(b -> transpose(b), B.blocks), B.nrow, B.ncol)
transpose(B::BasisMatrixBlocksTranspose) = BasisMatrixBlocks(map(b -> transpose(b), B.blocks), B.nrow, B.ncol)

getindex(B::BasisMatrixBlocksTranspose, inds...) = BasisMatrixBlocksTranspose([block[inds...] for block in B.blocks], length(inds...), B.nrow)


### Extra Utility Functions

function colmeans(B::BasisMatrixBlocks, w::AbstractVector{Float64})
    mul(transpose(B), w)
end

colmeans(B::BasisMatrixBlocks) = colmeans(B, ones(B.nrow) ./ B.nrow)

function squares(B::BasisMatrixTranspose, w::AbstractVector{Float64})
    return (mul(B.F, w .* B.l .^ 2) .+ (B.r .^ 2 .* mul(B.F, w)) .- 2 .* B.r .* mul(B.F, w .* B.l))
end

squares(B::BasisMatrixTranspose) = squares(B, ones(B.ncol))

function squares(B::BasisMatrixBlocksTranspose, w::AbstractVector{Float64})
    reduce(vcat, [squares(block, w) for block in B.blocks])
end

squares(B::BasisMatrixBlocksTranspose) = reduce(vcat, [squares(block) for block in B.blocks])

left_sum(B::BasisMatrixTranspose) = mul(B.F, B.l)
left_sum(B::BasisMatrixTranspose, w::AbstractVector{Float64}) = mul(B.F, w .* B.l)

function left_sum(B::BasisMatrixBlocksTranspose)
    reduce(vcat, [left_sum(block) for block in B.blocks])
end

function left_sum(B::BasisMatrixBlocksTranspose, w::AbstractVector{Float64})
    reduce(vcat, [left_sum(block, w) for block in B.blocks])
end

left_squares(B::BasisMatrixTranspose) = mul(B.F, B.l .^ 2)
left_squares(B::BasisMatrixTranspose, w::AbstractVector{Float64}) = mul(B.F, w .* B.l .^ 2)


function left_squares(B::BasisMatrixBlocksTranspose)
    reduce(vcat, [left_squares(block) for block in B.blocks])
end

function left_squares(B::BasisMatrixBlocksTranspose, w::AbstractVector{Float64})
    reduce(vcat, [left_squares(block, w) for block in B.blocks])
end

nonzero_count(B::BasisMatrixTranspose) = mul(B.F, ones(B.ncol))
nonzero_sum(B::BasisMatrixTranspose, w::AbstractVector{Float64}) = mul(B.F, w)


function nonzero_count(B::BasisMatrixBlocksTranspose)
    reduce(vcat, [nonzero_count(block) for block in B.blocks])
end

function nonzero_sum(B::BasisMatrixBlocksTranspose, w::AbstractVector{Float64})
    reduce(vcat, [nonzero_sum(block, w) for block in B.blocks])
end

wls_reweight(B::BasisMatrixBlocksTranspose, w, w_sum, μ, μ2, invσ2) = (squares(B, w) .+ (μ2 .* w_sum) .- (2 .* μ .* (B * w))) .* invσ2




