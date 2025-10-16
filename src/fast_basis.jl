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
struct NestedMatrix
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
    function NestedMatrix(M::NestedIndicators, X::AbstractMatrix)
        order = binary_bin_search(X[:, M.section], M.bins)
        return new(order, size(M.bins, 1)-1, length(order))
    end
end

# Matrix-free multiplication #

# Multiply a coefficient vector by each indicator basis
mul(B::NestedMatrix, v::AbstractVector) = vcat(cumsum(reverse(v)),[0])[B.order] # Assumes v and B have compatible length

function *(B::NestedMatrix, v::AbstractVector)
    length(v) != B.ncol && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

### Transpose of Indicator Basis Matrix ###
struct NestedMatrixTranspose
    order::Vector{Int64}
    ncol::Int64
    nrow::Int64
end

# Matrix-free multiplication #

# Take inner product of a vector of observations with each indicator basis
# TODO: Might be able to restructure this to eke out a little more performance
function muldif(B::NestedMatrixTranspose, v::AbstractVector) # assumes B and v are compatible
    out = zeros(B.nrow)
    for i in 1:length(v)
        if B.order[i] > B.nrow
            continue
        end
        out[B.order[i]] += v[i]
    end
    return reverse(out)
end

mul(B::NestedMatrixTranspose, v::AbstractVector) = cumsum(muldif(B, v))

function *(B::NestedMatrixTranspose, v::AbstractVector)
    B.ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

# Transpose methods #
transpose(B::NestedMatrix) = NestedMatrixTranspose(B.order, B.nrow, B.ncol)
transpose(B::NestedMatrixTranspose) = NestedMatrix(B.order, B.nrow, B.ncol)

### Blocks of NestedIndicators
struct NestedIndicatorBlocks
    blocks::AbstractVector{NestedIndicators}
    function NestedIndicatorBlocks(sections::AbstractVector{<:AbstractVector{Int64}}, X::AbstractMatrix)
        all_ranks = reduce(hcat, map(competerank, eachcol(X)))
        return new([NestedIndicators(all_ranks::AbstractMatrix{Int64}, section::AbstractVector{Int64}, X::AbstractMatrix) for section in sections])
    end
end

struct NestedMatrixBlocks
    blocks::AbstractVector{NestedMatrix}
    function NestedMatrixBlocks(nested_indicators::NestedIndicatorBlocks, X::AbstractMatrix)
        new(map(block -> NestedMatrix(block, X), nested_indicators.blocks))
    end
end

function mul(B::NestedMatrixBlocks, v::AbstractVector, block_col_ind) # assumes B and v are compatible
    block_starts = vcat([0], cumsum(block_col_ind))
    block_ranges = [(block_starts[i-1]+1):block_starts[i] for i in 2:length(block_starts)]
    reduce(+, mul(B.blocks[i], v[block_ranges[i]]) for i in 1:length(block_ranges))
end

function *(B::NestedMatrixBlocks, v::AbstractVector)
    block_col_ind = map(block -> block.ncol, B.blocks)
    sum(block_col_ind) != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v, block_col_ind)
end

struct NestedMatrixBlocksTranspose
    blocks::AbstractVector{NestedMatrixTranspose}
end

function mul(B::NestedMatrixBlocksTranspose, v::AbstractVector) # assumes B and v are compatible
    reduce(vcat, map(block -> mul(block, v), B.blocks))
end

function *(B::NestedMatrixBlocksTranspose, v::AbstractVector)
    B.blocks[1].ncol != length(v) && throw(ArgumentError(DIM_ERRMSG)) # check if B and v are compatible
    mul(B, v)
end

transpose(B::NestedMatrixBlocks) = NestedMatrixBlocksTranspose(map(b -> transpose(b), B.blocks))
transpose(B::NestedMatrixBlocksTranspose) = NestedMatrixBlocks(map(b -> transpose(b), B.blocks))

# Coordinate descent update
soft_threshold(z, λ) = sign(z) * max(0, abs(z) - λ)
soft_threshold_relaxed(z, λ) = z * (abs(z) > λ)

function coord_update_ls!(β::AbstractVector{Float64}, β0::Float64, X::NestedMatrix, y::AbstractVector{Float64}, σ2::AbstractVector{Float64}, α::Float64, λ::Float64)    
    # Compute residuals for entire block
    r = (y - X * β)

    # Compute correlation term for entire block
    z = (transpose(X) * r) .+ (σ2 .* β)
    
    # Set up variable to track change in loss update
    Δ = 0

    # Sequentially update residuals and soft-threshold
    for k in 1:X.ncol
        β_prev = β[k]
        β[k] = soft_threshold(z[k] - σ2[k]*β0 - Δ, X.nrow*λ*α) / ((1 + (1 - α)*λ) * σ2[k])
        β0 -= (β[k] - β_prev) / n
        Δ += σ2[k] * (β[k] - β_prev)
    end
end


all_ranks = reduce(hcat, map(competerank, eachcol(Xm)))
foo = NestedIndicators(all_ranks, [2], Xm)
X = NestedMatrix(foo, Xm)
#y = (y .- minimum(y)) ./ (maximum(y) - minimum(y))
y = y .- mean(y)
n = X.nrow
σ2 = transpose(X) * ones(X.nrow)

scatter(Xm[:, 2], y)
β = zeros(X.ncol)
λ = 0.0025
α = 1.0
# Compute residuals for entire block
#anim = @animate for _ in 1:100
for _ in 1:10000
    #scatter(Xm[:, 2], y, legend = :topleft)
    r = (y - X * β)

    # Compute correlation term for entire block
    z = (transpose(X) * r) .+ (σ2 .* β)
    
    # Set up variable to track change in loss update
    Δ = 0
    #pred_update = 0
    tracker = zeros(n+1)

    # Sequentially update residuals and soft-threshold
    for k in 1:X.ncol
        β_prev = β[k]
        β[k] = soft_threshold_relaxed(z[k] - Δ, X.nrow*λ*α) / ((1 + (1 - α)*λ) * σ2[k])
        Δ += (β[k] - β_prev) * σ2[k]

        tracker[k+1] = Δ
    end
    #scatter!(Xm[:, 2], X*β, legend = :topleft)
end

#gif(anim; fps = 20)

scatter(Xm[:, 2], y)
scatter!(Xm[:, 2], X*β)

# I think something is wrong with this, bc coordinate descent not working...
perm = reverse(sortperm(Xm[:, 2]))
sorter = reverse(sortperm(X.order))

X2 = Xm[:, 2] .>= Xm[perm, 2]'

# Check whether the compressed matrix matches the true matrix
X2 * ones(n) == (X * ones(X.ncol))
v = randn(n)
X2 * v ≈ (X * v)

transpose(X) * ones(n)

# Check whether the compressed transpose matches the true transpose
X2' * ones(n) == (transpose(X) * ones(n))
v = randn(n)
(X2' * v) ≈ (transpose(X) * v)

# Does HAL give the same thing?
# Turns out it suffers from the same problem...

using GLMNet


model = glmnet(X2, y)

scatter(Xm[:, 2], y)
scatter!(Xm[:, 2], GLMNet.predict(model, X2)[:,20])

# Can also try other bases. Does the same thing!
X3 = Xm[:, 2] .<= Xm[:, 2]'
model3 = glmnet(X3, y)
scatter(Xm[:, 2], y)
scatter!(Xm[:, 2], GLMNet.predict(model, X3)[:,20])

indb = NestedIndicatorBlocks([[1], [2], [3], [1,2], [1,3], [1,2,3]], Xm)
XB = NestedMatrixBlocks(indb, Xm)

