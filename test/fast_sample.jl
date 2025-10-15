using Tables
using CausalTables
using Distributions
import LogExpFunctions: logistic
using StatsBase
using DecisionTree

dgp = @dgp(
        X1 ~ Bernoulli(0.5),
        X2 ~ Beta(1, 1),
        X3 ~ Beta(1, 1),
        X4 ~ Beta(1, 1),
        X5 ~ Normal.(X2, 0.01),
        X6 ~ Normal.(X3 .* (1 .- (2 .*X1)), 0.001),
        X7 ~ Normal.(sin.(2*pi*X2), 0.0001),
        X8 ~ Normal.(1 .- cos.(2*pi*X2), 0.0001),
        X9 ~ Normal.((X3 .- 0.5) .* ((X3 .> 0.5) - (X3 .< 0.5)), 0.0),

        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2 + X2 * X3) - 2.5))),
        #Y ~ (@. Normal(A + X2 * X3 + A * X2 + A * X4 + 0.2 * (sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4)), 0.1))
        #Y ~ (@. Normal(sin.(2*pi * X2), 0.1))
        Y ~ (@. Normal((2*(X2-0.5))^2, 0.0))
    )
scm = StructuralCausalModel(dgp, :A, :Y)
n = 1000
ct = rand(scm, n)
X = Tables.Columns(treatmentparents(ct))
Xm = Tables.matrix(X)
y = vec(responsematrix(ct))

function path_sample(all_ranks::AbstractVector{Tuple}, S::AbstractVector{Int64}; start = 1)

    # Filter the ranks to only the sampled section
    ranks_orig = map(r -> r[S], all_ranks)

    # Sort the observations by their maximum rank, so we can iterate through them in order
    max_rank_order = sortperm(maximum.(ranks_orig))
    ranks = ranks_orig[max_rank_order]

    # Start the path at the first observation
    path = [start]

    # Bookkeeping variables
    i = start
    k = 1
    n = length(ranks)

    # Construct a path of nested ranks
    while (i + k <= n)
        if all(ranks[i + k] .>= ranks[i])
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

function path_sample2(all_ranks, S; start = 1)

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

function binary_bin_search2(ranks, bins)
    n = size(ranks, 1)
    lh = hcat(fill(1, n), fill(length(bins), n))
    mid = lh[:,1] .+ (lh[:,2] .- lh[:,1]) .÷ 2

    while any(lh[:,1] .+ 1 .< lh[:,2])
        checks = vec(all(ranks .<= bins[mid, :], dims = 2) .+ 1)
        lh[[CartesianIndex(i, checks[i]) for i in 1:n]] .= mid
        mid = lh[:,1] .+ (lh[:,2] .- lh[:,1]) .÷ 2
    end
    # We can always take the highest, since the only one that violates this is
    # the lowest bin, for which we wind up with low == high anyways
    return lh[:,2]
end

function fastsum4(v::Vector{Float64}, g::Vector{Int64}, nout)
    out = zeros(nout)
    for i in 1:length(v)
        out[g[i]] += v[i]
    end
    return cumsum(out)
end

@time fastsumtr2(b, g)
@time fastsumtr(b, groups, n)

cumsum(b)[g]

# MUCH FASTER AND EASIER
fastsumtr2(b::Vector{Float64}, g) = cumsum(b)[g]


S = [2,3]
@time begin
    all_ranks_mat = reduce(hcat, competerank.(X))
    path = path_sample2(all_ranks_mat, S; start = 1)
    bins = vcat(all_ranks_mat[path, S], fill(n, length(S))')
    ranks = all_ranks_mat[:, S]
    g = binary_bin_search2(ranks, bins)
end

@time a = fastsum4(y, g, length(path)+1)
# Is this faster than the version that maintains a list of groups instead of a list of where the output goes?
# Possibly due to issues with the sequential addition
# UPDATE: From the below, no it's not. Building the groups takes significant time, 
# AND the fast matrix-vector from the groups is also 2-4x slower.
function build_groups(X::NestedMatrix)
    groups = [Int64[] for _ in 1:X.ncol]
    @simd for i in 1:X.nrow
        append!(groups[X.order[i]], i)
    end
    groups
end
@time groups = build_groups(g)
@time b = fastsum(y, groups)



function binary_bin_search(cur, bins)
    # Set up bookkeeping variables for binary search
    low = 1
    high = length(bins)
    mid = low + (high - low) ÷ 2

    # Perform binary search to find the correct bin
    while (low + 1 < high)
        if all(cur .<= bins[mid])
            high = mid
            mid = low + (high - low) ÷ 2
        else
            low = mid
            mid = low + (high - low) ÷ 2
        end
    end

    # Determine which bin the current rank vector belongs to
    if all(cur .<= bins[low])
        return low
    else
        return high
    end
end

function binary_bin_sort(ranks::AbstractMatrix, bins::AbstractMatrix)
    groups = [Int64[] for _ in 1:size(bins, 1)]
    @simd for i in 1:size(ranks, 1)
        j = binary_bin_search(ranks[i, :], bins)
        append!(groups[j], i)
    end
    return groups
end

function fastsum(v::Vector{Float64}, groups::Vector{Vector{Int64}})
    out = Vector{Float64}(undef, length(groups))
    s = 0
    @simd for i in 1:length(groups)
        @simd for j in groups[i]
            s += v[j]
        end
        out[i] = s
    end
    out
end

function fastsum2(v, groups)
    out = Vector{Float64}(undef, length(groups))
    s = 0
    @simd for i in 1:length(groups)
        @simd for j in groups[i]
            s += v[j]
        end
        out[i] = s
    end
    out
end

fastsum3(v::Vector{Float64}, groups_vec::Vector{Int}, groups_cut::Vector{Int}) = cumsum(v[groups_vec])[groups_cut]

groups2 = convert(Vector{Vector{Int}}, groups)

@time fastsum(y, groups)
@time fastsum(y, groups2)
@time fastsum2(y, groups2)
@time fastsum3(y, groups_vec2, groups_cut)

b = randn(length(groups))

function fastsumtr(b::Vector{Float64}, groups, n)
    out = Vector{Float64}(undef, n)
    s = 0
    @simd for i in reverse(1:length(b))
        # Since entries are nested, cumulatively sum
        s += b[i]
        # Set the output to the current cumulative sum
        out[groups[i]] .= s
    end
    out
end

@time fastsumtr(b, groups2, n)

function get_groups(all_ranks, S)
    path = path_sample(all_ranks, S; start = 1)
    ranks = map(r -> getindex(r, S), all_ranks)
    bins = vcat(ranks[path], ntuple(i -> Inf, length(ranks[path][1])))
    groups = binary_bin_sort(ranks, bins)
    return groups
end

# Compute the ranks of each variable in the subset
@time all_ranks = collect(zip(competerank.(X)...))

S = [2,3]
@time path = path_sample(all_ranks, S; start = 1)
@time path = path_sample2(all_ranks_mat, S; start = 1)

@time ranks = map(r -> getindex(r, S), all_ranks)
@time bins = vcat(ranks[path], ntuple(i -> Inf, length(ranks[path][1])))
@time groups = binary_bin_sort(ranks, bins)
groups2 = convert(Vector{Vector{Int}}, groups)

A = float.(rand(Bernoulli(0.1), n, length(groups)))
@time A' * y
@time fastsum(y, groups2)

@time A * b
@time fastsumtr(b, groups2, n)

s = groups[122]


model = DecisionTreeRegressor(max_depth=10)
@time DecisionTree.fit!(model, Tables.matrix(X), y)


groups_vec = reduce(vcat, groups)
groups_vec2 = convert(Vector{Int}, groups_vec)
groups_cut = cumsum(length.(groups))







