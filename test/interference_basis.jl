using Test
using Random
using CausalTables
using Distributions
using StatsBase
using LinearAlgebra
using MLJBase
using Graphs
import LogExpFunctions: logistic
using Plots

using Profile

using Tables
import Combinatorics: powerset


using RandomHAL

Random.seed!(1234)
dgp = @dgp(
        G = Graphs.adjacency_matrix(erdos_renyi(n, 3/n)),
        X1 ~ Beta(2, 2),
        A ~ Bernoulli.(X1),
        μ = (X1 .> 0.33) .+ (X1 .> 0.66),
        Y ~ Normal.(μ .+ G * μ, 0.01)
    )

scm = StructuralCausalModel(dgp, :A, :Y)
n = 100
ct = rand(scm, n)
X = responseparents(ct)
y = vec(responsematrix(ct))

scatter(X.data.X1, y)
scatter(X.arrays.μ, y)
scatter(X.arrays.μ .+ X.arrays.G * X.arrays.μ, y)


# code for RandomHAL basis
x = Tables.Columns(X)
n = length(y)

# Construct the basis matrix from the data
smoothness = 0
sections, knots = random_sections_and_knots(x, 100)
basis = ha_basis_matrix(x, sections, knots, smoothness)

using FFTW

# X: n × m matrix (n nodes, m columns/features)
# G: n × n dense adjacency matrix (Bool, BitMatrix, or Int/Float)
# Returns Z of size (n, K * m) containing the U-statistics for k = 0, ..., K - 1.
function graph_neighbor_u_statistics(X::AbstractMatrix{<:Real}, G::AbstractMatrix, K::Int)
    K = K + 1
    n, m = size(X)
    @assert size(G) == (n, n) "G must be n × n"
    K <= 0 && return Matrix{Float64}(undef, n, 0)

    X_f64 = Float64.(X)
    Z = zeros(Float64, n, K, m)

    # Parallelize directly over nodes i = 1:n
    Threads.@threads for i in 1:n
        nbrs = findall(!iszero, view(G, i, :))
        d = length(nbrs)
        k_max = min(K, d + 1)
        k_max <= 0 && continue

        vals = X_f64[nbrs, :] # size (d × m)
        E = zeros(Float64, k_max, m)

        # Fast path: direct DP loop when degree d or K is small
        if k_max <= 40 || d <= 64
            E[1, :] .= 1.0
            for r in 1:d
                for k in min(r, k_max - 1):-1:1
                    @views E[k + 1, :] .+= vals[r, :] .* E[k, :]
                end
            end
        else
            # Batched FFT tree across all m columns simultaneously
            conv_trunc_cols(A, B) = begin
                la, lb = size(A, 1), size(B, 1)
                out_len = min(la + lb - 1, k_max)
                if la + lb <= 64
                    C = zeros(Float64, out_len, m)
                    for a_idx in 1:la, b_idx in 1:min(lb, out_len - a_idx + 1)
                        @views C[a_idx + b_idx - 1, :] .+= A[a_idx, :] .* B[b_idx, :]
                    end
                    return C
                end
                len = nextpow(2, la + lb - 1)
                pa, pb = zeros(Float64, len, m), zeros(Float64, len, m)
                pa[1:la, :] .= A
                pb[1:lb, :] .= B
                return irfft(rfft(pa, 1) .* rfft(pb, 1), len, 1)[1:out_len, :]
            end

            polys = [vcat(ones(1, m), reshape(vals[r, :], 1, m)) for r in 1:d]
            while length(polys) > 1
                polys = [length(p) == 2 ? conv_trunc_cols(p...) : first(p) for p in Iterators.partition(polys, 2)]
            end
            E .= first(polys)
        end

        # Convert subset sums E[k + 1, :] to U-statistics by dividing by binomial(d, k)
        Z[i, 1, :] .= 1.0
        binom = 1.0
        for k in 1:(k_max - 1)
            binom *= (d - k + 1) / k
            @views Z[i, k + 1, :] .= E[k + 1, :] ./ binom
        end
    end

    # Remove the zeroth-order statistic while preserving the 3D shape.
    return Z[:, 2:K, :]
end


basis_expanded = graph_neighbor_u_statistics(basis, X.arrays.G, 10) 

# Test first-order sum
first_function = basis_expanded[:, 1, :]
first_denom = sum(X.arrays.G, dims=2)
true_first_function = (X.arrays.G * basis) ./ ifelse.(first_denom .> 0, first_denom, 1)
@test first_function == true_first_function

# Test second-order sum
second_function = basis_expanded[:, 2, :]
second_denom = binomial.(sum(X.arrays.G, dims=2), 2)

true_second_function = ((X.arrays.G * basis).^2 - X.arrays.G * basis.^2) ./ (2 .* ifelse.(second_denom .> 0, second_denom, 1))

@test true_second_function  == second_function

# Test third-order sum
third_function = basis_expanded[:, 3, :]
denom = binomial.(sum(X.arrays.G, dims=2), 3)

true_third_function = ((X.arrays.G * basis).^3 .- 3 .* (X.arrays.G * basis.^2) .* (X.arrays.G * basis) .+ 2 .* X.arrays.G * basis.^3) ./ (6 .* ifelse.(denom .> 0, denom, 1))

@test true_third_function == third_function


