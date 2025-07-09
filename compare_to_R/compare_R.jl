using RCall
using RandomHAL
using Test
using StatsBase
using Tables
import DataAPI: ncol, nrow

#reval("install.packages('hal9001')")
reval("library('hal9001')")

@testset "Compare basis between Julia and R versions" begin
    d = 7
    n = 20
    x = R"matrix(c(rbinom($n, 1, 0.4), rbinom($n, 1, 0.6), rnorm($n * $(d-2))),$n,$d)"
    basis = R"enumerate_basis($x)"
    Rsections = [typeof(r[:cols]) <: AbstractVector ? Int.(r[:cols]) : [Int.(r[:cols])] for r in rcopy(basis)]
    Rknots = [typeof(r[:cutoffs]) <: AbstractVector ? r[:cutoffs] : [r[:cutoffs]] for r in rcopy(basis)]
    Rmat = rcopy(R"as.matrix(make_design_matrix($x, $basis))")

    float_data = Tables.columntable(Tables.table(rcopy(x)[:, 3:d]))
    boolmat = rcopy(x)[:, 1:2]
    bool_data = (binary1 = BitVector(boolmat[:,1]), binary2 = BitVector(boolmat[:,2]))
    data = Tables.Columns(merge(bool_data, float_data))
    Jmat, all_sections, term_lengths = ha_basis_matrix(data, 0)

    # Check whether all of the columns are the same
    R_in_J = mean([c in eachcol(Jmat) for c in eachcol(Rmat)])
    @test R_in_J == 1.0
    J_in_R = mean([c in eachcol(Rmat) for c in eachcol(Jmat)])

    @test J_in_R == 1.0

    findall(.![c in eachcol(Rmat) for c in eachcol(Jmat)])
    
    Jsections, Jknots = get_sections_and_knots(data, 1:size(Jmat, 2), all_sections, term_lengths)
    # All of the sections and knots are the same
    @test mean([jknot ∈ Rknots for jknot in Jknots]) == 1.
    #@test mean([rknot ∈ Jknots for rknot in Rknots]) == 1.

    Jmat2 = ha_basis_matrix(data, Jsections, Jknots, 0)

    R_in_J2 = mean([c in eachcol(Jmat2) for c in eachcol(Rmat)])
    @test R_in_J2 == 1.0
    J_in_R2 = mean([c in eachcol(Rmat) for c in eachcol(Jmat2)])
    @test J_in_R2 == 1.0

    findall(.![c in eachcol(Jmat2) for c in eachcol(Jmat)])

end