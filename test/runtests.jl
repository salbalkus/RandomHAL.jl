using Test

begin
    @test include("hal_tests.jl")
    @test include("fast_hal_tests.jl")
end


