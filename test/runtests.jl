using Test
using BreakingBallLBM
const BBL = BreakingBallLBM

@testset "BreakingBallLBM" begin
    include("test_lattice.jl")
    include("test_collision.jl")
    include("test_streaming.jl")
    include("test_taylor_green.jl")
end
