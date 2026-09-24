using Test
using BreakingBallLBM
const BBL = BreakingBallLBM

@testset "BreakingBallLBM" begin
    include("test_source.jl")
    include("test_lattice.jl")
    include("test_collision.jl")
    include("test_streaming.jl")
    include("test_geometry.jl")
    include("test_rotating.jl")
    include("test_boundary.jl")
    include("test_open.jl")
    include("test_central_moments.jl")
    include("test_aa_pattern.jl")
    include("test_aa_walls.jl")
    include("test_turbulence.jl")
    include("test_taylor_green.jl")
    include("test_refine.jl")
    include("test_multi_grid.jl")
    include("test_refined_flow.jl")
    include("test_fields.jl")
    include("test_vtk.jl")
    include("test_config.jl")
    include("test_pitch_view.jl")
    include("test_pitch_csv.jl")
    include("test_measured_break.jl")
    include("test_trajectory.jl")
    include("test_coupling.jl")
end
