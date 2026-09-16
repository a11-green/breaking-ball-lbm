#!/usr/bin/env julia
#
# How often the seam geometry has to be re-cut, and what that costs (§4.1, §5).
#
# The rotating boundary condition imposes u_wall = ω × r, which is all a smooth
# sphere needs. A seamed ball also needs its *shape* to turn, and since the ridge
# is sub-cell at every resolution this hardware can reach (§6.5), turning it
# means recomputing the wall fractions of interpolated bounce-back. This script
# says how often and how expensively.
#
#   julia --project=. scripts/recut_cost.jl

using BreakingBallLBM
using Printf

function main()
    geom = BaseballGeometry()
    flight = 0.445                      # s, the WBC sweeper (§8)

    @printf("%-6s %-9s %-9s %-10s %-9s %-11s %-9s %-10s\n",
            "N/D", "seam/dx", "shell", "boundary", "drift", "re-cut every", "one cut",
            "per pitch")
    for N in (20, 30, 40, 60)
        units = LatticeUnits(; nodes_per_diameter = N, speed = 39.0)
        ball = BallState(; velocity = (39.0, 0.0, 0.0),
                         spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
        spin = lattice_spin(units, ball)

        edge = 2 * (N + 8)              # just the ball and its shell
        rw = RotatingWall(geom, (edge, edge, edge), units.dx)
        recut!(rw, quat_from_axis_angle((0.0, 0.0, 1.0), 0.01))
        t = minimum(@elapsed(recut!(rw, quat_from_axis_angle((0.0, 0.0, 1.0), 0.01m)))
                    for m in 2:4)

        drift = surface_drift_per_step(rw, spin)
        nsub = max_substeps(rw, spin, 0.25)
        steps = flight / units.dt
        @printf("%-6d %-9.2f %-9d %-10d %-9.4f %-11d %-9.0f %-10.1f\n",
                N, geom.seam_height / units.dx, length(rw.shell),
                count(>(Int32(0)), rw.wall.kind), drift, nsub, 1000t,
                steps / nsub * t / 3600)
    end
    println()
    println("Columns: wall fractions the re-cut rewrites; nodes that carry them; how far")
    println("the surface turns per step in lattice spacings; the resulting sub-cycle")
    println("length; milliseconds per cut on this CPU; hours of cutting per pitch.")
    println()

    units = LatticeUnits(; nodes_per_diameter = 40, speed = 39.0)
    solve = 320.0^3 * (flight / units.dt) / 2.3e9 / 3600
    @printf("At 40 points per diameter the flow solve itself is %.1f h on 320³ (§6.6).\n", solve)
    println("Re-cutting on the CPU costs several times that, so it has to move to the")
    println("device. The per-cut work is not large — a few thousand boundary nodes, a")
    println("handful of solid links each, twenty bisection halvings apiece — it is simply")
    println("serial and in the wrong place. Nothing about it needs the host.")
end

main()
