#!/usr/bin/env julia
#
# P4: the 6DOF trajectory layer, exercised on the case V&V-5 will be judged
# against (§8), using the analytic coefficient model rather than the CFD.
#
# The point is not the numbers the model produces — a C_L = S fit cannot
# capture what a seam does, which is the whole reason for the CFD. The point is
# that **the same pitch shows wildly different "break" depending on which
# reference trajectory the break is measured against**, and the spread is much
# larger than the 20% tolerance V&V-5 sets. Before a simulated trajectory can be
# compared with a published figure, it has to be established which figure it is.
#
#   julia --project=. scripts/pitch_trajectory.jl

using BreakingBallLBM
using Printf

const RELEASE_HEIGHT = 1.75     # m, provisional — see §11
const EXTENSION = 2.0           # m in front of the rubber, provisional

function report(name, s0, props, aero)
    m = pitch_metrics(s0, props, aero)
    @printf("%-28s S=%.3f  %.1f→%.1f m/s in %.3f s\n", name, m.spin_parameter,
            m.release_speed, m.plate_speed, m.flight_time)
    @printf("%-28s   pfx (last 40 ft)  %6.1f\"  %6.1f\"\n", "",
            inches(m.pfx_horizontal), inches(m.pfx_vertical))
    @printf("%-28s   induced (release) %6.1f\"  %6.1f\"\n", "",
            inches(m.induced_horizontal), inches(m.induced_vertical))
    @printf("%-28s   total movement    %6.1f\"  %6.1f\"\n\n", "",
            inches(m.total_horizontal), inches(m.total_vertical))
    return m
end

function main()
    props = BaseballProperties()
    aero = CoefficientAero(props)

    println("Movement by definition (horizontal toward the pitcher's left, vertical up)\n")

    pitch(axis, rpm, speed) = BallState(;
        position = (EXTENSION, 0.0, RELEASE_HEIGHT), velocity = (speed, 0.0, 0.0),
        spin = spin_from_rpm(axis, rpm))

    report("4-seam, 2400 rpm backspin", pitch((0.0, -1.0, 0.0), 2400, 42.0), props, aero)
    sweeper = report("WBC 2023 sweeper (sidespin)",
                     pitch((0.0, 0.0, 1.0), 2708, 38.99), props, aero)

    println("Reported for the WBC pitch (§8): 17\" horizontal, 32\" vertical.")
    @printf("  horizontal: pfx %.1f\", induced %.1f\", total %.1f\"\n",
            inches(sweeper.pfx_horizontal), inches(sweeper.induced_horizontal),
            inches(sweeper.total_horizontal))
    @printf("  vertical:   pfx %.1f\", induced %.1f\", total %.1f\"\n",
            inches(sweeper.pfx_vertical), inches(sweeper.induced_vertical),
            inches(sweeper.total_vertical))
    println()
    println("The 17\" figure sits close to pfx and nowhere near the release-to-plate")
    println("figure, so it is a pfx number. The 32\" figure matches none of the three,")
    println("and has to be pinned down from Baseball Savant before it can be a test.")
    println()

    # What the run itself will cost, at the resolution the user picks.
    println("Lattice units by resolution (39 m/s free stream, lattice speed 0.05)")
    @printf("%-6s %-10s %-12s %-14s %-12s %-12s\n",
            "N/D", "dx (mm)", "dt (µs)", "tau - 1/2", "Re", "steps/pitch")
    for n in (20, 30, 40, 60, 80)
        u = LatticeUnits(; nodes_per_diameter = n, speed = 39.0)
        r = resolution_report(u)
        @printf("%-6d %-10.3f %-12.3f %-14.2e %-12.3e %-12.0f\n",
                n, r.dx_mm, r.dt_µs, r.tau_margin, r.reynolds,
                to_lattice_steps(u, sweeper.flight_time))
    end
    println()
    println("At a fixed lattice speed, nu_lattice is proportional to the resolution, so")
    println("refining the grid moves tau away from 1/2 and helps stability. It never")
    println("gets far: even at 80 nodes per diameter tau sits 6e-5 above the singular")
    println("value, which is why the operator and the subgrid model carry the run")
    println("(§2.3, §3.1) — and why the step count, not the stability, sets the cost.")
end

main()
