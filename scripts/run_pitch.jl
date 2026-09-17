#!/usr/bin/env julia
#
# Throw one pitch, end to end (§5).
#
# Everything this drives has been verified on its own and against the CPU — the
# collision, the walls, the force reduction, the coupled loop, the re-cut. What
# has not been done is running them together for a whole flight, which is where
# VRAM, stability at production tau, and slow divergence show up. Start with
# --smoke: it is the same code path at a resolution that finishes in a minute.
#
#   julia --project=. scripts/run_pitch.jl --smoke
#   julia --project=. scripts/run_pitch.jl --resolution 40 --domain 8
#   julia --project=. scripts/run_pitch.jl --help
#
# Output goes to a CSV of the trajectory plus a summary on stdout.

using BreakingBallLBM
using Printf

const HAS_CUDA = let
    try
        @eval using CUDA
        @eval using StaticArrays
        CUDA.functional()
    catch
        false
    end
end

"""Settings for a run, all of them overridable from the command line."""
Base.@kwdef mutable struct PitchConfig
    resolution::Int = 40          # nodes per diameter
    domain::Float64 = 8.0         # box edge, in diameters
    precision::Symbol = :f32
    speed::Float64 = 38.99        # m/s — the WBC sweeper (§8)
    rpm::Float64 = 2708.0
    axis::NTuple{3,Float64} = (0.0, 0.0, 1.0)
    release::NTuple{3,Float64} = (2.0, 0.0, 1.75)
    distance::Float64 = PLATE_DISTANCE
    # Spin-up is counted in flow-through times of the box, not in sub-cycles:
    # what has to happen is that the wake reaches its own length, and how many
    # steps that takes depends on the box. Two is not generous — a sphere wake
    # at this Reynolds number is still organising itself — but it is already a
    # quarter of the flight, and the flight continues to develop the flow
    # anyway, which is why the summary averages coefficients over its second
    # half rather than its whole.
    spinup_flowthroughs::Float64 = 2.0
    spinup::Int = 0               # sub-cycles; 0 means derive from the above
    recut_drift::Float64 = 0.25
    operator::Symbol = :central_moment
    rule::Symbol = :interpolated_local
    smagorinsky::Float64 = 0.1
    report_every::Int = 200       # sub-cycles between stdout lines
    max_cycles::Int = 2_000_000
    out::String = "pitch.csv"
    smoke::Bool = false
    refine::Float64 = 0.0         # fine-patch edge in diameters; 0 = uniform grid
    device::Bool = HAS_CUDA
end

function parse_args(args)
    c = PitchConfig()
    i = 1
    while i <= length(args)
        a = args[i]
        take() = (i += 1; args[i])
        if a == "--help"
            println("""
            run_pitch.jl [options]
              --smoke              low resolution, short flight, for checking the plumbing
              --resolution N       nodes per ball diameter (default $(c.resolution))
              --domain L           box edge in diameters (default $(c.domain))
              --precision f32|f64  (default $(c.precision))
              --speed V            release speed, m/s (default $(c.speed))
              --rpm R              spin rate (default $(c.rpm))
              --axis x,y,z         spin axis (default 0,0,1 — sidespin)
              --distance D         release to plate, m (default $(round(c.distance, digits=3)))
              --spinup N           sub-cycles before the trajectory is released
              --spinup-flowthroughs F  instead, in box flow-through times (default $(c.spinup_flowthroughs))
              --refine F           refine a box of F diameters around the ball to 2x
                                   (§6.5.1 — the only way to get a seam worth the
                                   name in an eight-diameter domain; at least 3)
              --smagorinsky C      subgrid constant, 0 to rely on the operator's own
                                   dissipation (default $(c.smagorinsky))
              --cpu                force the host path even if a GPU is present
              --out FILE           trajectory CSV (default $(c.out))
            Coordinates: x toward the plate, z up, y to the pitcher's left.""")
            exit(0)
        elseif a == "--smoke"
            c.smoke = true
            c.resolution = 12; c.domain = 4.0
            c.release = (0.0, 0.0, 1.75); c.distance = 0.4
            c.spinup_flowthroughs = 1.0
            c.report_every = 5; c.out = "pitch-smoke.csv"
        elseif a == "--resolution"; c.resolution = parse(Int, take())
        elseif a == "--domain";     c.domain = parse(Float64, take())
        elseif a == "--precision";  c.precision = Symbol(take())
        elseif a == "--speed";      c.speed = parse(Float64, take())
        elseif a == "--rpm";        c.rpm = parse(Float64, take())
        elseif a == "--axis";       c.axis = Tuple(parse.(Float64, split(take(), ',')))
        elseif a == "--distance";   c.distance = parse(Float64, take())
        elseif a == "--spinup";     c.spinup = parse(Int, take())
        elseif a == "--spinup-flowthroughs"; c.spinup_flowthroughs = parse(Float64, take())
        elseif a == "--refine";     c.refine = parse(Float64, take())
        elseif a == "--smagorinsky"; c.smagorinsky = parse(Float64, take())
        elseif a == "--cpu";        c.device = false
        elseif a == "--out";        c.out = take()
        else
            error("unknown option $a — try --help")
        end
        i += 1
    end
    return c
end

"""How many times the seam has been re-cut, whichever backend is carrying it."""
recut_count(flow) = flow.recuts
recut_count(flow::RefinedFlow) = flow.wall.recuts
recut_count(flow::NamedTuple) = 0

"""Refuse a run that cannot fit, before it spends an hour finding out."""
function plan(c::PitchConfig)
    T = c.precision === :f32 ? Float32 : Float64
    units = LatticeUnits(T; nodes_per_diameter = c.resolution, speed = c.speed)
    edge = round(Int, c.resolution * c.domain)
    nodes = edge^3
    gib = nodes * (27 * sizeof(T) + 8) / 2^30      # populations, kind, mask
    budget = grid_budget(; nodes_per_diameter = c.resolution, domain_diameters = c.domain)

    @printf("Grid        %d³ = %.1f M nodes, %.2f GiB (%s)%s\n",
            edge, nodes / 1e6, gib, T, c.refine > 0 ? ", coarse level" : "")
    @printf("Spacing     %.3f mm — seam ridge %.2f cells, boundary layer %.2f cells\n",
            budget.dx_mm, budget.seam_cells, budget.boundary_layer_cells)
    @printf("Lattice     tau - 1/2 = %.2e, Ma = %.3f, Re = %.3e\n",
            units.τ - 0.5, mach_number(units), lattice_reynolds(units))
    @printf("Subgrid     %s\n", c.smagorinsky > 0 ?
            @sprintf("Smagorinsky C_s = %.2f", c.smagorinsky) :
            "none — the collision operator's own dissipation only")
    @printf("Blockage    sphere radius is %.3f of the box edge\n", budget.blockage)
    if c.refine > 0
        fine_budget = grid_budget(; nodes_per_diameter = 2 * c.resolution,
                                  domain_diameters = c.refine)
        @printf("Refined     %.1f D patch at %d/D: %d³ fine nodes, %.3f mm, seam %.2f cells\n",
                c.refine, 2 * c.resolution, fine_budget.edge,
                fine_budget.dx_mm, fine_budget.seam_cells)
        @printf("            %.2f GiB for the patch on top of the coarse level\n",
                fine_budget.gib)
        budget = fine_budget      # the warning below should judge what resolves the ball
    end
    if budget.seam_cells < 0.2 || budget.boundary_layer_cells < 0.2
        println("\n*** The seam and the boundary layer are far below one cell here, so the")
        println("*** coefficients this produces are a test of the plumbing, not of the")
        println("*** aerodynamics. Expect C_D several times the real value: the sphere is")
        println("*** a staircase at this resolution. See §6.5 for what is reachable.\n")
    end

    if c.device
        free = try Int(CUDA.available_memory()) catch; Int(CUDA.totalmem(CUDA.device())) end
        @printf("VRAM        %.2f GiB free\n", free / 2^30)
        gib > 0.85 * free / 2^30 &&
            error("$(round(gib, digits=2)) GiB does not fit in $(round(free / 2^30, digits=2)) GiB — " *
                  "lower --resolution or --domain")
    end
    return T, units, edge
end

function main(args)
    c = parse_args(args)
    c.distance > c.release[1] ||
        error("the plate (--distance $(c.distance)) is behind the release point " *
              "(--release x = $(c.release[1]))")
    T, units, edge = plan(c)
    dims = (edge, edge, edge)

    geom = BaseballGeometry(; diameter = T(0.0748), seam_height = T(0.00079),
                            seam_amplitude = T(0.7))
    props = BaseballProperties(T; diameter = 0.0748)

    ball = BallState(T; position = c.release, velocity = (c.speed, 0.0, 0.0),
                     spin = spin_from_rpm(c.axis, c.rpm))
    spin_lat = lattice_spin(units, ball)

    local wall, flow, state_arg
    if c.refine > 0
        half = round(Int, c.refine * c.resolution / 2)
        mid = (edge + 1) ÷ 2
        clo = ntuple(_ -> mid - half, 3)
        chi = ntuple(_ -> mid + half, 3)
        grid = TwoGrid(T, dims, clo, chi, units.τ)
        wall = RotatingWall(geom, size(grid.fine)[1:3], units.dx / 2)
        flow = RefinedFlow(grid, wall)
        state_arg = grid
    else
        wall = RotatingWall(geom, dims, units.dx)
        flow = wall
        state_arg = nothing
    end
    nsub = min(max_substeps(flow, spin_lat, c.recut_drift), 100)
    run = PitchRun(units, props; substeps = nsub, control_time = 40 * nsub,
                   recut_drift = c.recut_drift, smagorinsky = c.smagorinsky,
                   operator = c.operator, rule = c.rule)

    flowthrough = edge / units.lattice_speed            # steps for the box to convect once
    spinup = c.spinup > 0 ? c.spinup :
             max(1, round(Int, c.spinup_flowthroughs * flowthrough / nsub))
    @printf("Sub-cycle   %d steps (surface turns %.4f spacings per step)\n",
            nsub, surface_drift_per_step(wall, spin_lat))
    @printf("Flight      %.0f steps expected, re-cut every %d\n",
            (c.distance - c.release[1]) / c.speed / units.dt, nsub)
    @printf("Spin-up     %d sub-cycles = %d steps = %.1f flow-through times\n",
            spinup, spinup * nsub, spinup * nsub / flowthrough)

    # The free stream in the frame is -V, uniform, at rest density.
    u0 = lattice_freestream(units, ball)
    local g
    if c.refine > 0
        init_refined_flow!(flow, (x, y, z) -> (1.0, u0[1], u0[2], u0[3]))
        if c.device
            flow = gpu_refined_flow(flow)
            g = flow.grid
            println("Backend     CUDA, two-level refinement, ", CUDA.name(CUDA.device()))
        else
            g = state_arg
            println("Backend     host, two-level refinement")
        end
    else
        lbm = LBMState{T}(dims..., units.τ; lattice = D3Q27())
        init_equilibrium!(lbm, (i, j, k) -> (1.0, u0[1], u0[2], u0[3]))
        g = to_cube_order!(similar(lbm.f), lbm.f)
        if c.device
            g = CuArray(g)
            flow = gpu_rotating_flow(wall)
            println("Backend     CUDA, ", CUDA.name(CUDA.device()))
        else
            println("Backend     host")
        end
    end
    println()

    st = PitchState(ball)
    print("Spin-up ($spinup sub-cycles, trajectory frozen) ... ")
    flush(stdout)
    t0 = time()
    res = spin_up!(g, run, st, flow; cycles = spinup)
    @printf("done in %.0f s, momentum residual %.3f\n\n", time() - t0, res)

    rows = NamedTuple[]
    @printf("%-9s %-8s %-8s %-8s %-8s %-7s %-7s %-7s %-6s %-8s\n",
            "t (s)", "x (m)", "y (m)", "z (m)", "|V|", "C_D", "C_L", "C_side", "cuts",
            "residual")
    t1 = time()
    cycles = 0
    stopped = :plate

    function sample!(s)
        cycles += 1
        CD, CL, Cs = aerodynamic_coefficients(s.force, s.ball, props)
        push!(rows, (t = s.ball.t, x = s.ball.x[1], y = s.ball.x[2], z = s.ball.x[3],
                     vx = s.ball.v[1], vy = s.ball.v[2], vz = s.ball.v[3],
                     CD = CD, CL = CL, Cside = Cs,
                     Fx = s.force[1], Fy = s.force[2], Fz = s.force[3],
                     Tx = s.torque[1], Ty = s.torque[2], Tz = s.torque[3],
                     rpm = spin_rpm(s.ball), steps = s.steps, fresh = s.fresh))

        if cycles % c.report_every == 0 || s.ball.x[1] >= c.distance
            @printf("%-9.4f %-8.3f %-8.4f %-8.4f %-8.2f %-7.3f %-7.3f %-7.3f %-6d %-8.3f\n",
                    s.ball.t, s.ball.x[1], s.ball.x[2], s.ball.x[3], speed(s.ball),
                    CD, CL, Cs, recut_count(flow), couple_residual(run, s, flow))
            flush(stdout)
        end

        # §5.1: stop on the first sign of trouble rather than burning the rest
        # of the run producing numbers nobody can use.
        if !all(isfinite, s.force) || !all(isfinite, s.ball.v)
            stopped = :diverged
            return false
        end
        if maximum(abs.(s.mean_velocity)) > 0.3
            stopped = :mach
            return false
        end
        s.ball.x[1] < c.distance
    end

    fly!(g, run, st, flow; cycles = c.max_cycles, callback = sample!)
    wall_time = time() - t1

    println()
    @printf("Stopped: %s after %d sub-cycles (%d lattice steps) in %.0f s\n",
            stopped, cycles, st.steps, wall_time)
    if stopped !== :plate
        @printf("  mean velocity %s, force %s\n",
                string(round.(st.mean_velocity, sigdigits = 3)),
                string(round.(st.force, sigdigits = 3)))
    end

    open(c.out, "w") do io
        println(io, join(string.(keys(rows[1])), ","))
        for r in rows
            println(io, join(string.(values(r)), ","))
        end
    end
    @printf("Trajectory: %d rows -> %s\n", length(rows), c.out)

    if stopped === :plate
        final = st.ball
        # The ballistic reference is the release line: straight, at the release
        # velocity, no gravity and no aerodynamics (§ trajectory.jl).
        tb = (c.distance - c.release[1]) / ball.v[1]
        drop = final.x[3] - (c.release[3] + tb * ball.v[3])
        sway = final.x[2] - (c.release[2] + tb * ball.v[2])
        @printf("\nAt the plate after %.4f s at %.2f m/s:\n", final.t, speed(final))
        @printf("  position   y = %+.3f m (%+.1f\"), z = %.3f m\n",
                final.x[2], inches(final.x[2]), final.x[3])
        @printf("  total movement from the release line: %+.1f\" horizontal, %+.1f\" vertical\n",
                inches(sway), inches(drop))
        n = length(rows)
        half = rows[max(1, n ÷ 2):n]
        @printf("  mean coefficients over the second half: C_D %.3f, C_L %.3f, C_side %.3f\n",
                sum(r.CD for r in half) / length(half),
                sum(r.CL for r in half) / length(half),
                sum(r.Cside for r in half) / length(half))
        c.smoke && println("  (a smoke run: these are plumbing, not aerodynamics — see above)")
        @printf("  spin %.0f -> %.0f rpm\n", c.rpm, spin_rpm(final))
        println("\nBreak measured the way a published figure means it needs the pfx")
        println("reference (§8): re-integrate from the CSV with the Magnus component")
        println("removed. The definition is settled; the reference value is not (§11).")
    end
end

main(ARGS)
