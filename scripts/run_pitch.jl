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
    # Snapshots: sub-cycles between them, 0 for none. A production array is
    # gigabytes, so what is written is a box around the ball — which in the
    # ball-following frame is the middle of the grid and stays there — thinned
    # by `snapshot_stride` after the derivatives have been taken on the full
    # grid, so the vorticity is the grid's own and not a coarser grid's.
    snapshot::Int = 0
    snapshot_crop::Float64 = 2.0      # half-width in diameters
    snapshot_stride::Int = 1
    snapshot_dir::String = "snapshots"
    smoke::Bool = false
    refine::Float64 = 0.0         # fine-patch edge in diameters; 0 = uniform grid
    # Open streamwise faces (§4.4.2.1). Without them the box is periodic and
    # the ball flies through its own wake, which V&V-2 measured as the largest
    # error in the whole configuration — the stream arriving at the ball was a
    # third of what the controller was holding. `domain` then sets the lateral
    # width only, and the box stops being a cube.
    open_faces::Bool = false
    upstream::Float64 = 3.0       # run-up ahead of the ball, in diameters
    downstream::Float64 = 8.0     # wake behind it
    device::Bool = HAS_CUDA
end

function parse_args(args)
    c = PitchConfig()
    i = 1
    while i <= length(args)
        a = args[i]
        # An option whose value is missing must say so. Reading past the end
        # of `args` raises a BoundsError with a stack trace into the parser,
        # which tells the person who typed the command nothing about the
        # command they typed.
        take() = i < length(args) ? (i += 1; args[i]) :
                 error("option $a needs a value — try --help")
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
              --snapshot N         write a flow-field snapshot every N sub-cycles
              --snapshot-crop D    half-width written, in diameters (default $(c.snapshot_crop))
              --snapshot-stride S  write every S-th node (default $(c.snapshot_stride))
              --snapshot-dir DIR   where they go (default $(c.snapshot_dir))
              --open               inlet and outlet instead of a periodic wrap, so
                                   the ball is not flying through its own wake
                                   (§4.4.2.1); --domain then sets the width only
              --upstream D         run-up ahead of the ball with --open (default $(c.upstream))
              --downstream D       wake behind it with --open (default $(c.downstream))
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
        elseif a == "--snapshot";   c.snapshot = parse(Int, take())
        elseif a == "--snapshot-crop"; c.snapshot_crop = parse(Float64, take())
        elseif a == "--snapshot-stride"; c.snapshot_stride = parse(Int, take())
        elseif a == "--snapshot-dir"; c.snapshot_dir = take()
        elseif a == "--open";       c.open_faces = true
        elseif a == "--upstream";   c.upstream = parse(Float64, take())
        elseif a == "--downstream"; c.downstream = parse(Float64, take())
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
    # With open faces the streamwise direction is run-up plus wake rather than
    # the lateral width, so the box is a duct and not a cube.
    edge_x = c.open_faces ? round(Int, c.resolution * (c.upstream + c.downstream)) : edge
    dims = (edge_x, edge, edge)
    nodes = prod(dims)
    gib = nodes * (27 * sizeof(T) + 8) / 2^30      # populations, kind, mask
    budget = grid_budget(; nodes_per_diameter = c.resolution, domain_diameters = c.domain)

    if c.open_faces
        @printf("Grid        %d x %d x %d = %.1f M nodes, %.2f GiB (%s)%s\n",
                edge_x, edge, edge, nodes / 1e6, gib, T,
                c.refine > 0 ? ", coarse level" : "")
        @printf("Faces       inlet and outlet: %.1f D of run-up, %.1f D of wake\n",
                c.upstream, c.downstream)
    else
        @printf("Grid        %d³ = %.1f M nodes, %.2f GiB (%s)%s\n",
                edge, nodes / 1e6, gib, T, c.refine > 0 ? ", coarse level" : "")
        println("Faces       periodic — the ball flies through its own wake (§4.4.2.1); " *
                "--open changes that")
    end
    @printf("Spacing     %.3f mm — seam ridge %.2f cells, boundary layer %.2f cells\n",
            budget.dx_mm, budget.seam_cells, budget.boundary_layer_cells)
    @printf("Lattice     tau - 1/2 = %.2e, Ma = %.3f, Re = %.3e\n",
            units.τ - 0.5, mach_number(units), lattice_reynolds(units))
    @printf("Subgrid     %s\n", c.smagorinsky > 0 ?
            @sprintf("Smagorinsky C_s = %.2f", c.smagorinsky) :
            "none — the collision operator's own dissipation only")
    @printf("Blockage    sphere radius is %.3f of the box edge\n", budget.blockage)
    gib_fine = 0.0
    if c.refine > 0
        fine_budget = grid_budget(; nodes_per_diameter = 2 * c.resolution,
                                  domain_diameters = c.refine)
        # The fine patch is a second array on the device, not a view of the
        # first, so it is the sum that has to fit. Checking the coarse level
        # alone passed configurations that then ran out of memory partway
        # through setup, which is the same wasted run as no check at all.
        gib_fine = (4 * round(Int, c.refine * c.resolution / 2) + 1)^3 *
                   (27 * sizeof(T) + 8) / 2^30
        @printf("Refined     %.1f D patch at %d/D: %d³ fine nodes, %.3f mm, seam %.2f cells\n",
                c.refine, 2 * c.resolution, fine_budget.edge,
                fine_budget.dx_mm, fine_budget.seam_cells)
        @printf("            %.2f GiB for the patch on top of the coarse level\n",
                gib_fine)
        budget = fine_budget      # the warning below should judge what resolves the ball
    end
    if budget.seam_cells < 0.2 || budget.boundary_layer_cells < 0.2
        println("\n*** The seam and the boundary layer are far below one cell here, so the")
        println("*** coefficients this produces are a test of the plumbing, not of the")
        println("*** aerodynamics. Expect C_D several times the real value: the sphere is")
        println("*** a staircase at this resolution. See §6.5 for what is reachable.\n")
    end

    # The host path has the same failure and deserves the same answer: without a
    # check it allocates until the operating system stops it, which on a 42 GiB
    # ask is an OutOfMemoryError several seconds in and no indication of which
    # option to lower.
    if !c.device
        free = Int(Sys.free_memory())
        @printf("Memory      %.2f GiB free\n", free / 2^30)
        total = gib + gib_fine
        total > 0.85 * free / 2^30 &&
            error("$(round(total, digits=2)) GiB" *
                  (gib_fine > 0 ? " ($(round(gib, digits=2)) coarse + $(round(gib_fine, digits=2)) fine)" : "") *
                  " does not fit in $(round(free / 2^30, digits=2)) GiB of free host memory — " *
                  "lower --resolution, --domain or --refine")
    end

    if c.device
        free = try Int(CUDA.available_memory()) catch; Int(CUDA.totalmem(CUDA.device())) end
        @printf("VRAM        %.2f GiB free\n", free / 2^30)
        total = gib + gib_fine
        total > 0.85 * free / 2^30 &&
            error("$(round(total, digits=2)) GiB" *
                  (gib_fine > 0 ? " ($(round(gib, digits=2)) coarse + $(round(gib_fine, digits=2)) fine)" : "") *
                  " does not fit in $(round(free / 2^30, digits=2)) GiB — " *
                  "lower --resolution, --domain or --refine")
    end
    return T, units, dims
end

function main(args)
    c = parse_args(args)
    c.distance > c.release[1] ||
        error("the plate (--distance $(c.distance)) is behind the release point " *
              "(--release x = $(c.release[1]))")
    T, units, dims = plan(c)
    edge = dims[2]
    # The stream runs along -x, so upstream is the high-x end: that is where the
    # inlet goes and the ball sits `upstream` diameters below it.
    centre3 = c.open_faces ?
        (T(dims[1] - round(Int, c.upstream * c.resolution)),
         T(edge + 1) / 2, T(edge + 1) / 2) :
        T.((dims .+ 1) ./ 2)

    geom = BaseballGeometry(; diameter = T(0.0748), seam_height = T(0.00079),
                            seam_amplitude = T(0.7))
    props = BaseballProperties(T; diameter = 0.0748)

    ball = BallState(T; position = c.release, velocity = (c.speed, 0.0, 0.0),
                     spin = spin_from_rpm(c.axis, c.rpm))
    spin_lat = lattice_spin(units, ball)

    local wall, flow, state_arg
    clo = (1, 1, 1)                      # the patch's corner, in coarse indices
    if c.refine > 0
        # The patch goes where the ball is, which with open faces is not the
        # middle of the grid: `RotatingWall` centres the ball in the patch, so a
        # patch centred on the box would quietly move the ball there and the
        # run-up would not be the one asked for.
        half = round(Int, c.refine * c.resolution / 2)
        mid = round.(Int, centre3)
        clo = ntuple(d -> mid[d] - half, 3)
        chi = ntuple(d -> mid[d] + half, 3)
        if c.open_faces
            depth = OpenChannel{T}().depth
            (clo[1] > depth + 1 && chi[1] < dims[1] - depth) ||
                error("the refined patch reaches a face buffer: raise --upstream/" *
                      "--downstream or lower --refine")
        end
        grid = TwoGrid(T, dims, clo, chi, units.τ)
        wall = RotatingWall(geom, size(grid.fine)[1:3], units.dx / 2)
        flow = RefinedFlow(grid, wall)
        state_arg = grid
    else
        wall = RotatingWall(geom, dims, units.dx; center = centre3)
        flow = wall
        state_arg = nothing
    end
    # Clear this run's own output before writing any: a directory holding two
    # runs' frames is the interleaving above, and a stale frame is worse than a
    # missing one because it looks like a result. Only the names this script
    # writes are touched, and the count is reported.
    if c.snapshot > 0
        mkpath(c.snapshot_dir)
        stale = filter(f -> occursin(r"^flow-\d+\.vtk$", f) || f == "flow.vtk.series",
                       readdir(c.snapshot_dir))
        if !isempty(stale)
            foreach(f -> rm(joinpath(c.snapshot_dir, f)), stale)
            @printf("Snapshots   removed %d stale frame%s from %s\n",
                    length(stale), length(stale) == 1 ? "" : "s", c.snapshot_dir)
        end
    end

    channel = c.open_faces ? OpenChannel{T}() : nothing
    # On the uniform path the wall field *is* the domain, so the geometry can be
    # asked directly. On the refined one the wall lives in the patch and the
    # faces belong to the coarse level, which is what the patch check above
    # covers instead.
    if channel !== nothing && c.refine == 0 && !open_is_clear(flow_wall(wall), channel)
        error("the ball reaches into a face buffer — raise --upstream or --downstream")
    end
    nsub = min(max_substeps(flow, spin_lat, c.recut_drift), 100)
    run = PitchRun(units, props; substeps = nsub, control_time = 40 * nsub,
                   recut_drift = c.recut_drift, smagorinsky = c.smagorinsky,
                   operator = c.operator, rule = c.rule, channel = channel)

    flowthrough = dims[1] / units.lattice_speed         # steps for the box to convect once
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
    @printf("done in %.0f s, momentum residual %.3f\n", time() - t0, res)

    rows = NamedTuple[]
    t1 = time()
    cycles = 0
    stopped = :plate

    # A snapshot is written from the same array the solver is using, in the even
    # layout `fly!` leaves it in between sub-cycles, and the geometry is read
    # back from whichever backend holds it — after a re-cut on the device, the
    # host copy of the wall is no longer where the ball is.
    # **The series file, and why it is not optional.** A viewer handed a
    # directory of numbered files groups them by name and calls the index the
    # time. Two runs writing into the same directory at different cadences then
    # interleave: the shorter run's frames survive at the indices the longer one
    # never wrote, and the animation cuts between two different simulations
    # while looking like one. So the run states its own frames, in physical
    # seconds, and the viewer is given that list instead of a wildcard.
    snapshots = 0
    series = Tuple{String,Float64}[]
    series_path = joinpath(c.snapshot_dir, "flow.vtk.series")

    function write_series()
        open(series_path, "w") do io
            println(io, "{")
            println(io, "  \"file-series-version\" : \"1.0\",")
            println(io, "  \"files\" : [")
            for (n, (name, t)) in enumerate(series)
                @printf(io, "    { \"name\" : \"%s\", \"time\" : %.6f }%s\n",
                        name, t, n == length(series) ? "" : ",")
            end
            println(io, "  ]")
            println(io, "}")
        end
    end

    function snapshot!(s, tag)
        c.snapshot > 0 || return
        # On the refined path the snapshot is of the *fine* level. It is the one
        # that resolves the ball, and the crop — a couple of diameters about the
        # ball — lies inside the patch by construction, since the patch has to
        # be at least three diameters for the restriction to have room (§6.5.1).
        # The coarse level outside it is the free stream doing very little.
        level = c.refine > 0 ? g.fine : g
        spacing = c.refine > 0 ? units.dx / 2 : units.dx
        origin = c.refine > 0 ? T.((clo .- 1) .* units.dx) : (zero(T), zero(T), zero(T))
        ctr = c.refine > 0 ? (size(g.fine)[1:3] .+ 1) ./ 2 : centre3
        crop = c.snapshot_crop * c.resolution * (c.refine > 0 ? 2 : 1)

        name = @sprintf("flow-%05d.vtk", tag)
        path = joinpath(c.snapshot_dir, name)
        write_snapshot(path, level, solid_mask_of(flow);
                       crop = crop, centre = ctr,
                       stride = c.snapshot_stride, spacing = spacing, origin = origin,
                       title = @sprintf("t = %.4f s, |V| = %.2f m/s", s.ball.t,
                                        speed(s.ball)))
        push!(series, (name, Float64(s.ball.t)))
        # Rewritten after every frame, so a run that is interrupted still leaves
        # a series that names exactly the frames it managed to write.
        write_series()
        snapshots += 1
        return path
    end

    function sample!(s)
        cycles += 1
        CD, CL, Cs = aerodynamic_coefficients(s.force, s.ball, props)
        push!(rows, (t = s.ball.t, x = s.ball.x[1], y = s.ball.x[2], z = s.ball.x[3],
                     vx = s.ball.v[1], vy = s.ball.v[2], vz = s.ball.v[3],
                     CD = CD, CL = CL, Cside = Cs,
                     Fx = s.force[1], Fy = s.force[2], Fz = s.force[3],
                     Tx = s.torque[1], Ty = s.torque[2], Tz = s.torque[3],
                     rpm = spin_rpm(s.ball), steps = s.steps, fresh = s.fresh))

        if c.snapshot > 0 && cycles % c.snapshot == 0
            snapshot!(s, cycles)
        end

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

    # The flow after spin-up is the first one worth looking at: the wake has
    # reached its own length and the trajectory has not started moving yet.
    if c.snapshot > 0
        # Frame zero, numbered rather than named, so it sorts where it belongs.
        p0 = snapshot!(st, 0)
        p0 === nothing || println("Snapshot    ", p0, "  (series: ", series_path, ")")
    end
    println()

    @printf("%-9s %-8s %-8s %-8s %-8s %-7s %-7s %-7s %-6s %-8s\n",
            "t (s)", "x (m)", "y (m)", "z (m)", "|V|", "C_D", "C_L", "C_side", "cuts",
            "residual")
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
