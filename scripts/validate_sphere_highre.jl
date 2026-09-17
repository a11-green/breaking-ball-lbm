#!/usr/bin/env julia
#
# V&V-2, the half that was deferred until the GPU existed: the drag of a smooth
# sphere against Reynolds number and resolution (§8).
#
# Two questions, and they have to be answered together because the answers
# interact:
#
#   1. At what resolution does this code produce a drag coefficient worth
#      believing? Every number the project has printed so far came from grids
#      where the boundary layer was a fraction of a cell.
#   2. Should the production run carry an explicit Smagorinsky model on top of
#      the central-moment operator's own dissipation, and at what C_s (§3.1.1)?
#      At production tau minus one half is 3e-5, so whatever answers this *is*
#      the dissipation at the grid scale.
#
# The sphere is smooth and not spinning, so there is a reference: the standard
# drag curve, which is reliable below the drag crisis and which a seamed,
# spinning ball has no equivalent of. That is the point of validating on it.
#
#   julia --project=. scripts/validate_sphere_highre.jl --quick
#   julia --project=. scripts/validate_sphere_highre.jl

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

"""
    clift_gauvin(Re)

Standard drag correlation for a smooth sphere, good to a few percent from
creeping flow up to about 2e5 — that is, up to but not through the drag crisis,
where no correlation is reliable and the answer depends on the free-stream
turbulence of whoever measured it.
"""
clift_gauvin(Re) = 24 / Re * (1 + 0.15 * Re^0.687) + 0.42 / (1 + 42500 * Re^-1.16)

"""
    drag_case(T; Re, resolution, domain, smagorinsky, ...)

Hold a uniform stream past a fixed smooth sphere and time-average the drag.

The stream is held by the same PI controller the coupled loop uses: a periodic
box has nowhere to put the momentum the sphere removes, so without one the free
stream decays and the Reynolds number drifts through the run.

**The coefficient is normalised by the measured box mean, not by the velocity
asked for.** The first version of this script used the target, and the low
Reynolds numbers then came out at two thirds of the reference and refused to move
when the resolution doubled — a systematic error, not a resolution one. The
controller had not settled: its time constant was ten flow-through times and the
run was four, so the stream never reached the target and `F/(½ρu_target²A)`
divided a force built on the real velocity by a dynamic pressure built on a
larger one. Normalising by what the flow actually did removes the dependence on
the controller converging at all, and the box mean is the right velocity anyway —
it is the superficial velocity, which is what a periodic array's drag is defined
against.

**Each case runs until it stops changing, rather than for a fixed time.** The
drag settles slowly here and how slowly is not obvious in advance: in a periodic
box the sphere sits in its own wake, so the controller holds the box *mean* at
the target while the velocity actually approaching the sphere keeps falling as
the wake fills the box, and the coefficient drifts downward for tens of
flow-through times. On a small box it took sixty-four of them to reach a tenth
of a percent — and a fixed-length run either wastes most of that on the cases
that settle early or reports an unconverged number for the ones that do not. So
the run is windowed, the drift between consecutive windows is the stopping test,
and it is reported: a case that hit the cap without settling says so in that
column rather than looking like the others.
"""
function drag_case(::Type{T}; Re::Real, resolution::Integer, domain::Real,
                   smagorinsky::Real = 0.0, operator::Symbol = :central_moment,
                   lattice_speed::Real = 0.05, window::Real = 4,
                   max_flowthroughs::Real = 80, tolerance::Real = 0.01,
                   chunk::Integer = 20, device::Bool = HAS_CUDA) where {T}
    N = Int(resolution)
    edge = round(Int, N * domain)
    R = T(N) / 2
    u = T(lattice_speed)

    # nu from the Reynolds number we asked for, not the other way round.
    ν = u * N / Re
    τ = T(ν / CS2 + 0.5)
    τ > 0.5 || error("tau came out at $τ — the Reynolds number is out of reach here")

    dims = (edge, edge, edge)
    ϕ = sphere_sdf_field(T, dims, R)
    wall = build_wall_field(ϕ; sdf_fn = sphere_sdf_fn(dims, R))

    s = LBMState{T}(dims..., τ; lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0, -u, 0.0, 0.0))
    g = to_cube_order!(similar(s.f), s.f)
    s = nothing

    flow = wall
    if device
        g = CuArray(g)
        flow = gpu_flow(wall)
    end

    per_window = round(Int, window * edge / u)
    per_window += per_window % (2chunk)
    max_windows = max(2, ceil(Int, max_flowthroughs / window))
    Tc = T(2 * edge / u)                  # controller: two flow-throughs

    target = (-u, zero(T), zero(T))
    control = (zero(T), zero(T), zero(T))
    integral = (zero(T), zero(T), zero(T))
    q0 = T(0.5) * T(π) * R^2
    prev = T(NaN)
    CD = T(NaN); CL = T(NaN); Ū = u; drift = T(NaN)
    done = 0
    windows = 0
    t0 = time()

    while windows < max_windows
        sumF = (zero(T), zero(T), zero(T))
        sumU = zero(T)
        n = 0
        while n * chunk < per_window
            ρ̄, ū = device ? flow_mean_velocity(g, flow, control) :
                            mean_fluid_velocity(g, wall, control)
            e = target .- ū
            integral = integral .+ e .* chunk
            control = (2 / Tc) .* e .+ (1 / Tc^2) .* integral

            F, _ = device ?
                gpu_run_walls!(g, flow.wall, flow.contrib, chunk, τ; force = control,
                               operator = operator, reduction = :mean,
                               smagorinsky = smagorinsky) :
                aa_run_walls!(g, wall, chunk, τ; force = control, operator = operator,
                              reduction = :mean, smagorinsky = smagorinsky)
            all(isfinite, F) || return (CD = T(NaN), CL = T(NaN), τ = τ, Re = T(NaN),
                                        deficit = T(NaN), drift = T(NaN),
                                        flowthroughs = done * u / edge, steps = done,
                                        seconds = time() - t0, diverged = true)
            sumF = sumF .+ F
            sumU += -ū[1]                 # the stream runs along -x
            n += 1
            done += chunk
        end

        Ū = sumU / n
        F̄ = sumF ./ n
        q = T(0.5) * Ū^2 * T(π) * R^2     # rho = 1
        CD = -F̄[1] / q
        CL = sqrt(F̄[2]^2 + F̄[3]^2) / q
        windows += 1
        drift = isnan(prev) || CD == 0 ? T(NaN) : (CD - prev) / CD
        prev = CD
        windows >= 2 && abs(drift) < tolerance && break
    end

    return (CD = CD, CL = CL, τ = τ, Re = Ū * 2R / ν, deficit = 1 - Ū / u,
            drift = drift, flowthroughs = done * u / edge, steps = done,
            seconds = time() - t0, diverged = false)
end

function sweep(label, cases; kwargs...)
    println(label)
    @printf("%-9s %-5s %-6s %-10s %-9s %-9s %-7s %-9s %-6s %-6s\n",
            "Re asked", "N/D", "C_s", "Re got", "C_D", "reference", "ratio", "drift",
            "f-thru", "s")
    for (Re, N, cs) in cases
        try
            r = drag_case(Float32; Re = Re, resolution = N, smagorinsky = cs, kwargs...)
            # The reference is evaluated at the Reynolds number the run achieved,
            # not the one it was asked for.
            ref = clift_gauvin(r.diverged ? Re : r.Re)
            if r.diverged
                @printf("%-9.3g %-5d %-6.2f diverged after %d steps\n", Re, N, cs, r.steps)
            else
                @printf("%-9.3g %-5d %-6.2f %-10.3g %-9.3f %-9.3f %-7.2f %+7.2f%%  %-6.0f %-6.0f\n",
                        Re, N, cs, r.Re, r.CD, ref, r.CD / ref, 100 * r.drift,
                        r.flowthroughs, r.seconds)
            end
        catch err
            @printf("%-9.3g %-5d %-6.2f skipped (%s)\n", Re, N, cs,
                    first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()
end

function main(args)
    quick = "--quick" in args
    device = !("--cpu" in args) && HAS_CUDA
    domain = quick ? 4.0 : 6.0
    ft = quick ? 60.0 : 160.0             # a cap, not a run length
    println(device ? "Backend: CUDA" : "Backend: host")
    @printf("Domain: %.0f diameters (blockage: sphere radius is %.3f of the box edge)\n",
            domain, 0.5 / domain)
    @printf("Each case runs until consecutive windows agree to 1%%, or gives up at %.0f\n", ft)
    println("flow-through times and says so in the drift column.\n")

    opts = (domain = domain, max_flowthroughs = ft, device = device)

    # Where the drag curve is trustworthy and the flow is within reach: does the
    # answer converge, and to what?
    sweep("Resolvable Reynolds numbers — convergence with resolution",
          [(Re, N, 0.0) for Re in (100, 300, 1000) for N in (quick ? (10, 20) : (10, 15, 20, 30))];
          opts...)

    # Where it is not: what does the code produce, and does an explicit subgrid
    # model change it?
    sweep("Beyond reach — the production regime, with and without Smagorinsky",
          [(Re, N, cs) for Re in (1e4, 1e5, 2e5)
                       for N in (quick ? (20,) : (20, 40))
                       for cs in (0.0, 0.16)];
          opts...)

    # How much of what is left is the box rather than the sphere. The periodic
    # images pull the drag up, and the wake deficit pulls the box mean down; both
    # shrink as the domain grows, and neither is a property of the sphere.
    println("Blockage — the same sphere in boxes of different size (Re = 1000, N/D = 20)")
    @printf("%-9s %-10s %-9s %-9s %-9s %-8s %-7s\n",
            "domain/D", "radius/L", "Re got", "C_D", "reference", "ratio", "drift")
    for L in (3.0, 4.0, 6.0, 8.0)
        try
            r = drag_case(Float32; Re = 1000, resolution = 20, domain = L,
                          max_flowthroughs = ft, device = device)
            ref = clift_gauvin(r.Re)
            @printf("%-9.1f %-10.3f %-9.4g %-9.3f %-9.3f %-8.2f %+.2f%%\n",
                    L, 0.5 / L, r.Re, r.CD, ref, r.CD / ref, 100 * r.drift)
        catch err
            @printf("%-9.1f skipped (%s)\n", L, first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()

    println("The reference is Clift-Gauvin, which is good to a few percent below the drag")
    println("crisis and meaningless through it — at 2e5 a real sphere's drag depends on the")
    println("free-stream turbulence of whoever measured it, which is exactly why a baseball")
    println("has seams and why this project resolves them rather than trusting a curve.")
    println()
    println("Read the drift column first: a case that gave up at the cap without settling")
    println("shows it there, and its other columns mean nothing. The drag settles slowly")
    println("because the sphere sits in its own wake.")
    println()
    println("Then: the resolution at which the ratio column settles near one,")
    println("and whether the Smagorinsky rows differ from their C_s = 0 neighbours. If they")
    println("do not, the operator's own dissipation is already doing the work and §3.1's")
    println("explicit model is redundant.")
    println()
    println("*** And one thing not to read off. A drag coefficient near the reference does")
    println("*** NOT license the seam physics. Most of a bluff body's drag is form drag —")
    println("*** the pressure distribution over a large separated wake — and a coarse grid")
    println("*** gets that roughly right even with the boundary layer at a hundredth of a")
    println("*** cell. What the seam manipulates is the separation *point*, which is set by")
    println("*** the boundary layer and is far more resolution-sensitive than C_D. Expect")
    println("*** this sweep to show C_D converging long before anything the project cares")
    println("*** about does, and read the low-Reynolds rows — where the reference is exact")
    println("*** and the geometry, not the boundary layer, is the limit — as the honest")
    println("*** measure of how well the surface is resolved.")
end

main(ARGS)
