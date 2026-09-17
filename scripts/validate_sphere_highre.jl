#!/usr/bin/env julia
#
# V&V-2, the half that was deferred until the GPU existed: the drag of a smooth
# sphere against Reynolds number and resolution (§8).
#
# **Read §3 of this comment before the numbers.** The first version of this sweep
# varied the Reynolds number by changing the viscosity at a fixed lattice
# velocity, which also changes tau — and the effective wall position of
# bounce-back depends on tau. So every row conflated a Reynolds-number effect
# with a boundary-condition one, and the ratio column came out non-monotone in
# Re for that reason rather than any physical one. Rows now report tau, the
# lattice Mach number and Lambda = (tau - 1/2)^2, and there is a section that
# holds the Reynolds number fixed and varies tau alone.
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
#   3. And underneath both: how much of any discrepancy belongs to tau rather
#      than to resolution or Reynolds number. LBM cannot separate them freely —
#      Re = Ma * c_s * N / (c_s^2 (tau - 1/2)), so reaching a high Reynolds
#      number *requires* tau near one half, and reaching a low one at a usable
#      Mach number requires either a large tau or a fine grid. The tau sweep at
#      fixed Re is what says whether that matters here.
#
# **And the answer that sweep gave: neither.** Holding Re at 100 and moving tau
# from 0.505 to 0.6 leaves the ratio at 0.47-0.49, and doubling the resolution
# leaves it there too. A discretisation error moves when you move the
# discretisation; this one does not, so it is not one.
#
# What is left is the velocity the coefficient is divided by. The box mean is
# fixed by the mass flux, which is the same through every plane whatever the
# wake does; what the wake changes is the *profile*, and the sphere sits on the
# axis, in the retarded part of it. So every row also reports `u_in`, the stream
# on a frontal disc one and a half diameters upstream, and `ratio_in`, the same
# measured force normalised by that.
#
# **The two are a bracket, not a right answer and a wrong one.** The sphere
# responds to something between the core it sits in and the mean over the whole
# plane: the box mean includes the bypass flow the sphere has accelerated, so it
# is too fast, and the core disc is the deepest part of the deficit, so it is
# too slow. C_D falls as the reference velocity rises, so the true coefficient
# lies between `ratio` and `ratio_in`. And in a periodic box there is no plane
# that escapes this — the upstream face is the downstream one, so the disc reads
# the wake wrapped round as well as the approach. At three diameters the disc
# sits exactly at the antipode and the bracket is at its widest.
#
# If that bracket is wider than the thing being measured, this configuration
# cannot validate a drag coefficient at all, whatever the code does. That is a
# statement about the production setup too: the coupled run uses the same
# periodic box (§4.4), so the ball is flying in the same wake.
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
                   lattice_speed::Union{Real,Nothing} = nothing,
                   tau::Union{Real,Nothing} = nothing, window::Real = 4,
                   max_settling::Real = 80, tolerance::Real = 0.01,
                   chunk::Integer = 20, device::Bool = HAS_CUDA) where {T}
    N = Int(resolution)
    edge = round(Int, N * domain)
    R = T(N) / 2

    # Re = u N / nu ties the three together, so exactly one of tau and the
    # lattice velocity can be chosen and the other follows. Choosing tau is the
    # honest option wherever the Mach number allows it, because then a sweep
    # over Re is a sweep over Re; choosing u forces tau to track Re and mixes
    # the boundary condition's tau dependence into every comparison.
    (lattice_speed === nothing) == (tau === nothing) &&
        throw(ArgumentError("give exactly one of `tau` and `lattice_speed`"))
    local u, ν, τ
    if tau === nothing
        u = T(lattice_speed)
        ν = u * N / Re
        τ = T(ν / CS2 + 0.5)
    else
        τ = T(tau)
        ν = T(CS2) * (τ - T(0.5))
        u = T(Re * ν / N)
    end
    τ > 0.5 || error("tau came out at $τ — the Reynolds number is out of reach here")
    ma = u / sqrt(T(CS2))
    ma < 0.3 || error("the lattice Mach number would be $(round(ma, digits=3)) — " *
                      "lower tau or raise the resolution")

    dims = (edge, edge, edge)
    ϕ = sphere_sdf_field(T, dims, R)
    wall = build_wall_field(ϕ; sdf_fn = sphere_sdf_fn(dims, R))

    # The disc that measures what arrives. The stream runs along -x, so upstream
    # is +x; the plane sits one and a half diameters that way, or at the antipode
    # if the box is too small for that — in a box of three diameters those are
    # the same plane, since a periodic box's upstream face is its downstream one.
    # One and a half diameters is far enough that the sphere's own potential
    # blockage accounts for only (1/3)^3 = 4% of the deficit read there; the rest
    # is wake. The disc is the frontal area, so it is the part of the profile the
    # sphere is actually in — and, being the deepest part of the deficit, a lower
    # bound on the stream the sphere responds to.
    centre = (edge + 1) / 2
    iapp = mod1(round(Int, centre) + min(round(Int, 3N ÷ 2), edge ÷ 2), edge)
    disc = zeros(T, dims)
    @inbounds for k in 1:edge, j in 1:edge
        (T(j) - centre)^2 + (T(k) - centre)^2 <= R^2 || continue
        wall.kind[iapp, j, k] == BreakingBallLBM.SOLID_NODE && continue
        disc[iapp, j, k] = one(T)
    end
    ndisc = count(!=(0), disc)

    s = LBMState{T}(dims..., τ; lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0, -u, 0.0, 0.0))
    g = to_cube_order!(similar(s.f), s.f)
    s = nothing

    flow = wall
    if device
        g = CuArray(g)
        flow = gpu_flow(wall)
        disc = CuArray(disc)
    end

    # The settling time is advective at high Reynolds number and viscous at low:
    # a creeping flow reaches steady state in L²/nu, which here is a seventh of
    # one flow-through, so counting windows in flow-throughs would run it a
    # hundred times longer than it needs. Take whichever is shorter.
    settle = min(edge / u, edge^2 / ν)
    per_window = round(Int, window * settle)
    per_window += per_window % (2chunk)
    max_windows = max(3, ceil(Int, max_settling / window))
    Tc = T(2 * edge / u)                  # controller: two flow-throughs
    nfluid = T(count(!=(BreakingBallLBM.SOLID_NODE), wall.kind))

    target = (-u, zero(T), zero(T))
    control = (zero(T), zero(T), zero(T))
    integral = (zero(T), zero(T), zero(T))
    feedforward = (zero(T), zero(T), zero(T))
    prev = T(NaN)
    prev_drift = T(NaN)
    CD = T(NaN); CL = T(NaN); Ū = u; Ūa = u; drift = T(NaN)
    last_force = (T(NaN), T(NaN), T(NaN))
    done = 0
    windows = 0
    t0 = time()

    while windows < max_windows
        sumF = (zero(T), zero(T), zero(T))
        sumU = zero(T)
        sumUa = zero(T)
        n = 0
        local F̄
        while n * chunk < per_window
            ρ̄, ū = device ? flow_mean_velocity(g, flow, control) :
                            mean_fluid_velocity(g, wall, control)
            _, ūa = region_mean_velocity(g, disc, control)
            # Feed-forward plus PI. At a steady state the body force has to
            # supply exactly the momentum the sphere removes, which is the
            # measured surface force spread over the fluid — so hand the
            # controller that and leave it only the difference to close. A pure
            # PI has to wind its integral all the way up to that value, and at
            # low Reynolds number, where the drag is large, it cannot do so
            # inside any run worth waiting for: the creeping-flow check reached
            # a fifth of the velocity it was asked for and was still climbing.
            e = target .- ū
            integral = integral .+ e .* chunk
            control = feedforward .+ (2 / Tc) .* e .+ (1 / Tc^2) .* integral

            F, _ = device ?
                gpu_run_walls!(g, flow.wall, flow.contrib, chunk, τ; force = control,
                               operator = operator, reduction = :mean,
                               smagorinsky = smagorinsky) :
                aa_run_walls!(g, wall, chunk, τ; force = control, operator = operator,
                              reduction = :mean, smagorinsky = smagorinsky)
            all(isfinite, F) || return (CD = T(NaN), CL = T(NaN), τ = τ,
                                        Λ = (τ - T(0.5))^2, Ma = ma, u = u,
                                        Re = T(NaN),
                                        CD_app = T(NaN), Re_app = T(NaN),
                                        approach = T(NaN), U_app = T(NaN),
                                        deficit = T(NaN), drift = T(NaN),
                                        flowthroughs = done * u / edge, windows = windows,
                                        steps = done, seconds = time() - t0,
                                        diverged = true)
            sumF = sumF .+ F
            sumU += -ū[1]                 # the stream runs along -x
            sumUa += -ūa[1]
            feedforward = F ./ nfluid
            n += 1
            done += chunk
        end

        Ū = sumU / n
        Ūa = sumUa / n
        F̄ = sumF ./ n
        q = T(0.5) * Ū^2 * T(π) * R^2     # rho = 1
        CD = -F̄[1] / q
        CL = sqrt(F̄[2]^2 + F̄[3]^2) / q
        windows += 1
        drift = isnan(prev) || CD == 0 ? T(NaN) : (CD - prev) / CD
        prev = CD
        # `force` is what the run reports back, so it has to survive the loop.
        last_force = F̄
        # Two consecutive windows under tolerance, not one. A single window can
        # land inside the bar by accident on a shedding flow, and the Re = 300
        # rows of the first sweep did exactly that — stopping at a drift of
        # −1.0% while still falling.
        settled = windows >= 3 && abs(drift) < tolerance && abs(prev_drift) < tolerance
        prev_drift = drift
        settled && break
    end

    # The same force, normalised by the stream that arrives instead of by the
    # box mean. If the gap between the two ratios is the whole discrepancy, the
    # periodic box is the error and the sphere is fine.
    qa = T(0.5) * Ūa^2 * T(π) * R^2
    CDa = ndisc == 0 || Ūa == 0 ? T(NaN) : -last_force[1] / qa
    return (CD = CD, CL = CL, τ = τ, Λ = (τ - T(0.5))^2, Ma = ma, u = u,
            Re = Ū * 2R / ν, deficit = 1 - Ū / u,
            CD_app = CDa, Re_app = Ūa * 2R / ν, approach = Ūa / Ū, U_app = Ūa,
            drift = drift, force = last_force, U = Ū, ν = ν, R = R, edge = edge,
            flowthroughs = done * u / edge, windows = windows, steps = done,
            seconds = time() - t0, diverged = false)
end

function sweep(label, cases; kwargs...)
    println(label)
    @printf("%-9s %-5s %-6s %-8s %-9s %-6s %-8s %-7s %-7s %-8s %-9s %-6s %-5s\n",
            "Re asked", "N/D", "C_s", "tau", "Lambda", "Ma", "C_D", "ratio",
            "u_in/U", "ratio_in", "drift", "f-thru", "s")
    for (Re, N, cs) in cases
        try
            r = drag_case(Float32; Re = Re, resolution = N, smagorinsky = cs,
                          lattice_speed = 0.05, kwargs...)
            # The reference is evaluated at the Reynolds number the run achieved,
            # not the one it was asked for.
            ref = clift_gauvin(r.diverged ? Re : r.Re)
            if r.diverged
                @printf("%-9.3g %-5d %-6.2f diverged after %d steps\n", Re, N, cs, r.steps)
            else
                ref_in = clift_gauvin(r.Re_app)
                @printf("%-9.3g %-5d %-6.2f %-8.5f %-9.2e %-6.3f %-8.3f %-7.2f %-7.3f %-8.2f %+8.2f%% %-6.0f %-5.0f\n",
                        Re, N, cs, r.τ, r.Λ, r.Ma, r.CD, r.CD / ref,
                        r.approach, r.CD_app / ref_in, 100 * r.drift,
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

"""
Creeping flow, against Hasimoto — the reference this *configuration* has.

The drag curve describes a sphere in a free stream. A periodic box does not
contain one: it contains a cubic array, and at four diameters the spheres are a
sixth of a box apart. Hasimoto solved that problem, `scripts/validate_sphere_drag.jl`
already checks the code against it at 0.97, and running the same check through
*this* script's measurement path separates a bug here from the configuration
being the wrong one for the reference above.
"""
function creeping_check(; device = HAS_CUDA, domains = (3.0, 4.0, 6.0))
    println("Creeping flow against Hasimoto — the reference a periodic box does have")
    @printf("%-9s %-5s %-7s %-9s %-11s %-11s %-8s %-7s %-7s\n",
            "domain/D", "N/D", "tau", "Ma", "F measured", "F Hasimoto", "ratio",
            "u_in/U", "drift")
    for L in domains
        try
            # tau = 0.8, so the wall sits where bounce-back nominally puts it.
            # Reaching Re = 0.05 by inflating nu instead drove tau to 24 and the
            # measured force to a third of theory — the code was fine, the setup
            # was asking where the wall was and getting no useful answer.
            r = drag_case(Float32; Re = 0.05, resolution = 16, domain = L, tau = 0.8,
                          max_settling = 40.0, tolerance = 0.005, device = device)
            μ = r.ν                                   # rho = 1
            F = hasimoto_factor(r.R, r.edge) * 6π * μ * r.R * r.U
            @printf("%-9.1f %-5d %-7.3f %-9.2e %-11.4g %-11.4g %-8.3f %-7.3f %+.2f%%\n",
                    L, 16, r.τ, r.Ma, -r.force[1], F, -r.force[1] / F,
                    r.approach, 100 * r.drift)
        catch err
            @printf("%-9.1f skipped (%s)\n", L, first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()
    println("A ratio near one here means the force measurement, the controller and the")
    println("normalisation are all sound, and that the discrepancies above belong to the")
    println("configuration rather than the code: a cubic array is not a free stream, and")
    println("the drag curve describes the latter. A ratio far from one means this script")
    println("has a defect and nothing else it prints can be trusted.")
    println()
end

"""
One Reynolds number, several relaxation times.

`Re = Ma c_s N / (c_s²(τ−1/2))` ties the Mach number, the resolution and τ
together, so a sweep over Reynolds number at a fixed lattice velocity is also a
sweep over τ — and the effective wall position of bounce-back depends on τ. This
holds the Reynolds number and the resolution fixed and moves τ alone, by moving
the velocity to compensate. Whatever the drag does here is the boundary
condition, not the physics.
"""
function tau_sweep(; device = HAS_CUDA, Re = 100, domain = 4.0, max_settling = 60.0)
    println("One Reynolds number, several relaxation times — the boundary condition alone")
    @printf("%-5s %-7s %-9s %-7s %-8s %-9s %-7s %-7s %-8s %-8s\n",
            "N/D", "tau", "Lambda", "Ma", "C_D", "reference", "ratio",
            "u_in/U", "ratio_in", "drift")
    for (N, τ) in ((20, 0.505), (20, 0.51), (20, 0.52),
                   (40, 0.52), (40, 0.56), (40, 0.6))
        try
            r = drag_case(Float32; Re = Re, resolution = N, domain = domain, tau = τ,
                          max_settling = max_settling, device = device)
            ref = clift_gauvin(r.Re)
            @printf("%-5d %-7.3f %-9.2e %-7.3f %-8.3f %-9.3f %-7.2f %-7.3f %-8.2f %+.2f%%\n",
                    N, r.τ, r.Λ, r.Ma, r.CD, ref, r.CD / ref,
                    r.approach, r.CD_app / clift_gauvin(r.Re_app), 100 * r.drift)
        catch err
            @printf("%-5d %-7.3f skipped (%s)\n", N, τ,
                    first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()
    println("Lambda = (tau - 1/2)^2 is the combination the wall position depends on; for")
    println("two-relaxation-time schemes it is exact at 3/16 = 0.1875, and everything")
    println("reachable at a pitch Reynolds number is orders below that. If the ratio moves")
    println("across these rows, the resolution sweeps above are measuring tau as much as")
    println("they are measuring resolution.")
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
    @printf("Each case runs until two consecutive windows agree to 1%%, or gives up at\n")
    @printf("%.0f settling times and says so in the drift column. A settling time is the\n", ft)
    println("shorter of the advective and viscous ones — a creeping flow reaches steady")
    println("state on L²/nu, which can be a small fraction of one flow-through.\n")

    opts = (domain = domain, max_settling = ft, device = device)

    # First, because everything else depends on it.
    creeping_check(; device = device, domains = quick ? (4.0,) : (3.0, 4.0, 6.0))
    tau_sweep(; device = device, domain = domain, max_settling = ft)

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
    @printf("%-9s %-10s %-9s %-8s %-9s %-7s %-7s %-8s %-7s\n",
            "domain/D", "radius/L", "Re got", "C_D", "reference", "ratio",
            "u_in/U", "ratio_in", "drift")
    for L in (3.0, 4.0, 6.0, 8.0)
        try
            r = drag_case(Float32; Re = 1000, resolution = 20, domain = L,
                          lattice_speed = 0.05, max_settling = ft, device = device)
            ref = clift_gauvin(r.Re)
            @printf("%-9.1f %-10.3f %-9.4g %-8.3f %-9.3f %-7.2f %-7.3f %-8.2f %+.2f%%\n",
                    L, 0.5 / L, r.Re, r.CD, ref, r.CD / ref,
                    r.approach, r.CD_app / clift_gauvin(r.Re_app), 100 * r.drift)
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
    println("Then the two ratio columns, which bracket rather than compete. `ratio` divides")
    println("by the box mean, which includes the bypass flow the sphere has accelerated and")
    println("is therefore too fast; `ratio_in` divides by the core disc upstream, which is")
    println("the deepest part of the wake deficit and is therefore too slow. C_D falls as")
    println("the reference velocity rises, so the true coefficient is between them.")
    println()
    println("A bracket that straddles one says the sphere is fine and the box is the error.")
    println("A bracket that is wide says this configuration cannot measure a drag")
    println("coefficient at all — and `u_in/U` says why, being the fraction of the stream")
    println("that survives the wrap-around to arrive at the sphere. Both are statements")
    println("about the production setup as much as about this sweep: the coupled run uses")
    println("the same periodic box (§4.4), so the pitch flies in the same wake. The way out")
    println("is an inflow and an outflow face, which is the one piece of §4.4 still missing.")
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
