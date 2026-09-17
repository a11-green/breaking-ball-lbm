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
"""
function drag_case(::Type{T}; Re::Real, resolution::Integer, domain::Real,
                   smagorinsky::Real = 0.0, operator::Symbol = :central_moment,
                   lattice_speed::Real = 0.05, flowthroughs::Real = 8,
                   sample_fraction::Real = 0.5, chunk::Integer = 20,
                   device::Bool = HAS_CUDA) where {T}
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

    steps = round(Int, flowthroughs * edge / u)
    steps += steps % (2chunk)
    sample_from = round(Int, (1 - sample_fraction) * steps)
    Tc = T(10 * edge / u)                 # controller time constant: ten flow-throughs

    target = (-u, zero(T), zero(T))
    control = (zero(T), zero(T), zero(T))
    integral = (zero(T), zero(T), zero(T))
    sumF = (zero(T), zero(T), zero(T))
    nsample = 0
    done = 0
    t0 = time()

    while done < steps
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
        done += chunk
        if done > sample_from
            sumF = sumF .+ F
            nsample += 1
        end
        all(isfinite, F) || return (CD = NaN, CL = NaN, τ = τ, steps = done,
                                    seconds = time() - t0, diverged = true)
    end

    F̄ = sumF ./ nsample
    q = T(0.5) * u^2 * T(π) * R^2         # rho = 1
    return (CD = -F̄[1] / q, CL = sqrt(F̄[2]^2 + F̄[3]^2) / q, τ = τ,
            steps = done, seconds = time() - t0, diverged = false)
end

function sweep(label, cases; kwargs...)
    println(label)
    @printf("%-9s %-5s %-6s %-11s %-9s %-9s %-9s %-7s %-6s\n",
            "Re", "N/D", "C_s", "tau-1/2", "C_D", "reference", "ratio", "C_L", "s")
    for (Re, N, cs) in cases
        try
            r = drag_case(Float32; Re = Re, resolution = N, smagorinsky = cs, kwargs...)
            ref = clift_gauvin(Re)
            if r.diverged
                @printf("%-9.3g %-5d %-6.2f %-11.2e diverged after %d steps\n",
                        Re, N, cs, r.τ - 0.5, r.steps)
            else
                @printf("%-9.3g %-5d %-6.2f %-11.2e %-9.3f %-9.3f %-9.2f %-7.3f %-6.0f\n",
                        Re, N, cs, r.τ - 0.5, r.CD, ref, r.CD / ref, r.CL, r.seconds)
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
    ft = quick ? 4.0 : 8.0
    println(device ? "Backend: CUDA" : "Backend: host")
    @printf("Domain: %.0f diameters (blockage: sphere radius is %.3f of the box edge)\n",
            domain, 0.5 / domain)
    @printf("Averaging the last half of %.0f flow-through times\n\n", ft)

    opts = (domain = domain, flowthroughs = ft, device = device)

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

    println("The reference is Clift-Gauvin, which is good to a few percent below the drag")
    println("crisis and meaningless through it — at 2e5 a real sphere's drag depends on the")
    println("free-stream turbulence of whoever measured it, which is exactly why a baseball")
    println("has seams and why this project resolves them rather than trusting a curve.")
    println()
    println("What to read off: the resolution at which the ratio column settles near one,")
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
