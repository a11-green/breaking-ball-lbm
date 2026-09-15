#!/usr/bin/env julia
#
# V&V-2 (creeping-flow limit): drag on a sphere in a periodic cubic array,
# against Hasimoto's analytic solution. A body force drives the flow; at steady
# state the wall force balances it, and the resulting superficial velocity gives
# the drag coefficient to compare.
#
# Run: julia --project=. scripts/validate_sphere_drag.jl

using BreakingBallLBM
using Printf

const τ = 1.0
const RATIO = 1 / 6          # sphere radius / box size, fixed across resolutions
const GFORCE = 1.0e-6        # body force per node, sets the Reynolds number

function run_case(L::Int; rule::Symbol = :interpolated, tol = 1e-5, maxsteps = 40_000)
    R = L * RATIO
    ν = viscosity_from_tau(τ)
    ϕ = sphere_sdf_field((L, L, L), R)
    solid = solid_mask(ϕ)
    links = build_links(ϕ)
    vals = zeros(Float64, length(links))
    g = (GFORCE, 0.0, 0.0)

    s = LBMState(L, L, L, τ)
    init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
    init_solid!(s, solid)

    drag = 0.0
    prev = 0.0
    steps = 0
    while steps < maxsteps
        for _ in 1:200
            F, _ = step!(s, links, vals; force = g, solid = solid, rule = rule)
            drag = F[1]
        end
        steps += 200
        if steps > 400 && abs(drag - prev) <= tol * abs(drag)
            break
        end
        prev = drag
    end

    u = superficial_velocity(s, solid, g)
    μ = ν                                   # ρ₀ = 1 in lattice units
    predicted = stokes_drag(R, μ, u) * hasimoto_factor(R, L)
    re = u * 2R / ν
    nfluid = count(!, solid)

    return (; L, R, steps, drag, u, re, predicted,
            ratio = drag / predicted,
            balance = drag / (nfluid * GFORCE))
end

function main()
    println("Sphere in a periodic cubic array, a/L = 1/6, τ = $τ")
    println("Hasimoto factor K = ", round(hasimoto_factor(RATIO, 1.0), digits = 4))
    println()
    @printf("%-6s %-6s %-8s %-12s %-11s %-9s %-10s %-10s\n",
            "rule", "L", "steps", "drag", "U", "Re", "drag/theory", "F/Σg")
    for rule in (:interpolated, :halfway)
        for L in (24, 36, 48)
            r = run_case(L; rule = rule)
            @printf("%-6s %-6d %-8d %-12.6e %-11.4e %-9.4f %-10.4f %-10.6f\n",
                    rule, r.L, r.steps, r.drag, r.u, r.re, r.ratio, r.balance)
            flush(stdout)
        end
    end
end

main()
