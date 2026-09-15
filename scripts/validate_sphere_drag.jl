#!/usr/bin/env julia
#
# V&V-2 (creeping-flow limit): drag on a sphere in a periodic cubic array,
# against Hasimoto's analytic solution. A body force drives the flow; at steady
# state the wall force balances it, and the resulting superficial velocity gives
# the drag to compare.
#
# The body force is scaled as 1/L³ so that every resolution runs at the same
# Reynolds number and the only thing changing is the grid.
#
# Run: julia --project=. scripts/validate_sphere_drag.jl

using BreakingBallLBM
using Printf

const τ = 1.0
const RATIO = 1 / 6          # sphere radius / box size, fixed across resolutions
const L0 = 24
const G0 = 1.0e-6            # body force at L0; scaled as (L0/L)³ to hold Re fixed

function run_case(L::Int; rule::Symbol = :interpolated, refine::Bool = true,
                  tol = 1e-5, maxsteps = 60_000)
    R = L * RATIO
    ν = viscosity_from_tau(τ)
    dims = (L, L, L)
    ϕ = sphere_sdf_field(dims, R)
    solid = solid_mask(ϕ)
    links = build_links(ϕ; sdf_fn = refine ? sphere_sdf_fn(dims, R) : nothing)
    vals = zeros(Float64, length(links))
    g = (G0 * (L0 / L)^3, 0.0, 0.0)

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
    predicted = stokes_drag(R, ν, u) * hasimoto_factor(R, L)   # ρ₀ = 1, so μ = ν
    return (; L, R, steps, drag, u, re = u * 2R / ν, ratio = drag / predicted,
            balance = drag / (count(!, solid) * g[1]))
end

function main()
    println("Sphere in a periodic cubic array, a/L = 1/6, τ = $τ, Re held fixed")
    println("Hasimoto factor K = ", round(hasimoto_factor(RATIO, 1.0), digits = 4))
    println()
    @printf("%-14s %-10s %-4s %-7s %-12s %-11s %-8s %-11s %-9s\n",
            "rule", "δ", "L", "steps", "drag", "U", "Re", "drag/theory", "F/Σg")
    cases = [(:interpolated, true, L) for L in (24, 36, 48)]
    append!(cases, [(:interpolated, false, L) for L in (24, 36, 48)])
    append!(cases, [(:halfway, false, L) for L in (24, 48)])
    for (rule, refine, L) in cases
        r = run_case(L; rule = rule, refine = refine)
        @printf("%-14s %-10s %-4d %-7d %-12.6e %-11.4e %-8.4f %-11.4f %-9.6f\n",
                rule, refine ? "refined" : "linear", r.L, r.steps, r.drag, r.u,
                r.re, r.ratio, r.balance)
        flush(stdout)
    end
end

main()
