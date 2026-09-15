"""
Bounce-back at a curved, possibly moving wall, and the force it transfers.

Two rules are available:

  * `:halfway` — plain bounce-back, which places the wall midway along every
    link regardless of where it actually is;
  * `:interpolated` — Bouzidi's linear rule, which honours the wall fraction δ
    and is the one the production runs use.

The force follows the momentum-exchange method: a population arriving at the
wall carries `c_q f̃_q` towards it and leaves carrying `-c_q f_q̄`, so the wall
receives `c_q (f̃_q + f_q̄)` per link per step.
"""

"""
    bounce_back_values!(vals, s, links; rule = :interpolated, spin = (0, 0, 0))

Compute, from the post-collision populations now in `s.f`, the values that
bounce-back imposes at the next time level, and return the `(force, torque)` the
wall receives. `spin` is the angular velocity of the body in lattice units; the
wall velocity at a link is `spin × x_w`.

Call this after [`collide!`](@ref) and before [`stream!`](@ref).
"""
function bounce_back_values!(vals::Vector{T}, s::LBMState{T}, links::BounceBackLinks{T};
                             rule::Symbol = :interpolated,
                             spin::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    length(vals) == length(links) ||
        throw(DimensionMismatch("vals has length $(length(vals)), links has $(length(links))"))
    rule in (:interpolated, :halfway) ||
        throw(ArgumentError("rule must be :interpolated or :halfway, got $rule"))

    lat = s.lattice
    f = s.f
    nx, ny, nz = s.nx, s.ny, s.nz
    ωx, ωy, ωz = T.(spin)
    Fx = Fy = Fz = zero(T)
    Tx = Ty = Tz = zero(T)

    @inbounds for n in eachindex(vals)
        i, j, k = Int(links.i[n]), Int(links.j[n]), Int(links.k[n])
        q = Int(links.q[n])
        qb = opposite(q)
        cx, cy, cz = T(cxs(lat)[q]), T(cys(lat)[q]), T(czs(lat)[q])
        δ = links.δ[n]

        ρw = zero(T)
        for p in 1:nvelocities(lat)
            ρw += f[i, j, k, p]
        end

        # Wall velocity from rigid-body rotation: u_w = ω × x_w.
        wx, wy, wz = links.xw[n]
        uwx = ωy * wz - ωz * wy
        uwy = ωz * wx - ωx * wz
        uwz = ωx * wy - ωy * wx
        wall = 2 * T(weights(lat)[q]) * ρw * (cx * uwx + cy * uwy + cz * uwz) / T(CS2)

        fq = f[i, j, k, q]
        if rule === :halfway || (δ < 1 // 2 && !links.second_fluid[n])
            val = fq - wall
        elseif δ < 1 // 2
            ib = mod1(i - cxs(lat)[q], nx)
            jb = mod1(j - cys(lat)[q], ny)
            kb = mod1(k - czs(lat)[q], nz)
            val = 2δ * fq + (1 - 2δ) * f[ib, jb, kb, q] - wall
        else
            val = fq / (2δ) + (2δ - 1) / (2δ) * f[i, j, k, qb] - wall / (2δ)
        end
        vals[n] = val

        # Momentum handed to the wall along this link.
        px = cx * (fq + val)
        py = cy * (fq + val)
        pz = cz * (fq + val)
        Fx += px; Fy += py; Fz += pz
        Tx += wy * pz - wz * py
        Ty += wz * px - wx * pz
        Tz += wx * py - wy * px
    end
    return (Fx, Fy, Fz), (Tx, Ty, Tz)
end

"""
    apply_bounce_back!(s, links, vals)

Write the bounce-back values into the streamed populations. Call after
[`stream!`](@ref).
"""
function apply_bounce_back!(s::LBMState{T}, links::BounceBackLinks{T},
                            vals::Vector{T}) where {T}
    f = s.f
    @inbounds for n in eachindex(vals)
        f[links.i[n], links.j[n], links.k[n], opposite(Int(links.q[n]))] = vals[n]
    end
    return s
end

"""
    init_solid!(s, solid)

Put solid nodes at rest equilibrium. They never collide, and every population
they would stream into the fluid is overwritten by bounce-back, but keeping them
finite makes the fields safe to inspect and to write out.
"""
function init_solid!(s::LBMState{T}, solid::AbstractArray{Bool,3}) where {T}
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        solid[i, j, k] || continue
        for q in 1:nvelocities(s.lattice)
            s.f[i, j, k, q] = equilibrium(s.lattice, q, one(T), zero(T), zero(T), zero(T))
        end
    end
    return s
end

"""
    step!(s, links, vals; force = nothing, solid = nothing, les = nothing,
          rule = :interpolated, spin = (0, 0, 0))

One time step with a wall: collide, bounce back, stream, apply. Returns the
`(force, torque)` transferred to the wall during this step.
"""
function step!(s::LBMState{T}, links::BounceBackLinks{T}, vals::Vector{T};
               force = nothing, solid = nothing, les = nothing,
               rule::Symbol = :interpolated, spin::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    collide!(s; force = force, solid = solid, les = les)
    F, τq = bounce_back_values!(vals, s, links; rule = rule, spin = spin)
    stream!(s)
    apply_bounce_back!(s, links, vals)
    return F, τq
end
