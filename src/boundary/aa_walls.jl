"""
Walls in the AA-pattern, applied on the scatter.

The natural place to put bounce-back in a fused kernel is the *write*, not the
read. At the scatter a thread already holds its own post-collision populations,
which is exactly what bounce-back needs, and it can drop the bounced value
straight into the slot its future self will read:

    even step, link s into solid:  write g[x + c_s, s]  (the solid node's slot)
    odd step,  link s into solid:  write g[x, 28 - s]

Work through where each gather looks and those are the two addresses it reads
for direction `28 - s`. So the gather needs no wall logic at all, and no thread
writes a slot another thread reads — the property that makes AA-pattern
race-free survives untouched.

Doing it this way also means the bounced value is built from the *current*
step's post-collision populations and the density of the same node and step,
matching the two-lattice reference term for term, so the two can be compared
exactly rather than approximately.

**What this costs.** Bouzidi's rule for `δ < 1/2` needs the post-collision
populations of the next fluid node out, which a thread does not have and cannot
get without a second pass. Those links fall back to halfway bounce-back here;
`δ ≥ 1/2` uses the interpolated rule, which only ever reads the node's own
populations. `:interpolated_local` on the two-lattice path is the same
compromise, so the reference agrees exactly, and the cost of the compromise is
measured rather than assumed (see `scripts/validate_sphere_drag.jl`).
"""

"""
Per-node wall geometry, compact enough to sit alongside the populations.

`kind[i, j, k]` is `SOLID_NODE` for solid, `0` for fluid with no solid
neighbour, or an index into `deltas` for a fluid node that touches the wall.
`deltas[s, b]` is the wall fraction along direction `s`, or zero when that
neighbour is fluid. Only boundary nodes carry the 27 fractions, so the
whole-grid cost is the one `Int32` per node.
"""
struct WallField{T<:AbstractFloat,K<:AbstractArray{Int32,3},D<:AbstractMatrix{T}}
    kind::K
    deltas::D
    center::NTuple{3,T}
end

const SOLID_NODE = Int32(-1)

"""
The solid nodes as a Bool array, from whatever is holding the geometry.

Broadcast rather than a loop so it works unchanged on a device array, where the
result stays on the device and a crop of it can be taken before anything is
copied back.
"""
solid_mask_of(w::WallField) = w.kind .== SOLID_NODE
solid_mask_of(w) = solid_mask_of(flow_wall(w))

Base.length(w::WallField) = size(w.deltas, 2)

"""
    build_wall_field(ϕ; center = domain centre, sdf_fn = nothing)

Build the wall geometry from a signed-distance field in lattice units, the same
way [`build_links`](@ref) does, but indexed by node so a kernel can look it up.
"""
function build_wall_field(ϕ::Array{T,3};
                          center::NTuple{3,<:Real} = (size(ϕ) .+ 1) ./ 2,
                          sdf_fn = nothing) where {T}
    nx, ny, nz = size(ϕ)
    solid = solid_mask(ϕ)
    kind = zeros(Int32, nx, ny, nz)
    columns = Vector{Vector{T}}()

    for k in 1:nz, j in 1:ny, i in 1:nx
        if solid[i, j, k]
            kind[i, j, k] = SOLID_NODE
            continue
        end
        φf = ϕ[i, j, k]
        column = zeros(T, 27)
        touches = false
        for s in 1:27
            cx, cy, cz = cube_velocity(s)
            (cx == 0 && cy == 0 && cz == 0) && continue
            is = shift_periodic(i, cx, nx)
            js = shift_periodic(j, cy, ny)
            ks = shift_periodic(k, cz, nz)
            solid[is, js, ks] || continue

            δ = φf / (φf - ϕ[is, js, ks])
            if sdf_fn !== nothing
                δ = refine_delta_cube(sdf_fn, i, j, k, s, δ)
            end
            column[s] = clamp(δ, T(1e-3), one(T))
            touches = true
        end
        if touches
            push!(columns, column)
            kind[i, j, k] = Int32(length(columns))
        end
    end

    deltas = isempty(columns) ? zeros(T, 27, 0) : reduce(hcat, columns)
    return WallField(kind, deltas, T.(center))
end

"""Bisect the true signed distance along a cube-slot link, as [`refine_delta`](@ref) does."""
function refine_delta_cube(sdf_fn, i::Integer, j::Integer, k::Integer, s::Integer,
                           δ0::T; iterations::Integer = 40) where {T}
    cx, cy, cz = cube_velocity(s)
    at(t) = sdf_fn((T(i) + t * cx, T(j) + t * cy, T(k) + t * cz))
    lo, hi = zero(T), one(T)
    at(lo) > 0 && at(hi) < 0 || return δ0
    for _ in 1:iterations
        mid = (lo + hi) / 2
        at(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    bounced_value(buf, s, δ, ρw, wall, halfway)

Population returning along `28 - s` after the wall, from this node's own
post-collision populations.
"""
@inline function bounced_value(buf, s::Int, δ::T, wall::T, halfway::Bool) where {T}
    @inbounds fq = buf[s]
    (halfway || δ < T(0.5)) && return fq - wall
    @inbounds return fq / (2δ) + (2δ - one(T)) / (2δ) * buf[28-s] - wall / (2δ)
end

"""
    aa_scatter_walls!(g, buf, even, i, j, k, nx, ny, nz, deltas, b, center, spin, halfway)

Scatter for a node that touches the wall, returning the `(force, torque)` it
hands the body this step.
"""
@inline function aa_scatter_walls!(g, buf, even::Bool, i::Int, j::Int, k::Int,
                                   nx::Int, ny::Int, nz::Int, deltas, b::Int,
                                   center::NTuple{3,T}, spin::NTuple{3,T},
                                   halfway::Bool) where {T}
    ρw = zero(T)
    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds ρw += buf[s]
    end

    Fx = zero(T); Fy = zero(T); Fz = zero(T)
    Tx = zero(T); Ty = zero(T); Tz = zero(T)

    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds δ_s = deltas[s, b]
        cv_s = cube_velocity(s)
        if δ_s == zero(T)
            if even
                @inbounds g[i, j, k, 28-s] = buf[s]
            else
                @inbounds g[shift_periodic(i, cv_s[1], nx),
                            shift_periodic(j, cv_s[2], ny),
                            shift_periodic(k, cv_s[3], nz), s] = buf[s]
            end
        else
            cx_s = T(cv_s[1]); cy_s = T(cv_s[2]); cz_s = T(cv_s[3])
            # Wall point, and the velocity of the body there.
            wx_s = T(i) - center[1] + δ_s * cx_s
            wy_s = T(j) - center[2] + δ_s * cy_s
            wz_s = T(k) - center[3] + δ_s * cz_s
            uwx_s = spin[2] * wz_s - spin[3] * wy_s
            uwy_s = spin[3] * wx_s - spin[1] * wz_s
            uwz_s = spin[1] * wy_s - spin[2] * wx_s
            wall_s = 2 * cube_weight(T, s) * ρw *
                     (cx_s * uwx_s + cy_s * uwy_s + cz_s * uwz_s) * T(INV_CS2)

            val_s = bounced_value(buf, s, δ_s, wall_s, halfway)
            if even
                @inbounds g[shift_periodic(i, cv_s[1], nx),
                            shift_periodic(j, cv_s[2], ny),
                            shift_periodic(k, cv_s[3], nz), s] = val_s
            else
                @inbounds g[i, j, k, 28-s] = val_s
            end

            @inbounds p_s = buf[s] + val_s
            px_s = cx_s * p_s; py_s = cy_s * p_s; pz_s = cz_s * p_s
            Fx += px_s; Fy += py_s; Fz += pz_s
            Tx += wy_s * pz_s - wz_s * py_s
            Ty += wz_s * px_s - wx_s * pz_s
            Tz += wx_s * py_s - wy_s * px_s
        end
    end
    return (Fx, Fy, Fz), (Tx, Ty, Tz)
end

"""
    aa_step_node_walls!(g, buf, even, i, j, k, nx, ny, nz, τ, force, operator, ωb, ωh,
                        kind, deltas, center, spin, halfway)

One AA-pattern step for a single node, wall included. Solid nodes are skipped
and contribute nothing.
"""
@inline function aa_step_node_walls!(g, buf, even::Bool, i::Int, j::Int, k::Int,
                                     nx::Int, ny::Int, nz::Int, τ::T,
                                     force::NTuple{3,T}, operator::Val, ωb::T, ωh::T,
                                     kind, deltas, center::NTuple{3,T},
                                     spin::NTuple{3,T}, halfway::Bool,
                                     smag::T = zero(T)) where {T}
    @inbounds b = kind[i, j, k]
    zero3 = (zero(T), zero(T), zero(T))
    b == SOLID_NODE && return zero3, zero3

    aa_gather!(buf, g, even, i, j, k, nx, ny, nz)
    collide_buffer!(buf, τ, force, operator, ωb, ωh, smag)
    if b == Int32(0)
        aa_scatter!(g, buf, even, i, j, k, nx, ny, nz)
        return zero3, zero3
    end
    return aa_scatter_walls!(g, buf, even, i, j, k, nx, ny, nz, deltas, Int(b),
                             center, spin, halfway)
end

"""
    aa_run_walls!(g, wall, nsteps, τ; force, operator, spin, rule, ...)

Reference CPU driver for the wall-aware AA-pattern, returning the `(force,
torque)`. `nsteps` must be even.

`reduction` picks what the return value is. `:last` gives the final step's
force, which is what the two-lattice reference produces and so what the
equality tests compare against. `:mean` averages over the whole call, which is
what the coupled loop wants: the instantaneous momentum-exchange force on a
body in turbulent flow fluctuates by far more than the mean it is fluctuating
about, so a single step is a poor sample to feed a trajectory.
"""
function aa_run_walls!(g::Array{T,4}, wall::WallField{T}, nsteps::Integer, τ::Real;
                       force::NTuple{3,<:Real} = (0, 0, 0),
                       operator::Symbol = :central_moment,
                       spin::NTuple{3,<:Real} = (0, 0, 0),
                       rule::Symbol = :interpolated_local,
                       reduction::Symbol = :last, smagorinsky::Real = 0.0,
                       omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                       channel = nothing,
                       inlet::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    rule in (:interpolated_local, :halfway) ||
        throw(ArgumentError("rule must be :interpolated_local or :halfway, got $rule"))
    reduction in (:last, :mean) ||
        throw(ArgumentError("reduction must be :last or :mean, got $reduction"))

    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    buf = Vector{T}(undef, 27)
    F = T.(force)
    ω = T.(spin)
    op = Val(operator)
    halfway = rule === :halfway
    total_force = (zero(T), zero(T), zero(T))
    total_torque = (zero(T), zero(T), zero(T))
    sum_force = (zero(T), zero(T), zero(T))
    sum_torque = (zero(T), zero(T), zero(T))

    for n in 1:nsteps
        even = isodd(n)
        total_force = (zero(T), zero(T), zero(T))
        total_torque = (zero(T), zero(T), zero(T))
        for k in 1:nz, j in 1:ny, i in 1:nx
            f, t = aa_step_node_walls!(g, buf, even, i, j, k, nx, ny, nz, T(τ), F, op,
                                       T(omega_bulk), T(omega_higher),
                                       wall.kind, wall.deltas, wall.center, ω, halfway,
                                       T(smagorinsky))
            total_force = total_force .+ f
            total_torque = total_torque .+ t
        end
        sum_force = sum_force .+ total_force
        sum_torque = sum_torque .+ total_torque
        # Only on the even layout, which is where an even step leaves the array
        # — and every even step, because the buffer is only as deep as two
        # steps of wrap.
        if channel !== nothing && iseven(n)
            apply_open!(g, channel, inlet)
        end
    end
    reduction === :mean && return sum_force ./ nsteps, sum_torque ./ nsteps
    return total_force, total_torque
end
