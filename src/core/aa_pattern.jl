"""
AA-pattern streaming on D3Q27, in cube-slot order — the form the GPU runs.

Two properties make this the layout for the device:

**One buffer instead of two.** The usual scheme keeps a second copy of the
populations to stream into. AA-pattern alternates two access patterns on a
single array, halving both the memory footprint and the traffic. At the
production resolution that is the difference between 4.3 GB and 7.1 GB — the
difference between fitting an 8 GB card comfortably and not fitting at all.

    even step, node x:  read f[x, s]        → collide → write f[x, opp(s)]
    odd step,  node x:  read f[x - c_s, opp(s)] → collide → write f[x + c_s, s]

Neither pattern lets two nodes touch the same slot, so no thread ever races
another: for a read to collide with a write you would need `x + c_q = x' - c_q'`
with `q = opp(q')`, which forces `x = x'`. After a pair of steps the array is
back in its normal orientation, so drivers advance in pairs.

**Index arithmetic instead of lookup tables.** Slot `s` holds the velocity
`c = (s-1 mod 3, (s-1)÷3 mod 3, (s-1)÷9) - 1`, so velocities, weights and
opposites all come from integer arithmetic rather than tuple indexing — worth
having in a kernel where every thread walks all 27 directions.

Only D3Q27 is supported here. D3Q19 stays on the reference two-lattice path,
where it is only ever used for verification against analytic solutions.
"""

"""
    shift_periodic(i, c, n)

Neighbour index `i + c` wrapped into `1:n`, for `c ∈ {-1, 0, 1}`.

`mod1` would do the same, but modulo by a runtime `n` is an integer division, and
a GPU has no instruction for that. With the slot loops unrolled `c` is a literal,
so this collapses to a single compare-and-select — worth it when the gather and
scatter between them wrap 162 indices per node update.
"""
@inline function shift_periodic(i::Int, c::Int, n::Int)
    c == 0 && return i
    c == 1 && return i == n ? 1 : i + 1
    return i == 1 ? n : i - 1
end

"""Velocity of cube slot `s`."""
@inline function cube_velocity(s::Integer)
    t = s - 1
    return (t % 3 - 1, (t ÷ 3) % 3 - 1, t ÷ 9 - 1)
end

"""Slot holding the velocity opposite to slot `s`."""
@inline cube_opposite(s::Integer) = 28 - s

"""Weight of cube slot `s`."""
@inline function cube_weight(::Type{T}, s::Integer) where {T}
    cx, cy, cz = cube_velocity(s)
    n = cx * cx + cy * cy + cz * cz
    return n == 0 ? T(8 / 27) : n == 1 ? T(2 / 27) : n == 2 ? T(1 / 54) : T(1 / 216)
end

# Reciprocals of the lattice speed of sound, exact in floating point. Dividing by
# CS2 instead costs a real division: x/CS2 and x*3 differ in the last bit, so the
# compiler is not allowed to make the substitution itself, and the BGK branch was
# issuing on the order of 240 divisions per node.
const INV_CS2 = 3.0          # 1 / c_s²
const INV_CS4 = 9.0          # 1 / c_s⁴
const HALF_INV_CS4 = 4.5     # 1 / (2 c_s⁴)
const HALF_INV_CS2 = 1.5     # 1 / (2 c_s²)

"""Equilibrium population for cube slot `s`."""
@inline function cube_equilibrium(s::Integer, ρ::T, ux::T, uy::T, uz::T) where {T}
    icx, icy, icz = cube_velocity(s)
    cu = T(icx) * ux + T(icy) * uy + T(icz) * uz
    usq = ux * ux + uy * uy + uz * uz
    return cube_weight(T, s) * ρ *
           (one(T) - T(HALF_INV_CS2) * usq + T(INV_CS2) * cu + T(HALF_INV_CS4) * cu * cu)
end

"""
    to_cube_order!(g, f)

Rewrite populations stored in lattice order `f[x, y, z, q]` into cube-slot order
`g[x, y, z, s]`.
"""
function to_cube_order!(g::AbstractArray{T,4}, f::AbstractArray{T,4}) where {T}
    @inbounds for q in 1:27
        s = CUBE27[q]
        for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
            g[i, j, k, s] = f[i, j, k, q]
        end
    end
    return g
end

"""Inverse of [`to_cube_order!`](@ref)."""
function from_cube_order!(f::AbstractArray{T,4}, g::AbstractArray{T,4}) where {T}
    @inbounds for q in 1:27
        s = CUBE27[q]
        for k in axes(f, 3), j in axes(f, 2), i in axes(f, 1)
            f[i, j, k, q] = g[i, j, k, s]
        end
    end
    return f
end

"""
    aa_gather!(buf, g, even, i, j, k, nx, ny, nz)

Collect the 27 populations that belong to node `(i, j, k)` at the current time.
"""
@inline function aa_gather!(buf, g, even::Bool, i::Int, j::Int, k::Int,
                            nx::Int, ny::Int, nz::Int)
    if even
        Base.Cartesian.@nexprs 27 s -> begin
            @inbounds buf[s] = g[i, j, k, s]
        end
    else
        Base.Cartesian.@nexprs 27 s -> begin
            cv_s = cube_velocity(s)
            @inbounds buf[s] = g[shift_periodic(i, -cv_s[1], nx),
                                 shift_periodic(j, -cv_s[2], ny),
                                 shift_periodic(k, -cv_s[3], nz), 28-s]
        end
    end
    return buf
end

"""
    aa_scatter!(g, buf, even, i, j, k, nx, ny, nz)

Write the post-collision populations of node `(i, j, k)` back out.
"""
@inline function aa_scatter!(g, buf, even::Bool, i::Int, j::Int, k::Int,
                             nx::Int, ny::Int, nz::Int)
    if even
        Base.Cartesian.@nexprs 27 s -> begin
            @inbounds g[i, j, k, 28-s] = buf[s]
        end
    else
        Base.Cartesian.@nexprs 27 s -> begin
            cv_s = cube_velocity(s)
            @inbounds g[shift_periodic(i, cv_s[1], nx),
                        shift_periodic(j, cv_s[2], ny),
                        shift_periodic(k, cv_s[3], nz), s] = buf[s]
        end
    end
    return g
end

"""
    collide_buffer!(buf, τ, force, Val(operator), ωb, ωh)

Collide the 27 populations of one node, in place, in cube-slot order. The
operator arrives as a `Val` so that it is a compile-time constant: a `Symbol` is
not an isbits type and cannot be passed to a GPU kernel. This is
the whole of the per-node physics, and it is what both the CPU driver and the
GPU kernel call — so the device runs code the analytic benchmarks have already
checked on the host.
"""
@inline function collide_buffer!(buf, τ::T, force::NTuple{3,T}, ::Val{OP},
                                 ωb::T, ωh::T) where {T,OP}
    ρ = zero(T)
    mx = zero(T)
    my = zero(T)
    mz = zero(T)
    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds fq_s = buf[s]
        cv_s = cube_velocity(s)
        ρ += fq_s
        mx += T(cv_s[1]) * fq_s
        my += T(cv_s[2]) * fq_s
        mz += T(cv_s[3]) * fq_s
    end
    invρ = one(T) / ρ
    ux = (mx + force[1] / 2) * invρ
    uy = (my + force[2] / 2) * invρ
    uz = (mz + force[3] / 2) * invρ

    if OP === :central_moment
        to_moments!(buf, ux, uy, uz)
        relax_moments!(buf, ρ, one(T) / τ, force, ωb, ωh)
        to_populations!(buf, ux, uy, uz)
    else
        ω = one(T) / τ
        pre = one(T) - ω / 2
        # Everything independent of the direction, computed once.
        usq_term = one(T) - T(HALF_INV_CS2) * (ux * ux + uy * uy + uz * uz)
        Base.Cartesian.@nexprs 27 s -> begin
            cv_s = cube_velocity(s)
            cx_s = T(cv_s[1])
            cy_s = T(cv_s[2])
            cz_s = T(cv_s[3])
            w_s = cube_weight(T, s)
            cu_s = cx_s * ux + cy_s * uy + cz_s * uz

            feq_s = w_s * ρ * (usq_term + T(INV_CS2) * cu_s + T(HALF_INV_CS4) * cu_s * cu_s)
            @inbounds fq_s = buf[s] - ω * (buf[s] - feq_s)

            sx_s = T(INV_CS2) * (cx_s - ux) + T(INV_CS4) * cu_s * cx_s
            sy_s = T(INV_CS2) * (cy_s - uy) + T(INV_CS4) * cu_s * cy_s
            sz_s = T(INV_CS2) * (cz_s - uz) + T(INV_CS4) * cu_s * cz_s
            @inbounds buf[s] = fq_s + pre * w_s *
                               (sx_s * force[1] + sy_s * force[2] + sz_s * force[3])
        end
    end
    return buf
end

"""
    aa_step_node!(g, buf, even, i, j, k, nx, ny, nz, τ, force, operator, ωb, ωh)

One AA-pattern time step for a single node: gather, collide, scatter.
"""
@inline function aa_step_node!(g, buf, even::Bool, i::Int, j::Int, k::Int,
                               nx::Int, ny::Int, nz::Int, τ::T, force::NTuple{3,T},
                               operator::Val, ωb::T, ωh::T) where {T}
    aa_gather!(buf, g, even, i, j, k, nx, ny, nz)
    collide_buffer!(buf, τ, force, operator, ωb, ωh)
    aa_scatter!(g, buf, even, i, j, k, nx, ny, nz)
    return nothing
end

"""
    aa_run!(g, nsteps, τ; force, operator, omega_bulk, omega_higher)

Reference CPU driver for the AA-pattern, mainly so the device path can be
checked against the two-lattice implementation. `nsteps` must be even, which
leaves the array in normal orientation.
"""
function aa_run!(g::Array{T,4}, nsteps::Integer, τ::Real;
                 force::NTuple{3,<:Real} = (0, 0, 0), operator::Symbol = :central_moment,
                 omega_bulk::Real = 1.0, omega_higher::Real = 1.0) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    buf = Vector{T}(undef, 27)
    F = T.(force)
    op = Val(operator)
    for n in 1:nsteps
        even = isodd(n)          # the first step of each pair is the "even" pattern
        for k in 1:nz, j in 1:ny, i in 1:nx
            aa_step_node!(g, buf, even, i, j, k, nx, ny, nz, T(τ), F,
                          op, T(omega_bulk), T(omega_higher))
        end
    end
    return g
end
