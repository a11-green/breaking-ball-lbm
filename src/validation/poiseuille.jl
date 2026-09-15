"""
Force-driven plane Poiseuille flow — the analytic case for the wall treatment.

Walls perpendicular to y at arbitrary, deliberately non-half-integer positions.
A wall that does not fall midway between two nodes is what separates plain
bounce-back (which always places it midway) from the interpolated rule, so the
steady profile measures whether δ is honoured.

    u(y) = G (y - y_lo)(y_hi - y) / (2 ρ ν)

Everything is in lattice units; node `j` sits at `y = j`.
"""

struct PoiseuilleChannel{T<:AbstractFloat}
    y_lo::T
    y_hi::T
    force::T
    ν::T
    ρ0::T
end

"""
    PoiseuilleChannel(; y_lo, y_hi, force, ν, ρ0 = 1.0)

Channel bounded by walls at `y_lo` and `y_hi`, driven by body force `force`.
"""
function PoiseuilleChannel(; y_lo::Real, y_hi::Real, force::Real, ν::Real, ρ0::Real = 1.0)
    y_hi > y_lo || throw(ArgumentError("y_hi must exceed y_lo"))
    T = promote_type(typeof(float(y_lo)), typeof(float(y_hi)), typeof(float(force)),
                     typeof(float(ν)), typeof(float(ρ0)))
    return PoiseuilleChannel{T}(T(y_lo), T(y_hi), T(force), T(ν), T(ρ0))
end

"""Channel width."""
width(ch::PoiseuilleChannel) = ch.y_hi - ch.y_lo

"""Analytic streamwise velocity at height `y`."""
@inline function poiseuille_velocity(ch::PoiseuilleChannel{T}, y::Real) where {T}
    return ch.force * (T(y) - ch.y_lo) * (ch.y_hi - T(y)) / (2 * ch.ρ0 * ch.ν)
end

"""Peak velocity, at mid-channel."""
poiseuille_peak(ch::PoiseuilleChannel) = poiseuille_velocity(ch, (ch.y_lo + ch.y_hi) / 2)

"""
    channel_sdf(ch, dims)

Signed-distance field of the fluid slab: positive inside the channel.
"""
function channel_sdf(ch::PoiseuilleChannel{T}, dims::NTuple{3,<:Integer}) where {T}
    ϕ = Array{T,3}(undef, dims)
    @inbounds for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        ϕ[i, j, k] = min(T(j) - ch.y_lo, ch.y_hi - T(j))
    end
    return ϕ
end
