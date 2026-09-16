"""
Signed-distance representation of the ball.

The solid is the union of the smooth sphere and a tube of radius `seam_height`
threaded along the seam curve. Because the curve lies on the sphere surface, the
tube protrudes exactly `seam_height` above it and is buried to the same depth
below — the raised ridge that trips the boundary layer.

Distances are in metres and negative inside the solid. `sdf` returns
`min(sphere, tube)`, which is the exact distance to the union outside the solid
(what the boundary treatment needs) and a conservative bound inside it.
"""

struct BaseballGeometry{T<:AbstractFloat}
    radius::T
    seam_height::T
    seam::BaseballSeam{T}
    polyline::Vector{NTuple{3,T}}
end

"""
    BaseballGeometry(; diameter = 0.0748, seam_height = 0.00079,
                       seam_amplitude = 0.7, seam_samples = 2048)

Ball geometry with the reference MLB dimensions: 74.8 mm across and a seam
standing 0.79 mm proud of the cover (≈ D/92, the measured mean).
"""
function BaseballGeometry(; diameter::Real = 0.0748, seam_height::Real = 0.00079,
                          seam_amplitude::Real = 0.7, seam_samples::Integer = 2048)
    diameter > 0 || throw(ArgumentError("diameter must be positive, got $diameter"))
    seam_height >= 0 || throw(ArgumentError("seam_height must be non-negative, got $seam_height"))
    T = promote_type(typeof(float(diameter)), typeof(float(seam_height)),
                     typeof(float(seam_amplitude)))
    radius = T(diameter) / 2
    seam = BaseballSeam(radius = radius, amplitude = seam_amplitude)
    return BaseballGeometry{T}(radius, T(seam_height), seam, seam_polyline(seam, seam_samples))
end

"""Signed distance to the smooth sphere alone."""
@inline function sphere_sdf(geom::BaseballGeometry{T}, p::NTuple{3,T}) where {T}
    return sqrt(p[1]^2 + p[2]^2 + p[3]^2) - geom.radius
end

"""
    sdf(geom, p)

Signed distance from `p` (ball-fixed frame, metres) to the seamed ball.

The seam term comes from [`seam_distance`](@ref), which searches a window of the
curve parameter seeded by the azimuth rather than walking the polyline. That is
what makes re-cutting the geometry as the ball turns affordable at all
(`geometry/rotating.jl`); [`sdf_exhaustive`](@ref) keeps the polyline version as
the reference the fast one is tested against.
"""
function sdf(geom::BaseballGeometry{T}, p::NTuple{3,T}) where {T}
    ds = sphere_sdf(geom, p)
    geom.seam_height > 0 || return ds
    return min(ds, seam_distance(geom.seam, p) - geom.seam_height)
end

"""The same distance from the sampled polyline, walked in full — the reference."""
function sdf_exhaustive(geom::BaseballGeometry{T}, p::NTuple{3,T}) where {T}
    ds = sphere_sdf(geom, p)
    geom.seam_height > 0 || return ds
    return min(ds, distance_to_seam(p, geom.polyline) - geom.seam_height)
end

sdf(geom::BaseballGeometry{T}, x::Real, y::Real, z::Real) where {T} =
    sdf(geom, (T(x), T(y), T(z)))

"""
    sdf_field!(ϕ, geom, dx; center, band)

Fill `ϕ[i, j, k]` with the signed distance at node position
`((i, j, k) .- center) .* dx`, in metres.

The seam is only queried for nodes within `band` of the sphere surface; farther
out the sphere term dominates and the seam can move the result by at most
`seam_height`, which is sub-cell for any grid that resolves the ridge at all.
`center` is given in (possibly fractional) node-index units.
"""
function sdf_field!(ϕ::Array{T,3}, geom::BaseballGeometry{T}, dx::Real;
                    center::NTuple{3,<:Real} = (size(ϕ) .+ 1) ./ 2,
                    band::Real = geom.seam_height + 3 * dx) where {T}
    nx, ny, nz = size(ϕ)
    Δ = T(dx)
    cx, cy, cz = T.(center)
    b = T(band)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        p = ((T(i) - cx) * Δ, (T(j) - cy) * Δ, (T(k) - cz) * Δ)
        ds = sphere_sdf(geom, p)
        ϕ[i, j, k] = abs(ds) <= b ? sdf(geom, p) : ds
    end
    return ϕ
end

"""
    sdf_field(geom, dims, dx; kwargs...)

Allocate and fill a signed-distance field of size `dims`.
"""
function sdf_field(geom::BaseballGeometry{T}, dims::NTuple{3,<:Integer}, dx::Real;
                   kwargs...) where {T}
    return sdf_field!(Array{T,3}(undef, dims), geom, dx; kwargs...)
end

"""
    solid_volume(ϕ, dx)

Volume of the region `ϕ < 0`, counting whole cells.
"""
solid_volume(ϕ::Array{T,3}, dx::Real) where {T} = count(<(0), ϕ) * T(dx)^3
