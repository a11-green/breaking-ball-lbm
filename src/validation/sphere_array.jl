"""
Stokes flow through a simple cubic array of spheres.

A body force drives the fluid through a periodic box holding one sphere. Two
things can be checked against theory:

  * at steady state the force on the sphere must equal the total body force
    applied to the fluid — an exact momentum balance, independent of resolution;
  * the drag itself, against Hasimoto's solution for a cubic array, which
    corrects Stokes' 6πμaU for the periodic neighbours (they matter: the Stokes
    velocity field decays only as 1/r).
"""

"""
    sphere_sdf_field(dims, radius; center)

Signed-distance field of a sphere, in lattice units, positive outside.
"""
function sphere_sdf_field(::Type{T}, dims::NTuple{3,<:Integer}, radius::Real;
                          center::NTuple{3,<:Real} = (dims .+ 1) ./ 2) where {T}
    ϕ = Array{T,3}(undef, dims)
    cx, cy, cz = T.(center)
    r = T(radius)
    @inbounds for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        ϕ[i, j, k] = sqrt((T(i) - cx)^2 + (T(j) - cy)^2 + (T(k) - cz)^2) - r
    end
    return ϕ
end

sphere_sdf_field(dims::NTuple{3,<:Integer}, radius::Real; kwargs...) =
    sphere_sdf_field(Float64, dims, radius; kwargs...)

"""
    hasimoto_factor(radius, box)

Drag enhancement `K` of a simple cubic array over an isolated sphere, from
Hasimoto's series in the solid fraction `c = (4/3)πa³/L³`:

    F = 6πμaU · K,   K = 1 / (1 - 1.7601 c^{1/3} + c - 1.5593 c²)

with `U` the superficial (box-averaged) velocity.
"""
function hasimoto_factor(radius::Real, box::Real)
    c = 4 / 3 * π * radius^3 / box^3
    return 1 / (1 - 1.7601 * cbrt(c) + c - 1.5593 * c^2)
end

"""
    stokes_drag(radius, μ, u)

Drag on an isolated sphere in creeping flow, `6πμaU`.
"""
stokes_drag(radius::Real, μ::Real, u::Real) = 6π * μ * radius * u

"""
    superficial_velocity(s, solid, force)

Streamwise velocity averaged over the whole box, counting solid nodes as at
rest — the velocity Hasimoto's factor is defined against.
"""
function superficial_velocity(s::LBMState{T}, solid::AbstractArray{Bool,3},
                              force::NTuple{3,T}) where {T}
    total = zero(T)
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        solid[i, j, k] && continue
        _, ux, _, _ = fluid_velocity(s, i, j, k, force)
        total += ux
    end
    return total / (s.nx * s.ny * s.nz)
end
