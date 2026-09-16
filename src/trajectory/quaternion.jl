"""
Just enough quaternion algebra to carry the ball's orientation.

The seam is fixed to the ball, so the phase of the seam relative to the airflow
is part of the state, not a constant — it is the whole reason a two-seam and a
four-seam grip behave differently (§4.1). Euler angles would gimbal-lock on a
spin axis that tilts toward the flight direction, which is exactly what a gyro
component is, so the orientation is a unit quaternion.

`Quat` maps **body coordinates to world coordinates**, and `ω` is kept in world
coordinates throughout. For a sphere the inertia tensor is isotropic, so `Iω` is
parallel to `ω`, the gyroscopic term `ω × Iω` vanishes, and there is no reason
to work in the body frame at all.
"""

struct Quat{T<:AbstractFloat}
    w::T
    x::T
    y::T
    z::T
end

"""The identity rotation."""
Base.one(::Type{Quat{T}}) where {T} = Quat{T}(1, 0, 0, 0)
Base.one(::Quat{T}) where {T} = one(Quat{T})

Base.:*(a::Quat{T}, b::Quat{T}) where {T} = Quat{T}(
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w)

Base.:*(s::Real, q::Quat{T}) where {T} = Quat{T}(s * q.w, s * q.x, s * q.y, s * q.z)
Base.:+(a::Quat{T}, b::Quat{T}) where {T} = Quat{T}(a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z)
Base.conj(q::Quat{T}) where {T} = Quat{T}(q.w, -q.x, -q.y, -q.z)
Base.abs(q::Quat) = sqrt(q.w^2 + q.x^2 + q.y^2 + q.z^2)

"""Renormalise. Called every step: RK4 leaves the norm off by O(dt⁵) and it accumulates."""
function normalize(q::Quat{T}) where {T}
    n = abs(q)
    n == 0 && return one(Quat{T})
    return Quat{T}(q.w / n, q.x / n, q.y / n, q.z / n)
end

"""Rotation of `angle` radians about `axis` (need not be normalised)."""
function quat_from_axis_angle(axis::NTuple{3,<:Real}, angle::Real)
    T = float(promote_type(eltype(axis), typeof(angle)))
    n = sqrt(T(axis[1])^2 + T(axis[2])^2 + T(axis[3])^2)
    n == 0 && return one(Quat{T})
    s, c = sincos(T(angle) / 2)
    return Quat{T}(c, s * T(axis[1]) / n, s * T(axis[2]) / n, s * T(axis[3]) / n)
end

"""Apply the rotation to a vector: `q v q*`, written out."""
function rotate(q::Quat{T}, v::NTuple{3,<:Real}) where {T}
    vx, vy, vz = T(v[1]), T(v[2]), T(v[3])
    # t = 2 (q_vec × v); result = v + q_w t + q_vec × t
    tx = 2 * (q.y * vz - q.z * vy)
    ty = 2 * (q.z * vx - q.x * vz)
    tz = 2 * (q.x * vy - q.y * vx)
    return (vx + q.w * tx + q.y * tz - q.z * ty,
            vy + q.w * ty + q.z * tx - q.x * tz,
            vz + q.w * tz + q.x * ty - q.y * tx)
end

"""The inverse rotation, world to body."""
unrotate(q::Quat, v::NTuple{3,<:Real}) = rotate(conj(q), v)

"""
    quat_rate(q, ω)

`dq/dt` for a body spinning at world-frame angular velocity `ω`: `½ ω ⊗ q`.
"""
function quat_rate(q::Quat{T}, ω::NTuple{3,<:Real}) where {T}
    ωq = Quat{T}(0, ω[1], ω[2], ω[3])
    return T(0.5) * (ωq * q)
end

"""The 3×3 rotation matrix, for callers that would rather have one."""
function rotation_matrix(q::Quat{T}) where {T}
    e1 = rotate(q, (one(T), zero(T), zero(T)))
    e2 = rotate(q, (zero(T), one(T), zero(T)))
    e3 = rotate(q, (zero(T), zero(T), one(T)))
    return T[e1[1] e2[1] e3[1]; e1[2] e2[2] e3[2]; e1[3] e2[3] e3[3]]
end
