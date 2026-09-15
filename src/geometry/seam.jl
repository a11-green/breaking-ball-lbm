"""
Parametric model of the baseball seam.

No open STL/CAD of an official ball was available (see `docs/design/DESIGN.md`
§4.1), so the seam is generated analytically. The curve used here is

    z/R = A·sin(2u),   φ = u,   u ∈ [0, 2π)

i.e. the latitude swings twice per azimuthal revolution. Despite its simplicity
this reproduces the properties that matter aerodynamically:

  * it is a single closed curve on the sphere with no self-intersection;
  * it splits the sphere into two *directly congruent* two-lobed ("peanut")
    pieces, swapped by a 180° rotation about the x axis — the way a real cover
    is cut;
  * spinning about z presents **four** seam crossings per revolution to the
    oncoming flow, while spinning about x or y presents **two** — the geometric
    content of "four-seam" versus "two-seam" (see `test/test_geometry.jl`).

`amplitude` (A, the highest |z|/R the seam reaches) is the shape knob to
calibrate against photographs; everything downstream reads it from here.
"""

struct BaseballSeam{T<:AbstractFloat}
    radius::T
    amplitude::T
end

"""
    BaseballSeam(; radius = 0.0374, amplitude = 0.7)

Seam on a ball of `radius` metres. The default radius is half of the 0.0748 m
reference diameter used throughout the project.
"""
function BaseballSeam(; radius::Real = 0.0374, amplitude::Real = 0.7)
    0 < amplitude < 1 || throw(ArgumentError("amplitude must lie in (0, 1), got $amplitude"))
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))
    T = promote_type(typeof(float(radius)), typeof(float(amplitude)))
    return BaseballSeam{T}(T(radius), T(amplitude))
end

"""
    seam_point(seam, u) -> (x, y, z)

Point on the seam at curve parameter `u ∈ [0, 2π)`, in the ball-fixed frame.
"""
@inline function seam_point(seam::BaseballSeam{T}, u::Real) where {T}
    cosθ = seam.amplitude * sin(2 * T(u))
    sinθ = sqrt(one(T) - cosθ * cosθ)
    return (seam.radius * sinθ * cos(T(u)),
            seam.radius * sinθ * sin(T(u)),
            seam.radius * cosθ)
end

"""
    seam_polyline(seam, n) -> Vector{NTuple{3,T}}

`n` samples of the closed seam, with the first point repeated at the end so the
result can be walked as `n` segments.
"""
function seam_polyline(seam::BaseballSeam{T}, n::Integer = 2048) where {T}
    n > 2 || throw(ArgumentError("need at least 3 samples, got $n"))
    pts = Vector{NTuple{3,T}}(undef, n + 1)
    @inbounds for i in 1:n
        pts[i] = seam_point(seam, 2π * (i - 1) / n)
    end
    @inbounds pts[n+1] = pts[1]
    return pts
end

"""
    seam_length(seam; n = 4096)

Arc length of the seam, from an `n`-segment polyline approximation.
"""
function seam_length(seam::BaseballSeam{T}; n::Integer = 4096) where {T}
    pts = seam_polyline(seam, n)
    total = zero(T)
    @inbounds for i in 1:n
        dx = pts[i+1][1] - pts[i][1]
        dy = pts[i+1][2] - pts[i][2]
        dz = pts[i+1][3] - pts[i][3]
        total += sqrt(dx * dx + dy * dy + dz * dz)
    end
    return total
end

"""
    distance_to_segment(p, a, b)

Euclidean distance from point `p` to the segment `a`–`b`.
"""
@inline function distance_to_segment(p::NTuple{3,T}, a::NTuple{3,T}, b::NTuple{3,T}) where {T}
    abx, aby, abz = b[1] - a[1], b[2] - a[2], b[3] - a[3]
    apx, apy, apz = p[1] - a[1], p[2] - a[2], p[3] - a[3]
    len2 = abx * abx + aby * aby + abz * abz
    t = len2 > 0 ? clamp((apx * abx + apy * aby + apz * abz) / len2, zero(T), one(T)) : zero(T)
    dx = apx - t * abx
    dy = apy - t * aby
    dz = apz - t * abz
    return sqrt(dx * dx + dy * dy + dz * dz)
end

"""
    distance_to_seam(p, polyline)

Distance from `p` to the seam curve sampled as `polyline`.
"""
function distance_to_seam(p::NTuple{3,T}, polyline::Vector{NTuple{3,T}}) where {T}
    d = T(Inf)
    @inbounds for i in 1:(length(polyline)-1)
        d = min(d, distance_to_segment(p, polyline[i], polyline[i+1]))
    end
    return d
end
