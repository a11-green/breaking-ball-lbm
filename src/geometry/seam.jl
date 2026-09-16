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

"""
    seam_distance(seam, p; window, scan, iterations)

Distance from `p` to the seam curve, found by searching a window of the curve
parameter rather than walking the whole polyline.

**Why this exists.** `distance_to_seam` costs one segment evaluation per
polyline sample — two thousand of them — and the geometry has to be re-cut every
dozen or so steps as the ball turns (`geometry/rotating.jl`). At that rate the
exhaustive search costs more than the flow solver by an order of magnitude, so it
stops being a reference implementation and starts being the reason the run is
impossible.

**Why a window is enough.** The parameterisation is `φ = u`: the curve passes
through every azimuth exactly once, so the azimuth of the query point is a seed
for the parameter of its nearest curve point, and points at a different azimuth
`Δφ` away are at least `R√(1−A²)·Δφ` distant — for the default amplitude, seven
tenths of a radius per radian. A window of ±0.6 rad therefore cannot exclude the
true nearest point unless that point is already many ridge heights away, in
which case the answer is "outside the seam" either way.

That is an argument, not a proof, so `test/test_geometry.jl` checks it: over
random points near the surface the windowed search agrees with the exhaustive
one wherever the exhaustive one says the point is anywhere near the seam, and
never returns less (a restricted minimum cannot).

The window is scanned coarsely first, because `|p − c(u)|²` need not be unimodal
across the whole of it, then the best bracket is closed by golden section.
"""
function seam_distance(seam::BaseballSeam{T}, p::NTuple{3,<:Real};
                       window::Real = 0.6, scan::Integer = 8,
                       iterations::Integer = 30) where {T}
    px, py, pz = T(p[1]), T(p[2]), T(p[3])
    d2(u) = begin
        q = seam_point(seam, u)
        (q[1] - px)^2 + (q[2] - py)^2 + (q[3] - pz)^2
    end

    u0 = atan(py, px)
    w = T(window)
    lo, hi = u0 - w, u0 + w
    step = (hi - lo) / scan

    # Coarse scan, keeping the bracket either side of the best sample.
    best = lo
    fbest = d2(lo)
    for m in 1:scan
        u = lo + m * step
        f = d2(u)
        if f < fbest
            fbest = f
            best = u
        end
    end
    a, b = max(best - step, lo), min(best + step, hi)

    # Golden section: no derivatives, and it cannot leave the bracket.
    invφ = T(0.6180339887498949)
    c, d = b - invφ * (b - a), a + invφ * (b - a)
    fc, fd = d2(c), d2(d)
    for _ in 1:iterations
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invφ * (b - a)
            fc = d2(c)
        else
            a, c, fc = c, d, fd
            d = a + invφ * (b - a)
            fd = d2(d)
        end
    end
    return sqrt(min(fbest, fc, fd))
end
