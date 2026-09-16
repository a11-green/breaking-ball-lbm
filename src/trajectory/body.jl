"""
The rigid body and its six degrees of freedom (§1.5).

Coordinates are right-handed and fixed to the field:

    x  toward home plate (the direction of flight)
    z  up
    y  = z × x, i.e. to the pitcher's left as they face the plate

so gravity is `(0, 0, -g)` and a pitch that "runs in on a right-hander" moves
toward −y. This is *not* the Statcast convention (which points y from the plate
back toward the mound); the conversion lives in `metrics.jl`, in one place, so
the physics never has to think about it.

Angular velocity is in world coordinates and the orientation quaternion maps
body to world. A baseball's moment of inertia is close enough to a uniform
sphere's — 2/5 m R² is 8.1e-5 kg·m² at the default mass and diameter, against
the ≈8.0e-5 usually quoted — that the cork-and-yarn layering is not worth
modelling.
"""

struct BallProperties{T<:AbstractFloat}
    mass::T        # kg
    radius::T      # m
    inertia::T     # kg·m², isotropic
    area::T        # m², frontal
end

"""
    BaseballProperties(; mass = 0.145, diameter = 0.0748)

MLB-legal middle of the range (§1.1).
"""
function BaseballProperties(::Type{T} = Float64; mass::Real = 0.145,
                            diameter::Real = 0.0748) where {T<:AbstractFloat}
    r = T(diameter) / 2
    return BallProperties{T}(T(mass), r, T(0.4) * T(mass) * r^2, T(π) * r^2)
end

"""Position, velocity, angular velocity, orientation and time — the full state."""
struct BallState{T<:AbstractFloat}
    x::NTuple{3,T}
    v::NTuple{3,T}
    ω::NTuple{3,T}
    q::Quat{T}
    t::T
end

function BallState(::Type{T} = Float64; position = (0.0, 0.0, 1.8),
                   velocity = (39.0, 0.0, 0.0), spin = (0.0, 0.0, 0.0),
                   orientation::Quat = one(Quat{T}), time::Real = 0.0) where {T<:AbstractFloat}
    return BallState{T}(T.(Tuple(position)), T.(Tuple(velocity)), T.(Tuple(spin)),
                        Quat{T}(orientation.w, orientation.x, orientation.y, orientation.z),
                        T(time))
end

"""
    spin_from_rpm(axis, rpm)

Angular velocity vector from a spin axis and a rate in rpm, the two numbers
pitch-tracking actually reports.
"""
function spin_from_rpm(axis::NTuple{3,<:Real}, rpm::Real)
    T = float(promote_type(eltype(axis), typeof(rpm)))
    n = sqrt(sum(abs2, T.(axis)))
    n == 0 && return (zero(T), zero(T), zero(T))
    ω = 2 * T(π) * T(rpm) / 60
    return (ω * T(axis[1]) / n, ω * T(axis[2]) / n, ω * T(axis[3]) / n)
end

spin_rpm(s::BallState) = 60 * sqrt(sum(abs2, s.ω)) / (2π)
speed(s::BallState) = sqrt(sum(abs2, s.v))

"""
    spin_parameter(state, props)

`S = ω r / U`, the dimensionless spin of §1.2. Uses the whole angular velocity;
the component along the velocity (gyro spin) contributes no Magnus force but is
still part of the surface speed a boundary condition has to impose.
"""
spin_parameter(s::BallState, p::BallProperties) =
    sqrt(sum(abs2, s.ω)) * p.radius / max(speed(s), eps(typeof(p.radius)))

"""The part of the spin that actually makes Magnus force: ω perpendicular to V."""
function transverse_spin(s::BallState)
    u = speed(s)
    u == 0 && return s.ω
    û = s.v ./ u
    along = sum(s.ω .* û)
    return s.ω .- along .* û
end

# --- 6DOF integration ------------------------------------------------------
#
# `aero` is any callable taking a `BallState` and returning `(force, torque)` in
# newtons and newton-metres. During a coupled run it is the momentum-exchange
# sum from the CFD, frozen for the step; for testing the integrator it is an
# analytic coefficient model. Keeping it a callable is what lets the same RK4
# be exercised against a closed-form trajectory.

struct BallRates{T<:AbstractFloat}
    v::NTuple{3,T}
    a::NTuple{3,T}
    α::NTuple{3,T}
    dq::Quat{T}
end

function ball_rates(s::BallState{T}, p::BallProperties{T}, aero,
                    gravity::NTuple{3,T}) where {T}
    F, M = aero(s)
    return BallRates{T}(s.v,
                        (T(F[1]) / p.mass + gravity[1],
                         T(F[2]) / p.mass + gravity[2],
                         T(F[3]) / p.mass + gravity[3]),
                        (T(M[1]) / p.inertia, T(M[2]) / p.inertia, T(M[3]) / p.inertia),
                        quat_rate(s.q, s.ω))
end

@inline function _advanced(s::BallState{T}, r::BallRates{T}, h::T) where {T}
    return BallState{T}(s.x .+ h .* r.v, s.v .+ h .* r.a, s.ω .+ h .* r.α,
                        normalize(s.q + h * r.dq), s.t + h)
end

"""
    advance(state, props, aero, dt; gravity)

One classical RK4 step of the 6DOF equations.

With a frozen aerodynamic sample the translation and rotation are linear in
time and RK4 is exact for them; the work it does is on the quaternion, whose
rate depends on `ω`, and on the case where `aero` is an analytic function of the
state. Either way it costs four evaluations of a callable that is a table lookup
during a coupled run, so there is no reason to use anything cheaper.
"""
function advance(s::BallState{T}, p::BallProperties{T}, aero, dt::Real;
                 gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    h = T(dt)
    g = T.(gravity)
    k1 = ball_rates(s, p, aero, g)
    k2 = ball_rates(_advanced(s, k1, h / 2), p, aero, g)
    k3 = ball_rates(_advanced(s, k2, h / 2), p, aero, g)
    k4 = ball_rates(_advanced(s, k3, h), p, aero, g)

    w(a, b, c, d) = (a .+ 2 .* b .+ 2 .* c .+ d) ./ 6
    dq = T(1 / 6) * (k1.dq + T(2) * k2.dq + T(2) * k3.dq + k4.dq)
    return BallState{T}(s.x .+ h .* w(k1.v, k2.v, k3.v, k4.v),
                        s.v .+ h .* w(k1.a, k2.a, k3.a, k4.a),
                        s.ω .+ h .* w(k1.α, k2.α, k3.α, k4.α),
                        normalize(s.q + h * dq), s.t + h)
end
