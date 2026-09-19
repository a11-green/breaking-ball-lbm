"""
Everything a pitch viewer draws, computed here rather than in the viewer.

The viewer needs a window, a graphics driver and a package that is not a
dependency of this one; none of that can be tested. So the viewer is left as a
thin script over this, and what is here is held to the same standard as the
solver: the reference trajectories are the ones `pitch_metrics` measures against,
and the tests check that the picture and the number agree.

**The three reference trajectories are the point of the figure.** Break is a
difference between two trajectories (see `trajectory.jl`), and which one is
subtracted changes the answer by a factor approaching two — the WBC sweeper is
15 inches of horizontal movement under the `pfx` convention and 27 under
`induced`. In a table that reads as a caveat. Drawn side by side against the
pitch they are subtracted from, it is a picture of what the conventions mean.
"""

"""A trajectory in the shape a plot wants: columns, not a vector of states."""
struct TrajectorySamples{T<:AbstractFloat}
    t::Vector{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    speed::Vector{T}
    q::Vector{Quat{T}}
    spin::Vector{NTuple{3,T}}
end

Base.length(s::TrajectorySamples) = length(s.t)

"""
    samples(states)

Columns from a trajectory. Position is in metres, in the frame of `body.jl`:
`x` toward the plate, `z` up, `y` to the pitcher's left.
"""
function samples(states::Vector{BallState{T}}) where {T}
    n = length(states)
    s = TrajectorySamples{T}(Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
                             Vector{T}(undef, n), Vector{T}(undef, n),
                             Vector{Quat{T}}(undef, n), Vector{NTuple{3,T}}(undef, n))
    for (i, b) in enumerate(states)
        s.t[i] = b.t
        s.x[i] = b.x[1]; s.y[i] = b.x[2]; s.z[i] = b.x[3]
        s.speed[i] = speed(b)
        s.q[i] = b.q
        s.spin[i] = b.ω
    end
    return s
end

"""
    break_references(state, props, aero; dt, distance, gravity)

The pitch and the three trajectories its movement is measured against.

  * `pfx` branches off the pitch forty feet from the plate, inheriting its
    position *and velocity* there, with the Magnus force switched off. It is
    what a published `pfx_x` means.
  * `induced` branches at release, same Magnus-free reference, so it is the
    whole effect of the spin over the whole flight.
  * `ballistic` is a straight line from the release point at the release
    velocity — no gravity, no drag, no spin.

Drawn together with the pitch they all start from, the gaps at the plate *are*
the three definitions.
"""
function break_references(s0::BallState{T}, p::BallProperties{T},
                          aero::CoefficientAero{T};
                          dt::Real = 1.0e-4, distance::Real = PLATE_DISTANCE,
                          gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    # Trajectories end *exactly* at the plate, not at the first sample past it.
    # `pitch_metrics` bisects the last sub-step to land on it, so a drawing that
    # stopped a few millimetres beyond would show a gap that disagreed with the
    # number in the table — by a part in ten thousand, which is small and would
    # be nobody's friend to discover later.
    function fly(state, model)
        traj = simulate_trajectory(state, p, model; dt = dt, distance = distance,
                                   gravity = gravity)
        traj[end] = state_at_distance(traj, p, model, distance; gravity = gravity)
        return traj
    end
    plain = no_magnus(aero)

    actual = fly(s0, aero)
    induced = fly(s0, plain)

    at40 = state_at_distance(actual, p, aero, T(distance) - T(PFX_SEGMENT);
                             gravity = gravity)
    pfx = fly(at40, plain)

    # The ballistic reference is a straight line, so it is written out rather
    # than integrated — and sampled at the same times as the pitch so the two
    # can be scrubbed together.
    ballistic = [BallState{T}(s0.x .+ b.t .* s0.v, s0.v, s0.ω, s0.q, b.t) for b in actual]

    return (actual = samples(actual), pfx = samples(pfx),
            induced = samples(induced), ballistic = samples(ballistic))
end

"""
    seam_world(geom, q, centre; samples, scale)

The seam curve placed in the world: rotated into the ball's current orientation
and moved to where the ball is.

This is what makes an animation show anything. A sphere looks identical at every
orientation, so without the seam a spinning ball is a still picture — and the
seam's orientation relative to the airflow is the whole subject (§4.1).
"""
function seam_world(geom::BaseballGeometry{T}, q::Quat{T}, centre::NTuple{3,<:Real};
                    samples::Integer = 256, scale::Real = 1.0) where {T}
    line = seam_polyline(geom.seam, samples)
    c = T.(centre)
    k = T(scale)
    return [c .+ k .* rotate(q, p) for p in line]
end

"""
    spin_axis_world(state, length)
    spin_axis_world(state, centre, length)

The spin axis through the ball's centre, as two endpoints — the other thing a
still picture cannot show.

The second form places it at `centre` instead of at `state.x`. The CFD run needs
that one: inside the ball-following box the ball sits wherever the lattice puts
it and stays there, while `state.x` is marching down the world trajectory, so
the two are different points and only the first is where the flow is.
"""
function spin_axis_world(b::BallState{T}, centre::NTuple{3,<:Real}, len::Real) where {T}
    c = T.(centre)
    n = sqrt(sum(abs2, b.ω))
    n == 0 && return (c, c)
    d = (T(len) / 2 / n) .* b.ω
    return (c .- d, c .+ d)
end

spin_axis_world(b::BallState, len::Real) = spin_axis_world(b, b.x, len)

"""
Named pitches, as a spin axis and a release speed.

The axes follow `body.jl`: backspin is `ω` along −y, so a four-seam fastball
carries `(0, −1, 0)`; a sweeper from a right-hander breaks toward +y and so
spins about +z; a gyroball spins about the flight direction and, by
construction, has no Magnus force at release.
"""
const PITCH_TYPES = (
    (name = "4-seam fastball", axis = (0.0, -1.0, 0.0), rpm = 2400.0, speed = 42.0),
    (name = "2-seam / sinker", axis = (0.25, -0.85, -0.3), rpm = 2200.0, speed = 41.0),
    (name = "sweeper", axis = (0.15, 0.0, 1.0), rpm = 2708.0, speed = 38.99),
    (name = "gyroball", axis = (1.0, 0.0, 0.0), rpm = 2400.0, speed = 40.0),
    (name = "12-6 curve", axis = (0.0, 1.0, 0.0), rpm = 2600.0, speed = 35.0),
)

"""
    pitch_family(; types, release, props, aero, dt, distance)

One trajectory per named pitch, with its metrics — the bundle a viewer draws as
a fan of curves.
"""
function pitch_family(; types = PITCH_TYPES,
                      release::NTuple{3,<:Real} = (2.0, 0.0, 1.75),
                      props::BallProperties{T} = BaseballProperties(),
                      aero::CoefficientAero{T} = CoefficientAero(BaseballProperties()),
                      dt::Real = 1.0e-4, distance::Real = PLATE_DISTANCE) where {T}
    map(types) do spec
        s0 = BallState(T; position = release, velocity = (spec.speed, 0.0, 0.0),
                       spin = spin_from_rpm(spec.axis, spec.rpm))
        traj = simulate_trajectory(s0, props, aero; dt = dt, distance = distance)
        (name = spec.name, spec = spec, samples = samples(traj),
         metrics = pitch_metrics(s0, props, aero; dt = dt, distance = distance))
    end
end

"""
    coefficient_series(traj, props, aero; ρ)

`C_D`, `C_L` and `C_side` along a trajectory, decomposed on `(û, n̂, b̂)` as
[`aerodynamic_coefficients`](@ref) does.

For the analytic model the first two are the model's own inputs and the third is
zero by construction — which is worth drawing anyway, because it is the line a
CFD trajectory will not leave flat, and having the axes already there is half
the point of building the viewer before the physics.
"""
function coefficient_series(traj::Vector{BallState{T}}, p::BallProperties{T}, aero;
                            ρ::Real = AIR_DENSITY) where {T}
    n = length(traj)
    cd = Vector{T}(undef, n); cl = Vector{T}(undef, n); cs = Vector{T}(undef, n)
    for (i, b) in enumerate(traj)
        F, _ = aero(b)
        cd[i], cl[i], cs[i] = aerodynamic_coefficients(F, b, p; ρ = ρ)
    end
    return (CD = cd, CL = cl, Cside = cs)
end

"""
    at_time(s, t)

The index of the sample nearest `t`, for a viewer scrubbing a slider. The
trajectory is sampled at a fixed step, so this is arithmetic rather than a
search.
"""
function at_time(s::TrajectorySamples{T}, t::Real) where {T}
    n = length(s)
    n <= 1 && return 1
    step = (s.t[end] - s.t[1]) / (n - 1)
    step <= 0 && return 1
    return clamp(round(Int, (T(t) - s.t[1]) / step) + 1, 1, n)
end

"""
    plate_box()

The strike zone as a closed rectangle at the plate, in world coordinates: the
only thing in the picture with a fixed, agreed size, so it is what gives the
break a scale a reader already has.
"""
function plate_box(; distance::Real = PLATE_DISTANCE, half_width::Real = 0.2159,
                   bottom::Real = 0.4699, top::Real = 1.0668)
    x = distance
    return [(x, -half_width, bottom), (x, half_width, bottom),
            (x, half_width, top), (x, -half_width, top), (x, -half_width, bottom)]
end
