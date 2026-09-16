"""
Integrating a pitch from release to the plate, and measuring how much it moved.

Break is a *difference between two trajectories*, never a property of one, so
everything here runs the pitch more than once and subtracts. Getting the
reference trajectory right is most of the definition, and there are three worth
having — they differ by factors of two, which is larger than the 20% tolerance
V&V-5 is trying to test:

  * **`pfx`** is the PITCHf/x and Statcast convention, and the one a published
    `pfx_x`/`pfx_z` means. The reference branches off the actual pitch **40 ft
    from the plate**, inheriting its position *and velocity* there, with the
    Magnus force switched off. Only the movement over that last 40 ft counts —
    roughly half of the flight, and since break grows like the square of the
    remaining time, roughly *one quarter* of the whole-flight figure before drag
    is accounted for.
  * **`induced`** uses the same no-Magnus reference but branches at release, so
    it is the whole effect of the spin over the whole flight. Around twice
    `pfx`.
  * **`total`** subtracts a straight line from the release point at the release
    velocity — no gravity, no drag, no spin. Its vertical component is the full
    drop a catcher sees, about a metre, most of it gravity.

Which of the three a published number refers to has to be established before it
can be compared against (§11); the metrics carry all three so the comparison can
be made once that is settled.

Every comparison happens where the trajectories cross the plate, not at equal
times: the pitches arrive at slightly different moments and it is the position
at the plate that a hitter sees.
"""

const PLATE_DISTANCE = 18.4404      # m, 60 ft 6 in from the rubber
const PFX_SEGMENT = 12.192          # m, the 40 ft over which pfx_x/pfx_z are defined
const INCH = 0.0254

"""
    simulate_trajectory(state, props, aero; dt, distance, gravity, maxsteps)

RK4 from release until the ball has passed `distance` in x. Returns the whole
sampled trajectory; the last state is past the target, and
[`state_at_distance`](@ref) lands exactly on it.
"""
function simulate_trajectory(s0::BallState{T}, p::BallProperties{T}, aero;
                             dt::Real = 1.0e-4, distance::Real = PLATE_DISTANCE,
                             gravity::NTuple{3,<:Real} = GRAVITY,
                             maxsteps::Integer = 1_000_000) where {T}
    s0.v[1] > 0 || throw(ArgumentError("the ball must be thrown toward the plate (v[1] > 0)"))
    xt = T(distance)
    traj = [s0]
    s = s0
    for _ in 1:maxsteps
        s.x[1] >= xt && return traj
        s = advance(s, p, aero, dt; gravity = gravity)
        push!(traj, s)
    end
    throw(ErrorException("the ball did not reach x = $distance in $maxsteps steps"))
end

"""
    state_at_distance(traj, props, aero, distance; gravity)

The state exactly where the trajectory crosses `x = distance`, found by
bisecting the sub-step rather than interpolating between samples — the same
integrator, just with a shorter final step.
"""
function state_at_distance(traj::Vector{BallState{T}}, p::BallProperties{T}, aero,
                           distance::Real; gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    xt = T(distance)
    i = findfirst(s -> s.x[1] >= xt, traj)
    i === nothing && throw(ArgumentError("the trajectory never reaches x = $distance"))
    i == 1 && return traj[1]

    s = traj[i-1]
    h = traj[i].t - s.t
    lo, hi = zero(T), T(h)
    for _ in 1:60
        mid = (lo + hi) / 2
        if advance(s, p, aero, mid; gravity = gravity).x[1] < xt
            lo = mid
        else
            hi = mid
        end
    end
    return advance(s, p, aero, (lo + hi) / 2; gravity = gravity)
end

"""
Movement of one pitch, in metres, with the flight it took to get there.

Horizontal components are along +y (the pitcher's left) and vertical along +z.
Statcast reports `pfx_x` from the catcher's point of view, so its sign is the
negative of `pfx_horizontal` here; `pfx_z` matches `pfx_vertical` directly.
"""
struct PitchMetrics{T<:AbstractFloat}
    flight_time::T
    release_speed::T
    plate_speed::T
    plate_position::NTuple{3,T}
    pfx_horizontal::T
    pfx_vertical::T
    induced_horizontal::T
    induced_vertical::T
    total_horizontal::T
    total_vertical::T
    spin_parameter::T
end

"""
    pitch_metrics(state, props, aero; dt, distance, gravity)

Run the pitch, run its three references, and report the differences at the
plate. See the module docstring for what each reference is and why they differ
so much.
"""
function pitch_metrics(s0::BallState{T}, p::BallProperties{T}, aero::CoefficientAero{T};
                       dt::Real = 1.0e-4, distance::Real = PLATE_DISTANCE,
                       gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    fly(state, model) = simulate_trajectory(state, p, model; dt = dt, distance = distance,
                                            gravity = gravity)
    at_plate(state, model) = state_at_distance(fly(state, model), p, model, distance;
                                               gravity = gravity)

    traj = fly(s0, aero)
    actual = state_at_distance(traj, p, aero, distance; gravity = gravity)
    spinless = at_plate(s0, no_magnus(aero))

    # The pfx reference inherits the actual pitch's state 40 ft out, so the
    # movement it has already accumulated is subtracted away rather than
    # counted twice.
    x40 = T(distance) - T(PFX_SEGMENT)
    at40 = state_at_distance(traj, p, aero, x40; gravity = gravity)
    pfx_ref = at_plate(at40, no_magnus(aero))

    # The ballistic reference is a straight line, so it is written out rather
    # than integrated: no forces at all, not even gravity.
    tb = (T(distance) - s0.x[1]) / s0.v[1]
    ballistic = s0.x .+ tb .* s0.v

    return PitchMetrics{T}(actual.t - s0.t, speed(s0), speed(actual), actual.x,
                           actual.x[2] - pfx_ref.x[2],
                           actual.x[3] - pfx_ref.x[3],
                           actual.x[2] - spinless.x[2],
                           actual.x[3] - spinless.x[3],
                           actual.x[2] - ballistic[2],
                           actual.x[3] - ballistic[3],
                           spin_parameter(s0, p))
end

"""Metres to inches, since every published break number is in inches."""
inches(x::Real) = x / INCH

function Base.show(io::IO, ::MIME"text/plain", m::PitchMetrics)
    r(x) = round(inches(x), digits = 1)
    println(io, "PitchMetrics")
    println(io, "  flight time        ", round(m.flight_time, digits = 4), " s")
    println(io, "  speed              ", round(m.release_speed, digits = 2), " → ",
            round(m.plate_speed, digits = 2), " m/s")
    println(io, "  plate position     ", map(x -> round(x, digits = 3), m.plate_position))
    println(io, "  spin parameter S   ", round(m.spin_parameter, digits = 4))
    println(io, "  pfx (last 40 ft)   ", r(m.pfx_horizontal), "\" horizontal, ",
            r(m.pfx_vertical), "\" vertical")
    println(io, "  induced (release)  ", r(m.induced_horizontal), "\" horizontal, ",
            r(m.induced_vertical), "\" vertical")
    print(io,   "  total movement     ", r(m.total_horizontal), "\" horizontal, ",
            r(m.total_vertical), "\" vertical")
end
