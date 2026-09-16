"""
The ball-following non-inertial frame (§4.4.2).

The domain is a small box fixed to the ball's centre of mass, so the ball spins
in place and never translates across the grid. That is what makes a 18.4 m
flight affordable on one consumer GPU: the grid only ever has to resolve the
few diameters of air the ball is currently disturbing.

**What the frame costs, and why gravity does not survive it.** The frame
accelerates with the ball,

    A = dV/dt = F_aero/m + g,

so the fluid momentum equation picks up a uniform fictitious force −ρA. Real
gravity would add +ρg, but in the laboratory the far-field air is at rest in
hydrostatic balance, so that +ρg is cancelled by the pressure gradient holding
it up; carrying neither is the same as carrying both, to within buoyancy, which
for a baseball is 2.6 mN against a 1.42 N weight — 0.18%. So the uniform body
acceleration the fluid feels is exactly

    a_fluid = −A = −(F_aero/m + g),

and gravity enters the *flow* only through the ball's response to it.

**The consistency that makes this work.** The free-stream velocity in the frame
is `u∞ = −V(t)`, and

    d u∞/dt = −A = a_fluid,

i.e. the same uniform body force that represents the frame's acceleration also
carries the far field to exactly the velocity the next step's inflow condition
is going to ask for. Boundary and interior cannot drift apart, which is the
usual way a moving-frame scheme goes quietly wrong.
"""

"""
    frame_acceleration(F_aero, props; gravity)

Acceleration of the ball, and therefore of the frame, in m/s².
"""
function frame_acceleration(F_aero::NTuple{3,<:Real}, p::BallProperties{T};
                            gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    return (T(F_aero[1]) / p.mass + T(gravity[1]),
            T(F_aero[2]) / p.mass + T(gravity[2]),
            T(F_aero[3]) / p.mass + T(gravity[3]))
end

"""
    fluid_body_acceleration(F_aero, props; gravity)

The uniform body acceleration to apply to every fluid node, in m/s². Note the
sign and note that gravity appears with the *ball's* mass, not the air's: this
is the reaction to the frame's motion, not a gravitational term.
"""
function fluid_body_acceleration(F_aero::NTuple{3,<:Real}, p::BallProperties;
                                 gravity::NTuple{3,<:Real} = GRAVITY)
    return .-frame_acceleration(F_aero, p; gravity = gravity)
end

"""
    freestream_velocity(state)

Air velocity far from the ball as seen in the frame: `−V(t)`, since the air is
at rest in the laboratory. Turbulence or wind would be added to this.
"""
freestream_velocity(s::BallState) = .-s.v

"""
    lattice_body_force(units, F_aero, props; gravity)

The body acceleration in lattice units, ready to hand to the collision as a
Guo forcing term. Lattice density is 1, so acceleration and force density are
numerically the same thing.
"""
function lattice_body_force(u::LatticeUnits, F_aero::NTuple{3,<:Real},
                            p::BallProperties; gravity::NTuple{3,<:Real} = GRAVITY)
    return to_lattice_acceleration(u, fluid_body_acceleration(F_aero, p; gravity = gravity))
end

"""
    lattice_freestream(units, state)

Free-stream velocity in lattice units. The run is designed around
`units.lattice_speed`, and this drifts below it as the ball slows, so it also
serves as the Mach-number monitor of §5.1.
"""
lattice_freestream(u::LatticeUnits, s::BallState) =
    to_lattice_velocity(u, freestream_velocity(s))

"""
    lattice_spin(units, state)

Angular velocity in radians per lattice step, which is what the rotating
bounce-back condition `u_wall = ω × r` needs once `r` is measured in lattice
spacings.
"""
lattice_spin(u::LatticeUnits, s::BallState) = to_lattice_rate(u, s.ω)

"""
    seam_orientation(state)

The rotation taking seam geometry from its reference pose to the current one.
The wall geometry has to be rebuilt when this has moved far enough to matter;
[`orientation_drift`](@ref) says how far it has moved.
"""
seam_orientation(s::BallState) = s.q

"""
    orientation_drift(q0, q1, radius_nodes)

How far the surface has rotated between two orientations, in lattice spacings
at the equator. The wall field is worth rebuilding when this approaches the
sub-grid accuracy that interpolated bounce-back is buying — rebuilding every
step would cost a full-grid signed-distance evaluation for nothing, since a
2,700 rpm ball turns about 7e-4 rad in one step at production resolution.
"""
function orientation_drift(q0::Quat{T}, q1::Quat{T}, radius_nodes::Real) where {T}
    rel = q1 * conj(q0)
    c = clamp(abs(rel.w) / abs(rel), -one(T), one(T))
    return 2 * acos(c) * T(radius_nodes)
end
