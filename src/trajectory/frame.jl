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

"""
    aerodynamic_coefficients(force, state, props; ρ)

Decompose an aerodynamic force into `(C_D, C_L, C_side)` on the natural frame of
a spinning ball:

  * `û = V/|V|` — drag acts along `−û`, so `C_D` is positive for a ball that is
    being slowed;
  * `n̂ = (ω × û)/|ω × û|` — the Magnus direction, which by construction ignores
    whatever part of the spin lies along the flight path;
  * `b̂ = û × n̂` — perpendicular to both, so a force here is lift the Magnus
    effect cannot explain.

That third number is the point of the whole project. §1.4: a real ball develops
force out of the Magnus plane because the seam makes the pressure distribution
asymmetric about it, and that is why sweepers and gyroballs do not break the way
the textbook says. A coefficient model can only ever report zero there; the CFD
can report what it measures.

When the spin is parallel to the flight path the Magnus direction is undefined —
there is no Magnus force to have — and the whole transverse force is reported as
side force, which is what it is.
"""
function aerodynamic_coefficients(force::NTuple{3,<:Real}, s::BallState{T},
                                  p::BallProperties{T};
                                  ρ::Real = AIR_DENSITY) where {T}
    U = speed(s)
    U == 0 && return (zero(T), zero(T), zero(T))
    q = T(0.5) * T(ρ) * U^2 * p.area
    F = T.(force)
    û = s.v ./ U

    cx = s.ω[2] * û[3] - s.ω[3] * û[2]
    cy = s.ω[3] * û[1] - s.ω[1] * û[3]
    cz = s.ω[1] * û[2] - s.ω[2] * û[1]
    cn = sqrt(cx^2 + cy^2 + cz^2)

    CD = -(F[1] * û[1] + F[2] * û[2] + F[3] * û[3]) / q

    if cn < eps(T) * max(sqrt(sum(abs2, s.ω)), one(T))
        # No Magnus axis: report the whole transverse force as side force.
        par = F[1] * û[1] + F[2] * û[2] + F[3] * û[3]
        perp = (F[1] - par * û[1], F[2] - par * û[2], F[3] - par * û[3])
        return (CD, zero(T), sqrt(sum(abs2, perp)) / q)
    end

    n̂ = (cx / cn, cy / cn, cz / cn)
    b̂ = (û[2] * n̂[3] - û[3] * n̂[2],
         û[3] * n̂[1] - û[1] * n̂[3],
         û[1] * n̂[2] - û[2] * n̂[1])
    CL = (F[1] * n̂[1] + F[2] * n̂[2] + F[3] * n̂[3]) / q
    Cs = (F[1] * b̂[1] + F[2] * b̂[2] + F[3] * b̂[3]) / q
    return (CD, CL, Cs)
end
