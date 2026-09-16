"""
The tight loop: CFD and the equations of motion, coupled every sub-cycle (§5).

One outer iteration is

  1. measure the box-mean velocity and work out the body force that holds it at
     `−V(t)` while also standing for the frame's acceleration,
  2. run `substeps` lattice steps with that force, the current spin on the wall,
     and the momentum-exchange force averaged over the whole sub-cycle,
  3. convert to newtons and newton-metres and advance the 6DOF equations,
  4. feed the new velocity and spin back into step 1.

**Why `substeps` > 1.** The ball's velocity changes by about 15% over the
185,000 steps of a pitch, so ~1e-6 per step: resolving the trajectory does not
need a 6DOF update every lattice step. The force does not behave that way — the
instantaneous momentum-exchange sum on a body in turbulent flow fluctuates by
more than the mean it fluctuates about — so the sub-cycle is there to *average*
the force, not merely to save work.

**The mean-flow controller, and what it admits.** The domain is periodic, so the
drag the sphere exerts removes momentum from the fluid with nowhere for it to
go, and the free stream would decay — over a whole pitch by more than the free
stream itself. Real air has an infinite reservoir; a periodic box has to be
given one. The controller holds the box-mean velocity at `−V(t)` with a uniform
body force, which is the standard constant-mass-flux forcing.

It is a **PI** controller, and the I alone will not do. With a pure integrator
the error obeys `ë = −K e`: undamped, so the body force rings about the value it
is looking for. Since the value it is looking for is tiny — the steady
correction changes the mean velocity by about 0.03% per sub-cycle — even a
modest ring buries it, and the first attempt here produced a control signal
whose oscillation was several times its own mean. Adding the proportional term
gives `ë + K_p ė + K_i e = 0`, and choosing `K_p = 2/T`, `K_i = 1/T²` makes it
critically damped with time constant `T`. `T` wants to be a few flow-through
times of the box, not a few steps.

The converged body force is then itself a drag measurement: at a steady state
the momentum it supplies per unit time must equal the force the surface removes.
`couple_residual` reports the two against each other, which makes the loop
self-checking rather than merely plausible.

What this does *not* do is replace inflow and outflow boundaries. The periodic
images still exert the blockage that §8's V&V-2 measured against Hasimoto's
solution, so the domain has to be wide enough for that to be small. Proper
non-reflecting boundaries are the upgrade (§11), and they slot in here by
replacing the controller.
"""

"""
    mean_fluid_velocity(g, wall, force)

Mass-averaged density and velocity over the fluid nodes.

Only valid with `g` in the even-step layout, where a node's own populations sit
in its own 27 slots — which is where `aa_run_walls!` always leaves it, since it
insists on an even step count. `force` is the body force that was applied, and
it belongs here: the Guo scheme defines momentum as `Σ c f + F/2`, so leaving it
out would bias the mean by half a step's worth of acceleration.
"""
function mean_fluid_velocity(g::Array{T,4}, wall::WallField{T},
                             force::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    Fx, Fy, Fz = T(force[1]) / 2, T(force[2]) / 2, T(force[3]) / 2
    ρtot = zero(T)
    mx = zero(T); my = zero(T); mz = zero(T)
    nfluid = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        wall.kind[i, j, k] == SOLID_NODE && continue
        ρ = zero(T)
        px = zero(T); py = zero(T); pz = zero(T)
        Base.Cartesian.@nexprs 27 s -> begin
            f_s = g[i, j, k, s]
            cv_s = cube_velocity(s)
            ρ += f_s
            px += T(cv_s[1]) * f_s
            py += T(cv_s[2]) * f_s
            pz += T(cv_s[3]) * f_s
        end
        ρtot += ρ
        mx += px + Fx; my += py + Fy; mz += pz + Fz
        nfluid += 1
    end
    nfluid == 0 && return zero(T), (zero(T), zero(T), zero(T))
    return ρtot / nfluid, (mx / ρtot, my / ρtot, mz / ρtot)
end

"""Fluid node count — the volume the controller's momentum is spread over."""
fluid_node_count(wall::WallField) = count(!=(SOLID_NODE), wall.kind)

"""
Everything about a coupled run that does not change from sub-cycle to
sub-cycle.

`control_time` is the controller's time constant, in lattice steps. It has to be
long compared with a sub-cycle and with the time the box takes to convect its
own length, or the controller chases turbulence instead of correcting drift.
"""
struct PitchRun{T<:AbstractFloat}
    units::LatticeUnits{T}
    props::BallProperties{T}
    substeps::Int
    control_time::T
    operator::Symbol
    rule::Symbol
    omega_bulk::T
    omega_higher::T
    gravity::NTuple{3,T}
end

function PitchRun(units::LatticeUnits{T}, props::BallProperties{T};
                  substeps::Integer = 100, control_time::Real = 20 * substeps,
                  operator::Symbol = :central_moment,
                  rule::Symbol = :interpolated_local,
                  omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                  gravity::NTuple{3,<:Real} = GRAVITY) where {T}
    iseven(substeps) ||
        throw(ArgumentError("substeps must be even, got $substeps"))
    control_time > substeps ||
        throw(ArgumentError("control_time ($control_time) must exceed substeps ($substeps)"))
    return PitchRun{T}(units, props, Int(substeps), T(control_time), operator, rule,
                       T(omega_bulk), T(omega_higher), T.(gravity))
end

"""
Everything that does change: the ball, the controller's accumulated body force,
the most recent aerodynamic sample, and how far the run has got.
"""
mutable struct PitchState{T<:AbstractFloat}
    ball::BallState{T}
    control::NTuple{3,T}        # lattice acceleration, from the controller
    integral::NTuple{3,T}       # ∫(target - mean) dt, in lattice units
    force::NTuple{3,T}          # N, most recent sub-cycle average
    torque::NTuple{3,T}         # N·m
    mean_velocity::NTuple{3,T}  # lattice units, measured
    steps::Int
    released::Quat{T}           # orientation when the wall geometry was last built
end

function PitchState(ball::BallState{T}) where {T}
    z = (zero(T), zero(T), zero(T))
    return PitchState{T}(ball, z, z, z, z, z, 0, ball.q)
end

"""
    couple_step!(g, run, state, wall; frozen = false)

One outer iteration. Returns the body force that was applied, in lattice units.

`frozen` is the spin-up mode of §5: the flow develops around a ball held at a
fixed velocity, so the frame is not accelerating and contributes no body force,
the controller alone drives the free stream, and the trajectory does not move.
The ball still turns, since the seam has to be in the right place.

**Everything in one iteration uses the aerodynamic sample from the previous
one**, including the force that advances the 6DOF equations. That is not
laziness about staleness: the design rests on the body force being exactly
`d(u∞)/dt` (see `frame.jl`), and that identity holds term for term only if the
force standing for the frame's acceleration and the force accelerating the ball
are the *same number*. Using the freshly measured force for the trajectory and
last cycle's for the fluid would put the free stream and the inflow condition a
sub-cycle out of step, which is the drift the whole scheme is built to avoid.
"""
function couple_step!(g::Array{T,4}, run::PitchRun{T}, st::PitchState{T},
                      wall::WallField{T}; frozen::Bool = false) where {T}
    u = run.units
    target = lattice_freestream(u, st.ball)
    _, ū = mean_fluid_velocity(g, wall, st.control)
    st.mean_velocity = ū

    # PI, critically damped at time constant `control_time`. The integral term
    # is what supplies the steady momentum the sphere removes; the proportional
    # term is what stops the body force ringing about it.
    e = target .- ū
    Tc = run.control_time
    st.integral = st.integral .+ e .* run.substeps
    st.control = (2 / Tc) .* e .+ (1 / Tc^2) .* st.integral

    sample_force, sample_torque = st.force, st.torque
    a_frame = frozen ? (zero(T), zero(T), zero(T)) :
              lattice_body_force(u, sample_force, run.props; gravity = run.gravity)
    body = a_frame .+ st.control

    F_lat, M_lat = aa_run_walls!(g, wall, run.substeps, u.τ;
                                 force = body, spin = lattice_spin(u, st.ball),
                                 operator = run.operator, rule = run.rule,
                                 reduction = :mean,
                                 omega_bulk = run.omega_bulk,
                                 omega_higher = run.omega_higher)

    dt = run.substeps * u.dt
    if frozen
        st.ball = BallState{T}(st.ball.x, st.ball.v, st.ball.ω,
                               normalize(st.ball.q + dt * quat_rate(st.ball.q, st.ball.ω)),
                               st.ball.t + dt)
    else
        st.ball = advance(st.ball, run.props, _ -> (sample_force, sample_torque), dt;
                          gravity = run.gravity)
    end

    st.force = to_physical_force(u, F_lat)
    st.torque = to_physical_torque(u, M_lat)
    st.steps += run.substeps
    return body
end

"""
    couple_residual(run, state, wall)

How far the loop is from closing on itself, as a dimensionless number.

At a quasi-steady state the momentum the controller pumps into the fluid every
step, `a_control × (fluid node count) × ρ`, has to equal the momentum the
surface takes out, which is the momentum-exchange force — two quantities
computed by completely different routes.

The frame's own body force does not enter. It accelerates the free stream at
exactly the rate the target is moving, by the identity in `frame.jl`, so it
cancels out of the balance and the controller is left facing the drag alone.
Normalising the difference by the force gives a running check that the forcing,
the boundary condition and the unit conversions all agree. It is not expected to
vanish while the flow is still developing.
"""
function couple_residual(run::PitchRun{T}, st::PitchState{T}, wall::WallField{T}) where {T}
    supplied = st.control .* fluid_node_count(wall)      # lattice force, ρ = 1
    removed = to_lattice_force(run.units, st.force)
    scale = max(sqrt(sum(abs2, removed)), eps(T))
    return sqrt(sum(abs2, supplied .- removed)) / scale
end

"""Newtons back to lattice force, the inverse of [`to_physical_force`](@ref)."""
to_lattice_force(u::LatticeUnits, F::Real) = F * u.dt^2 / (u.ρ_physical * u.dx^4)
to_lattice_force(u::LatticeUnits, F::NTuple{3,<:Real}) =
    (to_lattice_force(u, F[1]), to_lattice_force(u, F[2]), to_lattice_force(u, F[3]))

"""
    spin_up!(g, run, state, wall; cycles)

Develop the boundary layer with the trajectory frozen (§5). Returns the
residual after the last cycle, which is the honest measure of whether the flow
has settled enough to start integrating the trajectory.
"""
function spin_up!(g::Array{T,4}, run::PitchRun{T}, st::PitchState{T},
                  wall::WallField{T}; cycles::Integer = 10) where {T}
    local res = T(Inf)
    for _ in 1:cycles
        couple_step!(g, run, st, wall; frozen = true)
        res = couple_residual(run, st, wall)
    end
    return res
end

"""
    fly!(g, run, state, wall; cycles, callback)

Run the coupled loop with the trajectory live. `callback(state)` is called after
each sub-cycle and may return `false` to stop — which is how the caller ends the
run at the plate, or rebuilds the wall geometry once
[`orientation_drift`](@ref) says the seam has turned far enough to matter.
"""
function fly!(g::Array{T,4}, run::PitchRun{T}, st::PitchState{T}, wall::WallField{T};
              cycles::Integer = 100, callback = nothing) where {T}
    for _ in 1:cycles
        couple_step!(g, run, st, wall)
        if callback !== nothing && callback(st) === false
            return st
        end
    end
    return st
end
