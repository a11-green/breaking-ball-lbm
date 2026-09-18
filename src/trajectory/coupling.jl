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

**The accumulator is `Float64` whatever the solver's precision.** What the
controller acts on is `target − mean`, and at a production grid that difference
is around 0.03% of the mean itself — a small difference of two large sums. In
Float32 over 3×10⁷ nodes the rounding of the sum alone is of that order, so a
single-precision reduction would feed the controller as much noise as signal.
Double accumulation costs nothing here: the pass is bound by reading the
populations, not by adding them.
"""
function mean_fluid_velocity(g::Array{T,4}, wall::WallField{T},
                             force::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    Fx, Fy, Fz = Float64(force[1]) / 2, Float64(force[2]) / 2, Float64(force[3]) / 2
    ρtot = 0.0
    mx = 0.0; my = 0.0; mz = 0.0
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
        mx += Float64(px) + Fx; my += Float64(py) + Fy; mz += Float64(pz) + Fz
        nfluid += 1
    end
    nfluid == 0 && return zero(T), (zero(T), zero(T), zero(T))
    return T(ρtot / nfluid), (T(mx / ρtot), T(my / ρtot), T(mz / ρtot))
end

"""
    region_mean_velocity(g, weight, force)

Density and velocity averaged over an arbitrary weighted region.

`weight` is one number per node — zero where the node should not count, and it
is the caller's job to put zeros on the solid nodes, since nothing here knows
where they are. Otherwise the definition matches [`mean_fluid_velocity`](@ref)
exactly, `force/2` term and `Float64` accumulation included, and with a weight
of one on every fluid node the two agree to rounding.

The reason to want this is that **the box mean is not the velocity the body
sees**. In a periodic box the mean is fixed by the mass flux through any plane,
so it is the same whatever the wake does; what changes is the profile, and the
body sits on the axis, in the retarded part of it. A disc of nodes there reads
the deficit the box mean is blind to.

The two are bounds rather than rivals: the box mean includes the bypass flow the
body has accelerated, so it is too fast, and the core disc is the deepest part
of the deficit, so it is too slow. `scripts/validate_sphere_highre.jl` reports a
drag coefficient against both for exactly that reason.
"""
function region_mean_velocity(g::Array{T,4}, weight::Array{T,3},
                              force::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    size(weight) == (nx, ny, nz) ||
        throw(DimensionMismatch("weight is $(size(weight)), the lattice is $((nx, ny, nz))"))
    Fx, Fy, Fz = Float64(force[1]) / 2, Float64(force[2]) / 2, Float64(force[3]) / 2
    ρtot = 0.0
    mx = 0.0; my = 0.0; mz = 0.0
    wtot = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        w = Float64(weight[i, j, k])
        w == 0 && continue
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
        ρtot += w * Float64(ρ)
        mx += w * (Float64(px) + Fx)
        my += w * (Float64(py) + Fy)
        mz += w * (Float64(pz) + Fz)
        wtot += w
    end
    (wtot == 0 || ρtot == 0) && return zero(T), (zero(T), zero(T), zero(T))
    return T(ρtot / wtot), (T(mx / ρtot), T(my / ρtot), T(mz / ρtot))
end

# --- backend interface -----------------------------------------------------
#
# The loop below touches the flow through exactly three operations, so a
# backend is whatever answers them. On the host that is the `WallField` itself,
# with no scratch; on the device it is a handle that also owns the per-boundary
# contribution array and the fluid mask the reduction needs. Keeping the
# interface this narrow is what lets `couple_step!` be one function rather than
# two that have to be kept in agreement.

"""
    flow_wall(flow)

The `WallField` a backend is currently solving against — host or device, plain
or rotating. A snapshot needs it to know where the body is *now*, which after a
re-cut is not where the host copy of the geometry thinks it is: on the device
the re-cut happens in device memory and nothing copies it back.
"""
flow_wall(w::WallField) = w

"""How many nodes the controller's momentum is spread over."""
flow_fluid_count(w::WallField) = count(!=(SOLID_NODE), w.kind)
const fluid_node_count = flow_fluid_count

"""Advance the flow one sub-cycle, returning the mean `(force, torque)`."""
advance_flow!(g::Array{T,4}, w::WallField{T}, nsteps::Integer, τ::Real;
              kwargs...) where {T} =
    aa_run_walls!(g, w, nsteps, τ; reduction = :mean, kwargs...)

"""Mass-averaged density and velocity over the fluid nodes."""
flow_mean_velocity(g::Array{T,4}, w::WallField{T}, force::NTuple{3,<:Real}) where {T} =
    mean_fluid_velocity(g, w, force)

"""
Everything about a coupled run that does not change from sub-cycle to
sub-cycle.

`control_time` is the controller's time constant, in lattice steps. It has to be
long compared with a sub-cycle and with the time the box takes to convect its
own length, or the controller chases turbulence instead of correcting drift.

`recut_drift` is how far the ball's surface may turn, in lattice spacings at the
equator, before the seam geometry is re-cut. It is checked once per sub-cycle,
so the sub-cycle has to be short enough that the surface does not outrun the
threshold within one — see [`max_substeps`](@ref). With a static geometry it has
no effect.
"""
struct PitchRun{T<:AbstractFloat}
    units::LatticeUnits{T}
    props::BallProperties{T}
    substeps::Int
    control_time::T
    recut_drift::T
    smagorinsky::T
    operator::Symbol
    rule::Symbol
    omega_bulk::T
    omega_higher::T
    gravity::NTuple{3,T}
    channel::Union{Nothing,OpenChannel{T}}
end

function PitchRun(units::LatticeUnits{T}, props::BallProperties{T};
                  substeps::Integer = 100, control_time::Real = 20 * substeps,
                  recut_drift::Real = 0.25, smagorinsky::Real = 0.0,
                  operator::Symbol = :central_moment,
                  rule::Symbol = :interpolated_local,
                  omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                  gravity::NTuple{3,<:Real} = GRAVITY,
                  channel::Union{Nothing,OpenChannel{T}} = nothing) where {T}
    iseven(substeps) ||
        throw(ArgumentError("substeps must be even, got $substeps"))
    # The controller is only there to hold the stream a periodic box cannot
    # state, so with open faces its time constant is irrelevant rather than
    # wrong — but a run that sets both has misunderstood which one is acting.
    channel === nothing && control_time <= substeps &&
        throw(ArgumentError("control_time ($control_time) must exceed substeps ($substeps)"))
    return PitchRun{T}(units, props, Int(substeps), T(control_time), T(recut_drift),
                       T(smagorinsky), operator, rule, T(omega_bulk), T(omega_higher),
                       T.(gravity), channel)
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
    fresh::Int                  # nodes the last re-cut uncovered
    steps::Int
    released::Quat{T}           # orientation when the wall geometry was last built
end

function PitchState(ball::BallState{T}) where {T}
    z = (zero(T), zero(T), zero(T))
    return PitchState{T}(ball, z, z, z, z, z, 0, 0, ball.q)
end

"""
    couple_step!(g, run, state, flow; frozen = false)

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
function couple_step!(g, run::PitchRun{T}, st::PitchState{T},
                      flow; frozen::Bool = false) where {T}
    u = run.units
    target = lattice_freestream(u, st.ball)
    _, ū = flow_mean_velocity(g, flow, st.control)
    st.mean_velocity = ū

    # PI, critically damped at time constant `control_time`. The integral term
    # is what supplies the steady momentum the sphere removes; the proportional
    # term is what stops the body force ringing about it.
    #
    # With open faces there is nothing for it to do. The inlet states the free
    # stream instead of the box mean inferring it, and the momentum the ball
    # removes leaves through the outlet instead of having to be put back — so
    # the controller stays at zero and the only body force is the frame's own.
    # The two remain consistent without feedback: the identity d(u∞)/dt =
    # a_fluid (§4.4) means the interior accelerates at exactly the rate the
    # inlet value moves between sub-cycles.
    if run.channel === nothing
        e = target .- ū
        Tc = run.control_time
        st.integral = st.integral .+ e .* run.substeps
        st.control = (2 / Tc) .* e .+ (1 / Tc^2) .* st.integral
    end

    # The seam has to be where the ball is pointing before the sub-cycle runs,
    # not after: these are the δ values the bounce-back is about to use.
    spin_lat = lattice_spin(u, st.ball)
    st.fresh = maybe_recut!(g, flow, st.ball.q, spin_lat, run.recut_drift)

    sample_force, sample_torque = st.force, st.torque
    a_frame = frozen ? (zero(T), zero(T), zero(T)) :
              lattice_body_force(u, sample_force, run.props; gravity = run.gravity)
    body = a_frame .+ st.control

    F_lat, M_lat = advance_flow!(g, flow, run.substeps, u.τ;
                                 force = body, spin = spin_lat,
                                 operator = run.operator, rule = run.rule,
                                 smagorinsky = run.smagorinsky,
                                 omega_bulk = run.omega_bulk,
                                 omega_higher = run.omega_higher,
                                 channel = run.channel, inlet = target)

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
    couple_residual(run, state, flow)

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
function couple_residual(run::PitchRun{T}, st::PitchState{T}, flow) where {T}
    # With open faces the balance this checks does not exist: the controller
    # supplies nothing and the momentum leaves through the outlet. Restoring an
    # equivalent means measuring the momentum flux through the two faces and
    # comparing that with the surface force — a real check, and a different one,
    # which is not written yet. A NaN says so rather than a ratio of one saying
    # nothing.
    run.channel === nothing || return T(NaN)
    supplied = st.control .* flow_fluid_count(flow)      # lattice force, ρ = 1
    removed = to_lattice_force(run.units, st.force)
    scale = max(sqrt(sum(abs2, removed)), eps(T))
    return sqrt(sum(abs2, supplied .- removed)) / scale
end

"""Newtons back to lattice force, the inverse of [`to_physical_force`](@ref)."""
to_lattice_force(u::LatticeUnits, F::Real) = F * u.dt^2 / (u.ρ_physical * u.dx^4)
to_lattice_force(u::LatticeUnits, F::NTuple{3,<:Real}) =
    (to_lattice_force(u, F[1]), to_lattice_force(u, F[2]), to_lattice_force(u, F[3]))

"""
    spin_up!(g, run, state, flow; cycles)

Develop the boundary layer with the trajectory frozen (§5). Returns the
residual after the last cycle, which is the honest measure of whether the flow
has settled enough to start integrating the trajectory.
"""
function spin_up!(g, run::PitchRun{T}, st::PitchState{T},
                  flow; cycles::Integer = 10) where {T}
    local res = T(Inf)
    for _ in 1:cycles
        couple_step!(g, run, st, flow; frozen = true)
        res = couple_residual(run, st, flow)
    end
    return res
end

"""
    fly!(g, run, state, flow; cycles, callback)

Run the coupled loop with the trajectory live. `callback(state)` is called after
each sub-cycle and may return `false` to stop — which is how the caller ends the
run at the plate, or rebuilds the wall geometry once
[`orientation_drift`](@ref) says the seam has turned far enough to matter.
"""
function fly!(g, run::PitchRun{T}, st::PitchState{T}, flow;
              cycles::Integer = 100, callback = nothing) where {T}
    for _ in 1:cycles
        couple_step!(g, run, st, flow)
        if callback !== nothing && callback(st) === false
            return st
        end
    end
    return st
end
