"""
Measuring pfx/induced/total break from a real CFD trajectory (§8 V&V-5).

The three definitions (`src/trajectory/trajectory.jl`) are differences between
the actual trajectory and a *reference* trajectory with the Magnus force
switched off, branching from either release (induced), 40 ft from the plate
(pfx), or a straight line at the release velocity (total). For the analytic
model that reference is one line — `no_magnus(aero)` just zeroes the model's
`CL_slope`. A CFD run has no such knob: there is no "this run, but without the
Magnus term" to re-simulate.

What we *do* have is this run's own measured C_D(t) — a real, turbulent,
non-constant drag history, not a fitted constant like the analytic model uses.
[`MeasuredDragAero`](@ref) is a drag-only aerodynamic model built from that
history (interpolated by elapsed time) with lift and side force fixed at
exactly zero — i.e. it does not approximate "no Magnus", it *is* no Magnus,
using this run's own drag rather than a guessed one. [`measure_break`](@ref)
re-integrates the three reference trajectories with it, which is what the
console's own hint after a run — "re-integrate from the CSV with the Magnus
component removed" — means.

This lives in the library, not `scripts/measure_break.jl`, for the same
reason `postprocess/pitch_view.jl` does: it is plain numerical code (no
Makie, no GPU, no display) and is worth holding to the test suite's standard,
so both the printing script and a future plotting one call the same tested
function instead of two copies drifting apart.
"""

"""
    MeasuredDragAero(ts, cds, props, ρ)

Drag along −V̂ using a recorded C_D(t) (linearly interpolated, clamped at the
ends), zero lift and zero side force — the reference model for a break
measured from CFD data rather than a coefficient fit.
"""
struct MeasuredDragAero{T<:AbstractFloat}
    ts::Vector{T}
    cds::Vector{T}
    props::BallProperties{T}
    ρ::T
end

function cd_at(m::MeasuredDragAero{T}, t::T) where {T}
    ts = m.ts
    t <= ts[1] && return m.cds[1]
    t >= ts[end] && return m.cds[end]
    i = clamp(searchsortedlast(ts, t), 1, length(ts) - 1)
    frac = (t - ts[i]) / (ts[i+1] - ts[i])
    return m.cds[i] + frac * (m.cds[i+1] - m.cds[i])
end

function (m::MeasuredDragAero{T})(s::BallState{T}) where {T}
    zero3 = (zero(T), zero(T), zero(T))
    U = speed(s)
    U == 0 && return zero3, zero3
    qdyn = T(0.5) * m.ρ * U^2 * m.props.area
    Fd = (-cd_at(m, s.t) * qdyn) .* (s.v ./ U)
    return Fd, zero3
end

"""
    row_at_x(data, xtarget)

Linearly interpolate every column of a [`read_pitch_csv`](@ref) `data` to the
row where `:x` first reaches `xtarget` — the CFD analogue of
[`state_at_distance`](@ref) for recorded, not integrated, samples.
"""
function row_at_x(data, xtarget::Real)
    x = data[:x]
    n = length(x)
    i = findfirst(j -> x[j] >= xtarget, 1:n)
    i === nothing && throw(ArgumentError(
        "the run never reaches x = $xtarget m (only got to $(x[end]) m)"))
    i == 1 && return (t = data[:t][1], x = data[:x][1], y = data[:y][1], z = data[:z][1],
                      vx = data[:vx][1], vy = data[:vy][1], vz = data[:vz][1])
    frac = (xtarget - x[i-1]) / (x[i] - x[i-1])
    lerp(k) = data[k][i-1] + frac * (data[k][i] - data[k][i-1])
    return (t = lerp(:t), x = lerp(:x), y = lerp(:y), z = lerp(:z),
            vx = lerp(:vx), vy = lerp(:vy), vz = lerp(:vz))
end

"""
The pfx/induced/total break of one CFD run, in metres, plus enough of the
flight to report alongside them.
"""
struct MeasuredBreak{T<:AbstractFloat}
    release_x::T
    release_t::T
    release_speed::T
    plate_x::T
    plate_t::T
    plate_speed::T
    pfx_horizontal::T
    pfx_vertical::T
    induced_horizontal::T
    induced_vertical::T
    total_horizontal::T
    total_vertical::T
end

"""
    measure_break(data; dt = 1e-4, distance = PLATE_DISTANCE)

The pfx/induced/total break of a trajectory `data` (as [`read_pitch_csv`]
(@ref) returns), measured by re-integrating with [`MeasuredDragAero`](@ref)
from release and from 40 ft out — see the module docstring for what that
means and why it is the CFD analogue of `no_magnus(aero)`.

`data` needs `:t, :x, :y, :z, :vx, :vy, :vz, :CD`; anything else is ignored.
"""
function measure_break(data; dt::Real = 1.0e-4, distance::Real = PLATE_DISTANCE)
    require_columns(data, :t, :x, :y, :z, :vx, :vy, :vz, :CD)
    n = length(data[:t])
    n > 1 || throw(ArgumentError("data has $n row(s) — nothing to measure"))

    T = eltype(data[:t])
    props = BaseballProperties(T)
    aero = MeasuredDragAero(data[:t], data[:CD], props, T(AIR_DENSITY))

    function fly_from(branch)
        s0 = BallState(T; position = (branch.x, branch.y, branch.z),
                       velocity = (branch.vx, branch.vy, branch.vz), time = branch.t)
        traj = simulate_trajectory(s0, props, aero; dt = dt, distance = distance)
        state_at_distance(traj, props, aero, distance)
    end

    release = row_at_x(data, data[:x][1])
    actual = row_at_x(data, distance)

    induced_ref = fly_from(release)

    x40 = distance - PFX_SEGMENT
    branch40 = row_at_x(data, x40)
    pfx_ref = fly_from(branch40)

    tb = (distance - release.x) / release.vx
    total_y = release.y + tb * release.vy
    total_z = release.z + tb * release.vz

    return MeasuredBreak{T}(
        release.x, release.t, sqrt(release.vx^2 + release.vy^2 + release.vz^2),
        actual.x, actual.t,
        sqrt(sum(abs2, (data[:vx][end], data[:vy][end], data[:vz][end]))),
        actual.y - pfx_ref.x[2], actual.z - pfx_ref.x[3],
        actual.y - induced_ref.x[2], actual.z - induced_ref.x[3],
        actual.y - total_y, actual.z - total_z)
end
