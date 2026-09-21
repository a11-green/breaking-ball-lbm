#!/usr/bin/env julia
#
# Measure pfx/induced/total break from a real CFD trajectory CSV (§8 V&V-5).
#
# The three definitions (`src/trajectory/trajectory.jl`) are differences
# between the actual trajectory and a *reference* trajectory with the Magnus
# force switched off, branching from either release (induced), 40 ft from the
# plate (pfx), or a straight line at the release velocity (total). For the
# analytic model that reference is one line — `no_magnus(aero)` just zeroes
# the model's `CL_slope`. A CFD run has no such knob: there is no "this run,
# but without the Magnus term" to re-simulate.
#
# What we *do* have is this run's own measured C_D(t) — a real, turbulent,
# non-constant drag history, not a fitted constant like the analytic model
# uses. `MeasuredDragAero` below is a drag-only aerodynamic model built from
# that history (interpolated by elapsed time) with lift and side force fixed
# at exactly zero — i.e. it does not approximate "no Magnus", it *is* no
# Magnus, using this run's own drag rather than a guessed one. Re-integrating
# a reference trajectory with it is what the console's own hint after a
# run — "re-integrate from the CSV with the Magnus component removed" — means.
#
#   julia --project=. scripts/measure_break.jl pitch.csv
#   julia --project=. scripts/measure_break.jl pitch.csv --dt 1e-5
#   julia --project=. scripts/measure_break.jl --help
#
# No Makie, no GPU — this is plain numerical post-processing.

using BreakingBallLBM
using Printf

Base.@kwdef mutable struct BreakConfig
    csv::String = "pitch.csv"
    dt::Float64 = 1.0e-4
end

function parse_args(args)
    c = BreakConfig()
    positional_taken = false
    i = 1
    while i <= length(args)
        a = args[i]
        take() = i < length(args) ? (i += 1; args[i]) :
                 error("option $a needs a value — try --help")
        if a == "--help"
            println("""
            measure_break.jl [CSV] [options]
              CSV        trajectory file to read (default $(c.csv)), may also
                         be given as --csv FILE
              --csv FILE same, as a named option
              --dt T     integration step for the reference trajectories, s
                         (default $(c.dt))
            Prints the pfx/induced/total break of this run (§8 V&V-5),
            re-integrated from the CSV's own measured C_D with lift and side
            force switched off — see the header comment for why.""")
            exit(0)
        elseif a == "--csv"; c.csv = take()
        elseif a == "--dt";  c.dt = parse(Float64, take())
        elseif !startswith(a, "--") && !positional_taken
            c.csv = a
            positional_taken = true
        else
            error("unknown option $a — try --help")
        end
        i += 1
    end
    return c
end

"""
    MeasuredDragAero(ts, cds, props, ρ)

Drag along −V̂ using this run's own C_D(t) (linearly interpolated, clamped at
the ends), zero lift and zero side force — the reference model for a break
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

"""Linearly interpolate every column of `data` to the row where `:x` first
reaches `xtarget` — the CFD analogue of `state_at_distance` for recorded,
not integrated, samples."""
function row_at_x(data, xtarget::Real)
    x = data[:x]
    n = length(x)
    i = findfirst(j -> x[j] >= xtarget, 1:n)
    i === nothing && error("the run never reaches x = $xtarget m (only got to $(x[end]) m)")
    i == 1 && return (t = data[:t][1], x = data[:x][1], y = data[:y][1], z = data[:z][1],
                      vx = data[:vx][1], vy = data[:vy][1], vz = data[:vz][1])
    frac = (xtarget - x[i-1]) / (x[i] - x[i-1])
    lerp(k) = data[k][i-1] + frac * (data[k][i] - data[k][i-1])
    return (t = lerp(:t), x = lerp(:x), y = lerp(:y), z = lerp(:z),
            vx = lerp(:vx), vy = lerp(:vy), vz = lerp(:vz))
end

function main(c::BreakConfig)
    isfile(c.csv) || error("no such file: $(c.csv)")
    data = read_pitch_csv(c.csv)
    require_columns(data, :t, :x, :y, :z, :vx, :vy, :vz, :CD)
    n = length(data[:t])
    n > 1 || error("$(c.csv) has $n row(s) — nothing to measure")

    props = BaseballProperties()
    aero = MeasuredDragAero(data[:t], data[:CD], props, AIR_DENSITY)

    fly_from(branch) = begin
        s0 = BallState(; position = (branch.x, branch.y, branch.z),
                       velocity = (branch.vx, branch.vy, branch.vz), time = branch.t)
        traj = simulate_trajectory(s0, props, aero; dt = c.dt, distance = PLATE_DISTANCE)
        state_at_distance(traj, props, aero, PLATE_DISTANCE)
    end

    release = row_at_x(data, data[:x][1])
    actual = row_at_x(data, PLATE_DISTANCE)

    induced_ref = fly_from(release)

    x40 = PLATE_DISTANCE - PFX_SEGMENT
    branch40 = row_at_x(data, x40)
    pfx_ref = fly_from(branch40)

    tb = (PLATE_DISTANCE - release.x) / release.vx
    total_y = release.y + tb * release.vy
    total_z = release.z + tb * release.vz

    pfx_h, pfx_v = actual.y - pfx_ref.x[2], actual.z - pfx_ref.x[3]
    ind_h, ind_v = actual.y - induced_ref.x[2], actual.z - induced_ref.x[3]
    tot_h, tot_v = actual.y - total_y, actual.z - total_z

    @printf("%s: %d rows, release x=%.3f m (t=%.4f s) -> plate x=%.4f m (t=%.4f s)\n",
            c.csv, n, release.x, release.t, actual.x, actual.t)
    @printf("release speed %.2f m/s, plate speed %.2f m/s\n\n",
            sqrt(release.vx^2 + release.vy^2 + release.vz^2),
            sqrt(sum(abs2, (data[:vx][end], data[:vy][end], data[:vz][end]))))

    println("break (this run, measured C_D, lift+side removed from the branch point on):")
    @printf("  %-10s %10s %10s   %10s %10s\n", "definition", "horiz (m)", "horiz (in)",
            "vert (m)", "vert (in)")
    for (name, h, v) in (("pfx", pfx_h, pfx_v), ("induced", ind_h, ind_v), ("total", tot_h, tot_v))
        @printf("  %-10s %10.4f %10.1f   %10.4f %10.1f\n", name, h, inches(h), v, inches(v))
    end
    println()
    println("for comparison (§8 V&V-5):")
    println("  reported (CBS Sports / SI): horiz 17.0 in, vert 32.0 in (no definition stated)")
    println("  analytic coefficient model: pfx 15.4/-0.1 in, induced 27.4/-0.1 in, total 27.4/-36.9 in")
end

main(parse_args(ARGS))
