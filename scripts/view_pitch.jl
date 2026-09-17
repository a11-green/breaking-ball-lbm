#!/usr/bin/env julia
#
# Interactive view of a pitch from the analytic model (§7.4, stage one).
#
# No CFD and no GPU: the trajectory comes from `src/trajectory/`'s 6DOF
# integrator driven by CoefficientAero, so it appears in a couple of seconds and
# the drawing can be got right long before there is a simulation worth drawing.
# The same code takes a CFD trajectory later — only the source of the states
# changes.
#
# **What the figure is for.** Break is a difference between two trajectories, and
# which one is subtracted changes the answer by nearly a factor of two: the WBC
# sweeper shows 15 inches of horizontal movement under the `pfx` convention and
# 27 under `induced` (§8). Stated in a table that reads as a caveat; drawn as
# three references against the pitch they are subtracted from, it is a picture of
# what the conventions mean.
#
# Makie is not a dependency of this package. Install it into the default
# environment, the way CUDA is installed for the benchmark:
#
#   julia -e 'using Pkg; Pkg.add("GLMakie")'          # a window
#   julia -e 'using Pkg; Pkg.add("WGLMakie")'         # a browser, no OpenGL
#
#   julia -e 'using Pkg; Pkg.add("CairoMakie")'        # a file, no display
#
#   julia --project=. scripts/view_pitch.jl
#   julia --project=. scripts/view_pitch.jl --web           # WGLMakie
#   julia --project=. scripts/view_pitch.jl --record out.mp4
#   julia --project=. scripts/view_pitch.jl --backend cairomakie --record out.png
#   julia --project=. scripts/view_pitch.jl --help

using BreakingBallLBM
using Printf

Base.@kwdef mutable struct ViewConfig
    pitch::String = "sweeper"
    speed::Float64 = NaN          # NaN: take the named pitch's own
    rpm::Float64 = NaN
    axis::Union{NTuple{3,Float64},Nothing} = nothing
    release::NTuple{3,Float64} = (2.0, 0.0, 1.75)
    cd::Float64 = 0.35
    cl::Float64 = 1.0
    ball_scale::Float64 = 15.0    # the ball is 75 mm across and the flight 18 m
    seam_samples::Int = 256
    backend::String = "glmakie"   # glmakie | wglmakie | cairomakie
    record_to::String = ""
    seconds::Float64 = 6.0        # wall-clock length of one playthrough
    family::Bool = false
end

function parse_args(args)
    c = ViewConfig()
    i = 1
    while i <= length(args)
        a = args[i]
        take() = (i += 1; args[i])
        if a == "--help"
            println("""
            view_pitch.jl [options]
              --pitch NAME     one of: $(join((t.name for t in PITCH_TYPES), ", "))
              --family         draw all of them at once instead of one
              --speed V        override the release speed, m/s
              --rpm R          override the spin rate
              --axis x,y,z     override the spin axis
              --cd C           drag coefficient of the model (default $(c.cd))
              --cl K           lift slope, C_L = K S (default $(c.cl))
              --ball-scale S   draw the ball S times life size (default $(c.ball_scale))
              --web            WGLMakie in a browser instead of GLMakie in a window
              --backend NAME   glmakie (a window), wglmakie (a browser), or
                               cairomakie (a still file, no display at all)
              --record FILE    write a movie instead of opening a window
              --seconds T      playthrough length for the movie (default $(c.seconds))
            Coordinates: x toward the plate, z up, y to the pitcher's left.""")
            exit(0)
        elseif a == "--pitch";       c.pitch = take()
        elseif a == "--family";      c.family = true
        elseif a == "--speed";       c.speed = parse(Float64, take())
        elseif a == "--rpm";         c.rpm = parse(Float64, take())
        elseif a == "--axis";        c.axis = Tuple(parse.(Float64, split(take(), ',')))
        elseif a == "--cd";          c.cd = parse(Float64, take())
        elseif a == "--cl";          c.cl = parse(Float64, take())
        elseif a == "--ball-scale";  c.ball_scale = parse(Float64, take())
        elseif a == "--web";         c.backend = "wglmakie"
        elseif a == "--backend";     c.backend = lowercase(take())
        elseif a == "--record";      c.record_to = take()
        elseif a == "--seconds";     c.seconds = parse(Float64, take())
        else
            error("unknown option $a — try --help")
        end
        i += 1
    end
    return c
end

"""The named pitch, with any overrides applied."""
function chosen(c::ViewConfig)
    idx = findfirst(t -> t.name == c.pitch, PITCH_TYPES)
    idx === nothing && error("unknown pitch \"$(c.pitch)\" — try --help")
    t = PITCH_TYPES[idx]
    return (name = t.name,
            axis = c.axis === nothing ? t.axis : c.axis,
            rpm = isnan(c.rpm) ? t.rpm : c.rpm,
            speed = isnan(c.speed) ? t.speed : c.speed)
end

# The arguments are read before the backend is chosen, so `--help` and a bad
# option cost nothing and work in an environment with no Makie installed at all.
const CFG = parse_args(ARGS)

# CairoMakie has no window and no event loop, so it is only good for a file.
if CFG.backend == "cairomakie" && isempty(CFG.record_to)
    error("--backend cairomakie draws no window: give it --record out.png")
end

# `using` cannot sit inside an `if` block, and which backend to load is a
# command-line choice, so it goes through @eval. All three re-export the whole
# Makie API, so everything below is written against that and no backend name
# appears again.
function load_backend(name)
    pkg = name == "glmakie"    ? :GLMakie :
          name == "wglmakie"   ? :WGLMakie :
          name == "cairomakie" ? :CairoMakie :
          error("unknown backend \"$name\" — try --help")
    try
        @eval using $pkg
    catch err
        err isa ArgumentError || rethrow()
        error("$pkg is not installed: julia -e 'using Pkg; Pkg.add(\"$pkg\")'")
    end
end

load_backend(CFG.backend)

"""Draw one pitch against the three references break is measured from."""
function view_single(c::ViewConfig)
    props = BaseballProperties()
    aero = CoefficientAero(props; CD = c.cd, CL_slope = c.cl)
    spec = chosen(c)
    s0 = BallState(; position = c.release, velocity = (spec.speed, 0.0, 0.0),
                   spin = spin_from_rpm(spec.axis, spec.rpm))

    refs = break_references(s0, props, aero)
    traj = simulate_trajectory(s0, props, aero)
    traj[end] = state_at_distance(traj, props, aero, PLATE_DISTANCE)
    coeffs = coefficient_series(traj, props, aero)
    metrics = pitch_metrics(s0, props, aero)
    geom = BaseballGeometry()

    @printf("%s: %.1f m/s, %.0f rpm about %s\n", spec.name, spec.speed, spec.rpm,
            string(spec.axis))
    show(stdout, MIME"text/plain"(), metrics)
    println("\n")

    path(s) = [Point3f(s.x[i], s.y[i], s.z[i]) for i in 1:length(s)]

    fig = Figure(size = (1500, 900))
    ax = Axis3(fig[1:3, 1]; aspect = :data, title = spec.name,
               xlabel = "toward the plate (m)", ylabel = "pitcher's left (m)",
               zlabel = "up (m)")

    # The three references first, then the pitch on top of them. The gaps at the
    # plate are the three definitions of break.
    lines!(ax, path(refs.ballistic); color = (:grey, 0.6), linestyle = :dot,
           linewidth = 2, label = "no forces at all — total")
    lines!(ax, path(refs.induced); color = (:steelblue, 0.8), linestyle = :dash,
           linewidth = 2, label = "no Magnus from release — induced")
    lines!(ax, path(refs.pfx); color = (:seagreen, 0.9), linestyle = :dashdot,
           linewidth = 2, label = "no Magnus from 40 ft — pfx")
    lines!(ax, path(refs.actual); color = :crimson, linewidth = 4, label = "the pitch")
    lines!(ax, [Point3f(p...) for p in plate_box()]; color = :black, linewidth = 2)
    axislegend(ax; position = :lt, framevisible = false)

    # The ball: a sphere and its seam, both following the same index. Without
    # the seam a spinning sphere is a still picture, and the seam's orientation
    # to the airflow is the subject (§4.1).
    frame = Observable(1)
    ballpos = lift(i -> Point3f(traj[i].x...), frame)
    R = Float32(props.radius * c.ball_scale)
    mesh!(ax, lift(p -> Sphere(p, R), ballpos); color = (:white, 0.5),
          transparency = true)
    lines!(ax, lift(i -> [Point3f(p...) for p in
                          seam_world(geom, traj[i].q, traj[i].x;
                                     samples = c.seam_samples, scale = c.ball_scale)],
                    frame); color = :firebrick, linewidth = 3)
    linesegments!(ax, lift(i -> begin
                               a, b = spin_axis_world(traj[i], 4 * R)
                               [Point3f(a...), Point3f(b...)]
                           end, frame); color = :darkorange, linewidth = 2)

    # The catcher's view is the one a hitter has, and the one break is quoted in.
    ax2 = Axis(fig[1, 2]; title = "from the catcher", xlabel = "pitcher's left (m)",
               ylabel = "up (m)", aspect = DataAspect())
    for (s, col, ls, lw) in ((refs.ballistic, (:grey, 0.6), :dot, 2),
                             (refs.induced, (:steelblue, 0.8), :dash, 2),
                             (refs.pfx, (:seagreen, 0.9), :dashdot, 2),
                             (refs.actual, :crimson, :solid, 3))
        lines!(ax2, s.y, s.z; color = col, linestyle = ls, linewidth = lw)
    end
    lines!(ax2, [p[2] for p in plate_box()], [p[3] for p in plate_box()];
           color = :black, linewidth = 2)
    scatter!(ax2, lift(i -> Point2f(traj[i].x[2], traj[i].x[3]), frame);
             color = :crimson, markersize = 14)

    ax3 = Axis(fig[2, 2]; title = "from the side", xlabel = "toward the plate (m)",
               ylabel = "up (m)")
    lines!(ax3, refs.ballistic.x, refs.ballistic.z; color = (:grey, 0.6), linestyle = :dot)
    lines!(ax3, refs.actual.x, refs.actual.z; color = :crimson, linewidth = 3)
    vlines!(ax3, [PLATE_DISTANCE]; color = :black)
    scatter!(ax3, lift(i -> Point2f(traj[i].x[1], traj[i].x[3]), frame);
             color = :crimson, markersize = 12)

    ax4 = Axis(fig[3, 2]; title = "coefficients and speed", xlabel = "time (s)")
    lines!(ax4, refs.actual.t, coeffs.CD; label = "C_D")
    lines!(ax4, refs.actual.t, coeffs.CL; label = "C_L")
    # Flat at zero for a coefficient model, and the line a CFD trajectory will
    # not leave flat — the non-Magnus force of §1.4. Having the axis already
    # there is half the reason to build this before the physics.
    lines!(ax4, refs.actual.t, coeffs.Cside; label = "C_side")
    lines!(ax4, refs.actual.t, refs.actual.speed ./ spec.speed; label = "|V| / V0")
    vlines!(ax4, lift(i -> traj[i].t, frame); color = (:crimson, 0.6))
    axislegend(ax4; position = :rb, framevisible = false)

    return fig, frame, length(traj)
end

"""Draw the named pitches together, which is what a spin axis is worth seeing."""
function view_family(c::ViewConfig)
    props = BaseballProperties()
    aero = CoefficientAero(props; CD = c.cd, CL_slope = c.cl)
    fam = pitch_family(; release = c.release, props = props, aero = aero)

    @printf("%-18s %-10s %-9s %-9s %-9s\n", "", "S", "pfx h", "pfx v", "induced h")
    for e in fam
        @printf("%-18s %-10.3f %-9.1f %-9.1f %-9.1f\n", e.name, e.metrics.spin_parameter,
                inches(e.metrics.pfx_horizontal), inches(e.metrics.pfx_vertical),
                inches(e.metrics.induced_horizontal))
    end
    println()

    fig = Figure(size = (1500, 800))
    ax = Axis3(fig[1, 1]; aspect = :data, title = "one release point, five spin axes",
               xlabel = "toward the plate (m)", ylabel = "pitcher's left (m)",
               zlabel = "up (m)")
    ax2 = Axis(fig[1, 2]; title = "from the catcher", xlabel = "pitcher's left (m)",
               ylabel = "up (m)", aspect = DataAspect())

    for (n, e) in enumerate(fam)
        s = e.samples
        col = Makie.wong_colors()[mod1(n, 7)]
        lines!(ax, [Point3f(s.x[i], s.y[i], s.z[i]) for i in 1:length(s)];
               color = col, linewidth = 3, label = e.name)
        lines!(ax2, s.y, s.z; color = col, linewidth = 3, label = e.name)
    end
    lines!(ax, [Point3f(p...) for p in plate_box()]; color = :black, linewidth = 2)
    lines!(ax2, [p[2] for p in plate_box()], [p[3] for p in plate_box()];
           color = :black, linewidth = 2)
    axislegend(ax; position = :lt, framevisible = false)
    return fig, nothing, 0
end

function main(c::ViewConfig)
    fig, frame, n = c.family ? view_family(c) : view_single(c)

    if !isempty(c.record_to)
        # A still for anything Makie writes as one image, a movie otherwise.
        still = frame === nothing ||
                lowercase(splitext(c.record_to)[2]) in (".png", ".svg", ".pdf", ".eps")
        if still
            frame === nothing || (frame[] = 1)
            save(c.record_to, fig)
            println("wrote ", c.record_to, " (a still)")
        else
            fps = 30
            nframes = max(2, round(Int, c.seconds * fps))
            record(fig, c.record_to, 1:nframes; framerate = fps) do f
                frame[] = clamp(round(Int, (f - 1) / (nframes - 1) * (n - 1)) + 1, 1, n)
            end
            println("wrote ", c.record_to)
        end
        return
    end

    if frame !== nothing
        slider = Slider(fig[4, 1:2], range = 1:n, startvalue = 1)
        on(v -> frame[] = v, slider.value)
    end
    display(fig)
    println("Drag in the 3-D panel to rotate, scroll to zoom." *
            (frame === nothing ? "" : " Drag the slider to scrub."))
    println("Press Enter to close.")
    readline()
end

main(CFG)
