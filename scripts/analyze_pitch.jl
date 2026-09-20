#!/usr/bin/env julia
#
# Plot and inspect a trajectory CSV written by run_pitch.jl (§7.4, stage two).
#
# This is the CFD counterpart of view_pitch.jl (§7.4.1, stage one): stage one
# draws the analytic-model trajectory before any CFD exists so the figure's
# layout can be got right early; this script draws the real thing once a run
# has produced one. It reads whatever columns the CSV has (`read_pitch_csv`
# makes no assumption about the schema) and fails with a clear message naming
# what is missing rather than plotting garbage, so a CSV from an older run
# says plainly which columns it lacks instead of crashing three panels in.
#
#   julia --project=. scripts/analyze_pitch.jl pitch.csv
#   julia --project=. scripts/analyze_pitch.jl pitch.csv --backend cairomakie --record report.png
#   julia --project=. scripts/analyze_pitch.jl pitch.csv --smooth 41
#   julia --project=. scripts/analyze_pitch.jl pitch.csv --reference "4-seam fastball"
#   julia --project=. scripts/analyze_pitch.jl pitch.csv --compare other_pitch.csv
#   julia --project=. scripts/analyze_pitch.jl --help
#
# **--reference** overlays an analytic-model trajectory from view_pitch.jl's own
# `PITCH_TYPES` table (§7.4.1) — a generic, literature-typical parametrization
# (e.g. "4-seam fastball" = 42 m/s, 2400 rpm backspin), NOT a specific pitcher's
# actual pitch. A player-specific reference (Ohtani's own four-seam, say) needs
# a Baseball Savant-confirmed speed/spin/axis the way the WBC sweeper case has
# in DESIGN.md §8 — that lookup hasn't been done yet, so it isn't hard-coded
# here rather than guess at a number that would quietly be wrong (§11-style
# open item). Swap in a specific one with `--reference-speed`/`--reference-rpm`
# once a source is confirmed.
#
# **--compare** overlays another run's pitch.csv (may be repeated) for
# comparing your own CFD runs against each other — a sweeper vs. a fastball
# you also simulated, or two resolutions of the same pitch.
#
# Same Makie install note as view_pitch.jl: it is not a dependency of this
# package, and cmd.exe does not take single quotes as quotes.
#
#   julia -e 'using Pkg; Pkg.add("GLMakie")'        # POSIX shells
#   julia -e "import Pkg; Pkg.add(\"GLMakie\")"     # cmd.exe / PowerShell

using BreakingBallLBM
using Printf
using Statistics: mean

# `moving_average` and `default_smoothing_window` live in the library
# (`src/postprocess/pitch_csv.jl`), not here, so they are covered by
# `test/test_pitch_csv.jl` without needing Makie or a display.

Base.@kwdef mutable struct AnalyzeConfig
    csv::String = "pitch.csv"
    backend::String = "glmakie"
    record_to::String = ""
    smooth::Int = 0        # 0: pick a window from the data length
    seam_samples::Int = 200
    ball_scale::Float64 = 15.0
    compare::Vector{String} = String[]
    reference::String = ""       # a PITCH_TYPES name, or "" for none
    reference_speed::Float64 = NaN   # NaN: take the named pitch's own
    reference_rpm::Float64 = NaN
    fps::Float64 = 30.0
end

const REQUIRED_COLUMNS = (:t, :x, :y, :z, :speed, :CD, :CL, :Cside, :rpm,
                          :u_in_x, :u_in_y, :u_in_z, :qw, :qx, :qy, :qz,
                          :wx, :wy, :wz, :recuts, :residual)

# 1 mph = 0.44704 m/s exactly, by definition — not an approximation.
const MPH_PER_MPS = 1 / 0.44704

function parse_args(args)
    c = AnalyzeConfig()
    positional_taken = false
    i = 1
    while i <= length(args)
        a = args[i]
        take() = i < length(args) ? (i += 1; args[i]) :
                 error("option $a needs a value — try --help")
        if a == "--help"
            println("""
            analyze_pitch.jl [CSV] [options]
              CSV                 trajectory file to read (default $(c.csv)), may
                                  also be given as --csv FILE
              --csv FILE          same, as a named option
              --smooth N          moving-average window for the coefficient panel,
                                  in sub-cycles (default: chosen from the row count)
              --seam-samples N    points on the drawn seam curve (default $(c.seam_samples))
              --ball-scale S      draw the ball S times life size at startup — also
                                  adjustable live with the "ball ×" slider
                                  (default $(c.ball_scale))
              --compare FILE      overlay another run's trajectory CSV (repeatable)
              --reference NAME    overlay a generic analytic-model pitch: one of
                                  $(join((t.name for t in PITCH_TYPES), ", "))
                                  (a typical parametrization, not a specific
                                  pitcher — see the header comment)
              --reference-speed V override the reference pitch's release speed, m/s
              --reference-rpm R   override the reference pitch's spin rate
              --fps N             auto-play frame rate (default $(c.fps))
              --web               WGLMakie in a browser instead of GLMakie in a window
              --backend NAME      glmakie (a window), wglmakie (a browser), or
                                  cairomakie (a still file, no display at all)
              --record FILE       write a still image instead of opening a window
            Coordinates: x toward the plate, z up, y to the pitcher's left.
            The seam and spin axis drawn in the 3-D panel come from the run's own
            qw,qx,qy,qz — reconstructed orientation, not redrawn from scratch.""")
            exit(0)
        elseif a == "--csv";              c.csv = take()
        elseif a == "--smooth";           c.smooth = parse(Int, take())
        elseif a == "--seam-samples";     c.seam_samples = parse(Int, take())
        elseif a == "--ball-scale";       c.ball_scale = parse(Float64, take())
        elseif a == "--compare";          push!(c.compare, take())
        elseif a == "--reference";        c.reference = take()
        elseif a == "--reference-speed";  c.reference_speed = parse(Float64, take())
        elseif a == "--reference-rpm";    c.reference_rpm = parse(Float64, take())
        elseif a == "--fps";              c.fps = parse(Float64, take())
        elseif a == "--web";              c.backend = "wglmakie"
        elseif a == "--backend";          c.backend = lowercase(take())
        elseif a == "--record";           c.record_to = take()
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

const CFG = parse_args(ARGS)

if CFG.backend == "cairomakie" && isempty(CFG.record_to)
    error("--backend cairomakie draws no window: give it --record out.png")
end

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

"""A loaded comparison run: its own time origin, its own smoothing window."""
function load_run(path::AbstractString, smooth::Int)
    isfile(path) || error("no such file: $path")
    data = read_pitch_csv(path)
    require_columns(data, REQUIRED_COLUMNS...)
    n = length(data[:t])
    n > 1 || error("$path has $(n) row(s) — nothing to plot")
    t = data[:t] .- data[:t][1]
    window = smooth > 0 ? smooth : default_smoothing_window(n)
    return (path = path, data = data, t = t, n = n, window = window)
end

"""The analytic-model reference pitch (view_pitch.jl's own model), or nothing."""
function reference_run(c::AnalyzeConfig, release::NTuple{3,<:Real}, distance::Real)
    isempty(c.reference) && return nothing
    idx = findfirst(t -> t.name == c.reference, PITCH_TYPES)
    idx === nothing && error("unknown --reference \"$(c.reference)\" — one of: " *
                             join((t.name for t in PITCH_TYPES), ", "))
    spec = PITCH_TYPES[idx]
    props = BaseballProperties()
    aero = CoefficientAero(props; CD = 0.35, CL_slope = 1.0)
    speed0 = isnan(c.reference_speed) ? spec.speed : c.reference_speed
    rpm0 = isnan(c.reference_rpm) ? spec.rpm : c.reference_rpm
    s0 = BallState(; position = release, velocity = (speed0, 0.0, 0.0),
                   spin = spin_from_rpm(spec.axis, rpm0))
    traj = simulate_trajectory(s0, props, aero; distance = distance)
    traj[end] = state_at_distance(traj, props, aero, distance)
    return (name = spec.name, samples = samples(traj))
end

function main(c::AnalyzeConfig)
    run = load_run(c.csv, c.smooth)
    data, t, n, window = run.data, run.t, run.n, run.window

    @printf("%s: %d rows, t = %.4f – %.4f s (%.4f s), x = %.3f – %.3f m\n",
            c.csv, n, data[:t][1], data[:t][end], t[end], data[:x][1], data[:x][end])
    @printf("speed %.3f -> %.3f m/s, rpm %.1f -> %.1f, smoothing window %d sub-cycles\n",
            data[:speed][1], data[:speed][end], data[:rpm][1], data[:rpm][end], window)
    half = (n ÷ 2 + 1):n
    @printf("mean over 2nd half: C_D %.3f, C_L %.3f, C_side %.4f, residual %.4f (max %.4f)\n",
            mean(@view data[:CD][half]), mean(@view data[:CL][half]),
            mean(@view data[:Cside][half]), mean(@view data[:residual][half]),
            maximum(data[:residual]))

    compares = [load_run(p, c.smooth) for p in c.compare]
    for cmp in compares
        @printf("compare %s: %d rows, x = %.3f – %.3f m\n",
                cmp.path, cmp.n, cmp.data[:x][1], cmp.data[:x][end])
    end
    cmp_colors = Makie.wong_colors()
    cmp_label(p) = splitext(basename(p))[1]

    reference = reference_run(c, (data[:x][1], data[:y][1], data[:z][1]), data[:x][end])
    if reference !== nothing
        println("reference: ", reference.name,
                " (analytic model, typical parametrization — not a specific pitcher)")
    end
    println()

    geom = BaseballGeometry()

    fig = Figure(size = (1600, 1200))

    # --- 3-D trajectory, with the ball and its seam at a scrubbable frame ---
    ax3d = Axis3(fig[1:3, 1]; aspect = :data, title = basename(c.csv),
                 xlabel = "toward the plate (m)", ylabel = "pitcher's left (m)",
                 zlabel = "up (m)")
    path3 = [Point3f(data[:x][i], data[:y][i], data[:z][i]) for i in 1:n]
    lines!(ax3d, path3; color = :crimson, linewidth = 3, label = "CFD trajectory")
    for (k, cmp) in enumerate(compares)
        cd = cmp.data
        p3 = [Point3f(cd[:x][i], cd[:y][i], cd[:z][i]) for i in 1:cmp.n]
        lines!(ax3d, p3; color = cmp_colors[mod1(k, length(cmp_colors))],
              linewidth = 2, linestyle = :dash, label = cmp_label(cmp.path))
    end
    if reference !== nothing
        rs = reference.samples
        lines!(ax3d, [Point3f(rs.x[i], rs.y[i], rs.z[i]) for i in 1:length(rs)];
              color = (:gray30, 0.8), linewidth = 2, linestyle = :dot,
              label = reference.name * " (reference)")
    end
    lines!(ax3d, [Point3f(p...) for p in plate_box()]; color = :black, linewidth = 2)
    axislegend(ax3d; position = :lt, framevisible = false)

    frame = Observable(n)     # start at the end: the whole flight is already drawn
    slider = Slider(fig[4, 1:3], range = 1:n, startvalue = n)
    on(v -> frame[] = v, slider.value)

    ball_scale_obs = Observable(c.ball_scale)
    radius_obs = lift(s -> Float32(geom.radius * s), ball_scale_obs)
    ballpos = lift(i -> path3[i], frame)
    mesh!(ax3d, lift((p, r) -> Sphere(p, r), ballpos, radius_obs);
          color = (:white, 0.5), transparency = true)
    orientation(i) = Quat(data[:qw][i], data[:qx][i], data[:qy][i], data[:qz][i])
    lines!(ax3d, lift((i, s) -> [Point3f(p...) for p in
                                 seam_world(geom, orientation(i),
                                           (data[:x][i], data[:y][i], data[:z][i]);
                                           samples = c.seam_samples, scale = s)],
                      frame, ball_scale_obs); color = :firebrick, linewidth = 3)
    linesegments!(ax3d, lift((i, r) -> begin
                                 b = BallState(; position = (data[:x][i], data[:y][i], data[:z][i]),
                                              velocity = (0.0, 0.0, 0.0),
                                              spin = (data[:wx][i], data[:wy][i], data[:wz][i]))
                                 a, e = spin_axis_world(b, 4 * r)
                                 [Point3f(a...), Point3f(e...)]
                             end, frame, radius_obs); color = :darkorange, linewidth = 2)

    # --- catcher's view: the plane break is quoted in ---
    axc = Axis(fig[1, 2]; title = "from the catcher", xlabel = "pitcher's left (m)",
              ylabel = "up (m)", aspect = DataAspect())
    lines!(axc, data[:y], data[:z]; color = :crimson, linewidth = 2)
    for (k, cmp) in enumerate(compares)
        lines!(axc, cmp.data[:y], cmp.data[:z];
              color = cmp_colors[mod1(k, length(cmp_colors))], linewidth = 2, linestyle = :dash)
    end
    reference !== nothing && lines!(axc, reference.samples.y, reference.samples.z;
                                    color = (:gray30, 0.8), linewidth = 2, linestyle = :dot)
    lines!(axc, [p[2] for p in plate_box()], [p[3] for p in plate_box()];
          color = :black, linewidth = 2)
    scatter!(axc, lift(i -> Point2f(data[:y][i], data[:z][i]), frame);
             color = :crimson, markersize = 14)

    # --- side view: true scale, rubber to plate, ground up — a real-world
    #     reference frame rather than one autoscaled to whatever this run's
    #     own --distance happened to be ---
    axs = Axis(fig[1, 3]; title = "from the side", xlabel = "toward the plate (m)",
              ylabel = "up (m)", aspect = DataAspect())
    lines!(axs, data[:x], data[:z]; color = :crimson, linewidth = 2)
    for (k, cmp) in enumerate(compares)
        lines!(axs, cmp.data[:x], cmp.data[:z];
              color = cmp_colors[mod1(k, length(cmp_colors))], linewidth = 2, linestyle = :dash)
    end
    reference !== nothing && lines!(axs, reference.samples.x, reference.samples.z;
                                    color = (:gray30, 0.8), linewidth = 2, linestyle = :dot)
    scatter!(axs, lift(i -> Point2f(data[:x][i], data[:z][i]), frame);
              color = :crimson, markersize = 12)
    xlims!(axs, 0, PLATE_DISTANCE)
    ylims!(axs, 0, nothing)

    # --- inflow velocity: u_in_x dwarfs u_in_y/u_in_z, so they get their own
    #     independently-scaled axis rather than being flattened by the shared one ---
    axin = Axis(fig[2, 2]; title = "inlet condition (u_in = -v, world frame)",
               xlabel = "t (s)", ylabel = "u_in_x (m/s)")
    l_inx = lines!(axin, t, data[:u_in_x]; color = :steelblue)
    axin2 = Axis(fig[2, 2]; ylabel = "u_in_y, u_in_z (m/s)", yaxisposition = :right,
                ygridvisible = false)
    hidespines!(axin2); hidexdecorations!(axin2)
    l_iny = lines!(axin2, t, data[:u_in_y]; color = :seagreen)
    l_inz = lines!(axin2, t, data[:u_in_z]; color = :goldenrod)
    vlines!(axin, lift(i -> t[i], frame); color = (:black, 0.3))
    axislegend(axin, [l_inx], ["u_in_x"]; position = :lt, framevisible = false, fontsize = 10)
    axislegend(axin2, [l_iny, l_inz], ["u_in_y", "u_in_z"]; position = :rb,
              framevisible = false, fontsize = 10)

    # --- speed and spin: raw values on their own axes, not a decay ratio.
    #     Speed in mph on the graph (§ballpark convention); the console
    #     summary above stays in m/s, the solver's own unit. ---
    axsp = Axis(fig[2, 3]; title = "speed and spin", xlabel = "t (s)", ylabel = "|V| (mph)")
    l_v = lines!(axsp, t, data[:speed] .* MPH_PER_MPS; color = :steelblue)
    axsp2 = Axis(fig[2, 3]; ylabel = "spin (rpm)", yaxisposition = :right, ygridvisible = false)
    hidespines!(axsp2); hidexdecorations!(axsp2)
    l_rpm = lines!(axsp2, t, data[:rpm]; color = :firebrick)
    for (k, cmp) in enumerate(compares)
        col = cmp_colors[mod1(k, length(cmp_colors))]
        lines!(axsp, cmp.t, cmp.data[:speed] .* MPH_PER_MPS; color = col, linestyle = :dash)
        lines!(axsp2, cmp.t, cmp.data[:rpm]; color = (col, 0.6), linestyle = :dash)
    end
    vlines!(axsp, lift(i -> t[i], frame); color = (:black, 0.3))
    sp_text = lift(i -> @sprintf("|V| = %.1f mph   spin = %.0f rpm",
                                 data[:speed][i] * MPH_PER_MPS, data[:rpm][i]), frame)
    text!(axsp, 0.02, 0.98; text = sp_text, space = :relative,
          align = (:left, :top), fontsize = 13)
    axislegend(axsp, [l_v], ["|V|"]; position = :lt, framevisible = false, fontsize = 10)
    axislegend(axsp2, [l_rpm], ["rpm"]; position = :rb, framevisible = false, fontsize = 10)

    # --- coefficients: raw (faint) and smoothed (bold), plus the current value ---
    axco = Axis(fig[3, 2]; title = "force coefficients (raw + $(window)-point average)",
               xlabel = "t (s)")
    for (name, col, colr) in (("C_D", data[:CD], :steelblue),
                             ("C_L", data[:CL], :seagreen),
                             ("C_side", data[:Cside], :firebrick))
        lines!(axco, t, col; color = (colr, 0.25), linewidth = 1)
        lines!(axco, t, moving_average(col, window); color = colr, linewidth = 2,
              label = name)
    end
    for (k, cmp) in enumerate(compares)
        lines!(axco, cmp.t, moving_average(cmp.data[:CD], cmp.window);
              color = cmp_colors[mod1(k, length(cmp_colors))], linewidth = 2,
              linestyle = :dash, label = cmp_label(cmp.path) * " C_D")
    end
    hlines!(axco, [0.0]; color = (:black, 0.4), linewidth = 1)
    vlines!(axco, lift(i -> t[i], frame); color = (:black, 0.3))
    co_text = lift(i -> @sprintf("C_D = %.3f   C_L = %.3f   C_side = %.4f",
                                 data[:CD][i], data[:CL][i], data[:Cside][i]), frame)
    text!(axco, 0.02, 0.98; text = co_text, space = :relative,
          align = (:left, :top), fontsize = 13)
    axislegend(axco; position = :rt, framevisible = false, fontsize = 10)

    # --- solver health: drift/residual and cumulative re-cuts ---
    axh = Axis(fig[3, 3]; title = "solver health", xlabel = "t (s)",
              ylabel = "residual")
    lines!(axh, t, data[:residual]; color = :steelblue)
    axh2 = Axis(fig[3, 3]; ylabel = "re-cuts (cumulative)", yaxisposition = :right,
               ygridvisible = false)
    hidespines!(axh2); hidexdecorations!(axh2)
    lines!(axh2, t, data[:recuts]; color = (:darkorange, 0.7))

    if !isempty(c.record_to)
        frame[] = n
        save(c.record_to, fig)
        println("wrote ", c.record_to)
        return
    end

    # --- playback and camera controls ---
    controls = GridLayout(fig[5, 1:3])
    reset_btn = Button(controls[1, 1]; label = "Reset")
    play_btn = Button(controls[1, 2]; label = "Play")
    mlb_btn = Button(controls[1, 3]; label = "MLB view")
    Label(controls[1, 4], "azimuth"; tellwidth = false)
    az_slider = Slider(controls[1, 5]; range = 0:1:359,
                       startvalue = round(Int, rad2deg(ax3d.azimuth[])) % 360)
    Label(controls[1, 6], "elevation"; tellwidth = false)
    el_slider = Slider(controls[1, 7]; range = -60:1:60,
                       startvalue = round(Int, rad2deg(ax3d.elevation[])))
    Label(controls[1, 8], "ball ×"; tellwidth = false)
    scale_slider = Slider(controls[1, 9]; range = 1:1:60, startvalue = round(Int, c.ball_scale))

    on(v -> ax3d.azimuth[] = deg2rad(v), az_slider.value)
    on(v -> ax3d.elevation[] = deg2rad(v), el_slider.value)
    on(v -> ball_scale_obs[] = Float64(v), scale_slider.value)

    playing = Observable(false)
    on(reset_btn.clicks) do _
        playing[] = false
        play_btn.label[] = "Play"
        set_close_to!(slider, 1)
    end
    on(play_btn.clicks) do _
        playing[] = !playing[]
        play_btn.label[] = playing[] ? "Pause" : "Play"
    end
    # Best-guess camera for the classic center-field broadcast angle: behind
    # the pitcher (looking down the +x flight path, y ~ 0) and a shallow
    # downward tilt from a high, distant camera. Unverified — this project's
    # Makie install is blocked in the dev sandbox (no display to check
    # against), so nudge the sliders to match a real broadcast still and this
    # button's target will get corrected to match.
    MLB_AZIMUTH_DEG = 180
    MLB_ELEVATION_DEG = 10
    on(mlb_btn.clicks) do _
        set_close_to!(az_slider, MLB_AZIMUTH_DEG)
        set_close_to!(el_slider, MLB_ELEVATION_DEG)
    end

    # A plain `while events(fig).window_open[] ... end` looked right but, on
    # whatever Makie version actually ran this, threw on its very first check
    # and — being wrapped in a try/catch that rethrows — silently killed the
    # task before the loop body (the part that advances the frame) ever ran
    # once: the slider still worked (it's driven by its own `on`, unrelated
    # to this task) but Play did nothing. Checking the window per iteration,
    # with a fallback that keeps playing if the check itself is unreliable,
    # means a bad check can no longer stop playback from working at all.
    @async begin
        while true
            open = try
                events(fig).window_open[]
            catch
                true
            end
            open || break
            if playing[]
                nxt = slider.value[] < n ? slider.value[] + 1 : 1
                set_close_to!(slider, nxt)
            end
            sleep(1.0 / max(c.fps, 1.0))
        end
    end

    display(fig)
    println("Drag in the 3-D panel to rotate, scroll to zoom, drag the slider",
            " to scrub through the flight.")
    println("Reset/Play control playback; the azimuth/elevation sliders aim the",
            " 3-D camera and \"MLB view\" jumps to a broadcast-angle starting guess.")
    println("Press Enter to close.")
    readline()
end

main(CFG)
