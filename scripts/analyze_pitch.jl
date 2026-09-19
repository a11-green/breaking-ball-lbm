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
#   julia --project=. scripts/analyze_pitch.jl --help
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
end

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
              CSV              trajectory file to read (default $(c.csv)), may
                               also be given as --csv FILE
              --csv FILE       same, as a named option
              --smooth N       moving-average window for the coefficient panel,
                               in sub-cycles (default: chosen from the row count)
              --seam-samples N points on the drawn seam curve (default $(c.seam_samples))
              --ball-scale S   draw the ball S times life size in the 3-D panel
                               (default $(c.ball_scale))
              --web            WGLMakie in a browser instead of GLMakie in a window
              --backend NAME   glmakie (a window), wglmakie (a browser), or
                               cairomakie (a still file, no display at all)
              --record FILE    write a still image instead of opening a window
            Coordinates: x toward the plate, z up, y to the pitcher's left.
            The seam and spin axis drawn in the 3-D panel come from the run's own
            qw,qx,qy,qz — reconstructed orientation, not redrawn from scratch.""")
            exit(0)
        elseif a == "--csv";          c.csv = take()
        elseif a == "--smooth";       c.smooth = parse(Int, take())
        elseif a == "--seam-samples"; c.seam_samples = parse(Int, take())
        elseif a == "--ball-scale";   c.ball_scale = parse(Float64, take())
        elseif a == "--web";          c.backend = "wglmakie"
        elseif a == "--backend";      c.backend = lowercase(take())
        elseif a == "--record";       c.record_to = take()
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

function main(c::AnalyzeConfig)
    isfile(c.csv) || error("no such file: $(c.csv)")
    data = read_pitch_csv(c.csv)
    require_columns(data, :t, :x, :y, :z, :speed, :CD, :CL, :Cside, :rpm,
                    :u_in_x, :u_in_y, :u_in_z, :qw, :qx, :qy, :qz,
                    :wx, :wy, :wz, :recuts, :residual)
    n = length(data[:t])
    n > 1 || error("$(c.csv) has $(n) row(s) — nothing to plot")

    t = data[:t] .- data[:t][1]
    window = c.smooth > 0 ? c.smooth : default_smoothing_window(n)

    @printf("%s: %d rows, t = %.4f – %.4f s (%.4f s), x = %.3f – %.3f m\n",
            c.csv, n, data[:t][1], data[:t][end], t[end], data[:x][1], data[:x][end])
    @printf("speed %.3f -> %.3f m/s, rpm %.1f -> %.1f, smoothing window %d sub-cycles\n",
            data[:speed][1], data[:speed][end], data[:rpm][1], data[:rpm][end], window)
    half = (n ÷ 2 + 1):n
    @printf("mean over 2nd half: C_D %.3f, C_L %.3f, C_side %.4f, residual %.4f (max %.4f)\n",
            mean(@view data[:CD][half]), mean(@view data[:CL][half]),
            mean(@view data[:Cside][half]), mean(@view data[:residual][half]),
            maximum(data[:residual]))
    println()

    geom = BaseballGeometry()
    R = Float32(geom.radius * c.ball_scale)

    fig = Figure(size = (1600, 1100))

    # --- 3-D trajectory, with the ball and its seam at a scrubbable frame ---
    ax3d = Axis3(fig[1:3, 1]; aspect = :data, title = basename(c.csv),
                 xlabel = "toward the plate (m)", ylabel = "pitcher's left (m)",
                 zlabel = "up (m)")
    path3 = [Point3f(data[:x][i], data[:y][i], data[:z][i]) for i in 1:n]
    lines!(ax3d, path3; color = :crimson, linewidth = 3, label = "CFD trajectory")
    lines!(ax3d, [Point3f(p...) for p in plate_box()]; color = :black, linewidth = 2)
    axislegend(ax3d; position = :lt, framevisible = false)

    frame = Observable(n)     # start at the end: the whole flight is already drawn
    ballpos = lift(i -> path3[i], frame)
    mesh!(ax3d, lift(p -> Sphere(p, R), ballpos); color = (:white, 0.5),
          transparency = true)
    orientation(i) = Quat(data[:qw][i], data[:qx][i], data[:qy][i], data[:qz][i])
    lines!(ax3d, lift(i -> [Point3f(p...) for p in
                            seam_world(geom, orientation(i),
                                      (data[:x][i], data[:y][i], data[:z][i]);
                                      samples = c.seam_samples, scale = c.ball_scale)],
                      frame); color = :firebrick, linewidth = 3)
    linesegments!(ax3d, lift(i -> begin
                                 b = BallState(; position = (data[:x][i], data[:y][i], data[:z][i]),
                                              velocity = (0.0, 0.0, 0.0),
                                              spin = (data[:wx][i], data[:wy][i], data[:wz][i]))
                                 a, e = spin_axis_world(b, 4 * R)
                                 [Point3f(a...), Point3f(e...)]
                             end, frame); color = :darkorange, linewidth = 2)

    # --- catcher's view: the plane break is quoted in ---
    axc = Axis(fig[1, 2]; title = "from the catcher", xlabel = "pitcher's left (m)",
              ylabel = "up (m)", aspect = DataAspect())
    lines!(axc, data[:y], data[:z]; color = :crimson, linewidth = 2)
    lines!(axc, [p[2] for p in plate_box()], [p[3] for p in plate_box()];
          color = :black, linewidth = 2)
    scatter!(axc, lift(i -> Point2f(data[:y][i], data[:z][i]), frame);
             color = :crimson, markersize = 14)

    # --- side view ---
    axs = Axis(fig[1, 3]; title = "from the side", xlabel = "toward the plate (m)",
              ylabel = "up (m)")
    lines!(axs, data[:x], data[:z]; color = :crimson, linewidth = 2)
    scatter!(axs, lift(i -> Point2f(data[:x][i], data[:z][i]), frame);
              color = :crimson, markersize = 12)

    # --- inflow velocity: what the request specifically asked to see ---
    axin = Axis(fig[2, 2]; title = "inlet condition (u_in = -v, world frame)",
               xlabel = "t (s)", ylabel = "m/s")
    lines!(axin, t, data[:u_in_x]; label = "u_in_x", color = :steelblue)
    lines!(axin, t, data[:u_in_y]; label = "u_in_y", color = :seagreen)
    lines!(axin, t, data[:u_in_z]; label = "u_in_z", color = :goldenrod)
    axislegend(axin; position = :rt, framevisible = false, fontsize = 10)

    # --- speed and spin ---
    axsp = Axis(fig[2, 3]; title = "speed and spin", xlabel = "t (s)")
    lines!(axsp, t, data[:speed] ./ data[:speed][1]; color = :steelblue,
          label = "|V| / V0")
    lines!(axsp, t, data[:rpm] ./ data[:rpm][1]; color = :firebrick,
          label = "rpm / rpm0")
    axislegend(axsp; position = :rt, framevisible = false, fontsize = 10)

    # --- coefficients: raw (faint) and smoothed (bold) ---
    axco = Axis(fig[3, 2]; title = "force coefficients (raw + $(window)-point average)",
               xlabel = "t (s)")
    for (name, col, colr) in (("C_D", data[:CD], :steelblue),
                             ("C_L", data[:CL], :seagreen),
                             ("C_side", data[:Cside], :firebrick))
        lines!(axco, t, col; color = (colr, 0.25), linewidth = 1)
        lines!(axco, t, moving_average(col, window); color = colr, linewidth = 2,
              label = name)
    end
    hlines!(axco, [0.0]; color = (:black, 0.4), linewidth = 1)
    axislegend(axco; position = :rt, framevisible = false, fontsize = 10)

    # --- solver health: drift/residual and cumulative re-cuts ---
    axh = Axis(fig[3, 3]; title = "solver health", xlabel = "t (s)",
              ylabel = "residual")
    lines!(axh, t, data[:residual]; color = :steelblue)
    axh2 = Axis(fig[3, 3]; ylabel = "re-cuts (cumulative)", yaxisposition = :right,
               ygridvisible = false)
    hidespines!(axh2); hidexdecorations!(axh2)
    lines!(axh2, t, data[:recuts]; color = (:darkorange, 0.7))

    if !isempty(CFG.record_to)
        save(CFG.record_to, fig)
        println("wrote ", CFG.record_to)
        return
    end

    slider = Slider(fig[4, 1:3], range = 1:n, startvalue = n)
    on(v -> frame[] = v, slider.value)
    display(fig)
    println("Drag in the 3-D panel to rotate, scroll to zoom, drag the slider",
            " to scrub through the flight.")
    println("Press Enter to close.")
    readline()
end

main(CFG)
