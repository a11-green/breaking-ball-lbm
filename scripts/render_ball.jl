#!/usr/bin/env julia
#
# Generate a ball mesh (a sphere) matching every frame of a run's seam series,
# so ParaView can show the sphere the seam sits on instead of lines floating in
# space (§7.4). This is post-processing only: it reads files a run already
# wrote (`seam.vtk.series` and the `seam-*.vtk` frames `run_pitch.jl` writes
# with `--snapshot N`) and needs neither the GPU nor a re-run.
#
#   julia --project=. scripts/render_ball.jl
#   julia --project=. scripts/render_ball.jl --snapshot-dir snapshots
#   julia --project=. scripts/render_ball.jl --slices 48 --stacks 32
#   julia --project=. scripts/render_ball.jl --help
#
# Writes ball-#####.vtk (one per seam frame, same numbering) and a matching
# ball.vtk.series, in the same directory. Open flow.vtk.series, seam.vtk.series
# and ball.vtk.series together in ParaView: same names, same physical-time
# axis, so scrubbing one scrubs all three.
#
# **Why the ball's centre and radius come from the seam file, not a config.**
# The seam curve sits exactly on the ball's surface by construction
# (`seam_world` rotates and translates a curve that starts on a sphere of the
# ball's own radius), so the ball this run actually used is recoverable from
# the seam alone — no need to know the run's resolution, scale, or which
# script produced the file. `ball_from_seam` does the recovery.

using BreakingBallLBM
using Printf

Base.@kwdef mutable struct BallConfig
    snapshot_dir::String = "snapshots"
    slices::Int = 24
    stacks::Int = 16
end

function parse_args(args)
    c = BallConfig()
    i = 1
    while i <= length(args)
        a = args[i]
        take() = i < length(args) ? (i += 1; args[i]) :
                 error("option $a needs a value — try --help")
        if a == "--help"
            println("""
            render_ball.jl [options]
              --snapshot-dir DIR  where seam.vtk.series and seam-*.vtk already
                                  live, and where ball-*.vtk / ball.vtk.series
                                  are written (default $(c.snapshot_dir))
              --slices N          longitude divisions of the sphere mesh
                                  (default $(c.slices))
              --stacks N          latitude divisions (default $(c.stacks))
            The ball's centre and radius are read back off each seam frame, not
            passed in — see the header comment. Run this after run_pitch.jl has
            finished writing --snapshot frames; it does not touch the GPU.""")
            exit(0)
        elseif a == "--snapshot-dir"; c.snapshot_dir = take()
        elseif a == "--slices";       c.slices = parse(Int, take())
        elseif a == "--stacks";       c.stacks = parse(Int, take())
        else
            error("unknown option $a — try --help")
        end
        i += 1
    end
    return c
end

function main(c::BallConfig)
    series_path = joinpath(c.snapshot_dir, "seam.vtk.series")
    isfile(series_path) ||
        error("no such file: $series_path — run_pitch.jl writes this alongside " *
              "the seam frames when given --snapshot N")
    frames = read_vtk_series(series_path)
    isempty(frames) && error("$series_path names no frames")

    out_series = Tuple{String,Float64}[]
    out_series_path = joinpath(c.snapshot_dir, "ball.vtk.series")

    println("Reading  ", series_path, " (", length(frames), " frame(s))")
    for (n, f) in enumerate(frames)
        seam_path = joinpath(c.snapshot_dir, f.name)
        isfile(seam_path) ||
            error("$series_path names $(f.name), which is not in $(c.snapshot_dir)")
        lines, ids, _ = read_polylines(seam_path)
        centre, radius = ball_from_seam(lines, ids)

        # Same numbering as the seam frame it matches, so the two sort and
        # pair up identically in a plain directory listing too.
        ball_name = replace(f.name, "seam-" => "ball-")
        ball_path = joinpath(c.snapshot_dir, ball_name)
        write_ball(ball_path, centre, radius; slices = c.slices, stacks = c.stacks,
                  title = @sprintf("ball, t = %.4f s", f.time))
        push!(out_series, (ball_name, f.time))

        if n == 1 || n == length(frames) || n % 20 == 0
            @printf("  %4d / %-4d  %-16s centre (%.4f, %.4f, %.4f) m  radius %.5f m\n",
                    n, length(frames), ball_name, centre..., radius)
            flush(stdout)
        end
    end

    open(out_series_path, "w") do io
        println(io, "{")
        println(io, "  \"file-series-version\" : \"1.0\",")
        println(io, "  \"files\" : [")
        for (i, (name, t)) in enumerate(out_series)
            @printf(io, "    { \"name\" : \"%s\", \"time\" : %.6f }%s\n",
                    name, t, i == length(out_series) ? "" : ",")
        end
        println(io, "  ]")
        println(io, "}")
    end

    println()
    @printf("wrote %d ball frame(s) and %s\n", length(out_series), out_series_path)
    println("Open flow.vtk.series, seam.vtk.series and ball.vtk.series together in ",
            "ParaView — matching names on the same physical-time axis, so scrubbing ",
            "one scrubs all three.")
end

main(parse_args(ARGS))
