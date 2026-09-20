#!/usr/bin/env julia
#
# Generate a ball mesh (a sphere) to view alongside a run's seam series, so
# ParaView can show the sphere the seam sits on instead of lines floating in
# space (§7.4). This is post-processing only: it reads a file a run already
# wrote (one frame of the `seam-*.vtk` series `run_pitch.jl` writes with
# `--snapshot N`) and needs neither the GPU nor a re-run.
#
#   julia --project=. scripts/render_ball.jl
#   julia --project=. scripts/render_ball.jl --snapshot-dir snapshots
#   julia --project=. scripts/render_ball.jl --slices 48 --stacks 32
#   julia --project=. scripts/render_ball.jl --help
#
# Writes a single ball.vtk in the snapshot directory — not a series, not one
# file per frame. §4.4.2's ball-following frame keeps the box fixed to the
# ball's centre of mass (`src/trajectory/frame.jl`): the ball spins in place
# and never translates across the grid, so every seam frame reports the same
# centre and radius (this script checks that and errors out if a run ever
# violates it). Only the seam moves, which is exactly what its own series is
# for. Open ball.vtk once in ParaView alongside flow.vtk.series and
# seam.vtk.series: a plain (non-series) source just sits there through the
# whole animation while the other two play.
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
                                  live, and where ball.vtk is written
                                  (default $(c.snapshot_dir))
              --slices N          longitude divisions of the sphere mesh
                                  (default $(c.slices))
              --stacks N          latitude divisions (default $(c.stacks))
            Writes one ball.vtk, not a series — the ball never translates in
            the ball-following frame (§4.4.2), only the seam does, so a
            single static mesh is correct for the whole run. The centre and
            radius are read back off a seam frame, not passed in — see the
            header comment. Run this after run_pitch.jl has finished writing
            --snapshot frames; it does not touch the GPU.""")
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

function ball_geometry(c::BallConfig, series_path::AbstractString)
    frames = read_vtk_series(series_path)
    isempty(frames) && error("$series_path names no frames")

    first_path = joinpath(c.snapshot_dir, frames[1].name)
    isfile(first_path) ||
        error("$series_path names $(frames[1].name), which is not in $(c.snapshot_dir)")
    lines, ids, _ = read_polylines(first_path)
    centre, radius = ball_from_seam(lines, ids)

    if length(frames) > 1
        last_path = joinpath(c.snapshot_dir, frames[end].name)
        isfile(last_path) ||
            error("$series_path names $(frames[end].name), which is not in $(c.snapshot_dir)")
        lines2, ids2, _ = read_polylines(last_path)
        centre2, radius2 = ball_from_seam(lines2, ids2)
        drift = sqrt(sum(abs2, centre .- centre2))
        # The ball-following frame (§4.4.2) keeps this at zero by design; a
        # few percent of a cell is normal float noise from a finite point
        # count, anything larger means the ball translated and one static
        # mesh is the wrong answer for this run.
        tol = 0.02 * radius
        if drift > tol || abs(radius - radius2) > tol
            error("the ball moved between the first and last snapshot " *
                  "(centre $centre -> $centre2, radius $radius -> $radius2) — " *
                  "this run's frame is not fixed to the ball, so a single " *
                  "ball.vtk is not correct here; each seam frame would need " *
                  "its own ball mesh instead")
        end
    end

    return centre, radius, length(frames)
end

function main(c::BallConfig)
    series_path = joinpath(c.snapshot_dir, "seam.vtk.series")
    isfile(series_path) ||
        error("no such file: $series_path — run_pitch.jl writes this alongside " *
              "the seam frames when given --snapshot N")

    centre, radius, nframes = ball_geometry(c, series_path)

    ball_path = joinpath(c.snapshot_dir, "ball.vtk")
    write_ball(ball_path, centre, radius; slices = c.slices, stacks = c.stacks,
              title = "ball")

    @printf("checked %d frame(s): centre stays at (%.4f, %.4f, %.4f) m, radius %.5f m\n",
            nframes, centre..., radius)
    println("wrote ", ball_path)
    println("Open flow.vtk.series and seam.vtk.series as usual, and ball.vtk ",
            "once alongside them (not as a series — it does not change over ",
            "the run). ParaView shows a plain source through the whole ",
            "animation while the two series play.")
end

main(parse_args(ARGS))
