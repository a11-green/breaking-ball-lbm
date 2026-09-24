#!/usr/bin/env julia
#
# Print pfx/induced/total break from a real CFD trajectory CSV (§8 V&V-5).
#
# The measurement itself — re-integrating with this run's own measured C_D(t)
# and lift/side switched off — lives in the library
# (`src/postprocess/measured_break.jl`, `measure_break`), not here, so it is
# covered by `test/test_measured_break.jl` and shared with
# `scripts/plot_break.jl` instead of being copied between the two.
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
            force switched off — see src/postprocess/measured_break.jl for why.
            scripts/plot_break.jl draws the same numbers as a chart.""")
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

function main(c::BreakConfig)
    isfile(c.csv) || error("no such file: $(c.csv)")
    data = read_pitch_csv(c.csv)
    b = measure_break(data; dt = c.dt)

    @printf("%s: %d rows, release x=%.3f m (t=%.4f s) -> plate x=%.4f m (t=%.4f s)\n",
            c.csv, length(data[:t]), b.release_x, b.release_t, b.plate_x, b.plate_t)
    @printf("release speed %.2f m/s, plate speed %.2f m/s\n\n",
            b.release_speed, b.plate_speed)

    println("break (this run, measured C_D, lift+side removed from the branch point on):")
    @printf("  %-10s %10s %10s   %10s %10s\n", "definition", "horiz (m)", "horiz (in)",
            "vert (m)", "vert (in)")
    for (name, h, v) in (("pfx", b.pfx_horizontal, b.pfx_vertical),
                        ("induced", b.induced_horizontal, b.induced_vertical),
                        ("total", b.total_horizontal, b.total_vertical))
        @printf("  %-10s %10.4f %10.1f   %10.4f %10.1f\n", name, h, inches(h), v, inches(v))
    end
    println()
    println("for comparison (§8 V&V-5):")
    println("  reported (CBS Sports / SI): horiz 17.0 in, vert 32.0 in (no definition stated)")
    println("  analytic coefficient model: pfx 15.4/-0.1 in, induced 27.4/-0.1 in, total 27.4/-36.9 in")
end

main(parse_args(ARGS))
