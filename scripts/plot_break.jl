#!/usr/bin/env julia
#
# Chart the pfx/induced/total break of a CFD run against the analytic
# coefficient model and the reported WBC 2023 final figures (§8 V&V-5).
#
# The three break numbers for this run come from `measure_break`
# (`src/postprocess/measured_break.jl`, also used by `scripts/measure_break.jl`
# for the plain-text version of the same numbers). The analytic-model bars are
# computed live from `pitch_metrics` on a named `PITCH_TYPES` pitch
# (`src/postprocess/pitch_view.jl`) rather than hard-coded, so they track the
# model if it or the pitch spec ever changes.
#
# **The reported figure (17 in horizontal, 32 in vertical) has no definition
# stated in the source** (CBS Sports / SI, DESIGN.md §8). DESIGN.md's own
# reading is that the horizontal number matches the pfx definition; the
# vertical number matches none of the three (§11, still open). This chart
# draws the reported values as a single reference line each, not tied to one
# bar, and leaves reading it to the person looking — pass --assume to change
# which definition the dashed line is labelled against.
#
#   julia --project=. scripts/plot_break.jl pitch.csv
#   julia --project=. scripts/plot_break.jl pitch.csv --assume pfx
#   julia --project=. scripts/plot_break.jl pitch.csv --backend cairomakie --record break.png
#   julia --project=. scripts/plot_break.jl --help
#
# Same Makie install note as analyze_pitch.jl — it is not a dependency of
# this package, and cmd.exe does not take single quotes as quotes.
#
#   julia -e 'using Pkg; Pkg.add("GLMakie")'        # POSIX shells
#   julia -e "import Pkg; Pkg.add(\"GLMakie\")"     # cmd.exe / PowerShell

using BreakingBallLBM
using Printf

# REPORTED_HORIZONTAL_IN / REPORTED_VERTICAL_IN (CBS Sports / SI, DESIGN.md §8
# V&V-5, no definition stated) come from BreakingBallLBM — shared with
# scripts/analyze_pitch.jl's catcher's-view marker rather than copied here.

Base.@kwdef mutable struct BreakPlotConfig
    csv::String = "pitch.csv"
    dt::Float64 = 1.0e-4
    reference::String = "sweeper"
    assume::String = "total"
    backend::String = "glmakie"
    record_to::String = ""
end

function parse_args(args)
    c = BreakPlotConfig()
    positional_taken = false
    i = 1
    while i <= length(args)
        a = args[i]
        take() = i < length(args) ? (i += 1; args[i]) :
                 error("option $a needs a value — try --help")
        if a == "--help"
            println("""
            plot_break.jl [CSV] [options]
              CSV              trajectory file to read (default $(c.csv)), may
                               also be given as --csv FILE
              --csv FILE       same, as a named option
              --dt T           integration step for the reference trajectories,
                               s (default $(c.dt))
              --reference NAME analytic-model pitch to draw alongside this run:
                               one of $(join((t.name for t in PITCH_TYPES), ", "))
                               (default "$(c.reference)")
              --assume DEF     which break definition the reported 17in/32in
                               line is drawn against — pfx, induced, or total
                               (default "$(c.assume)"); the source states no
                               definition, so this only changes the label
              --web            WGLMakie in a browser instead of GLMakie in a window
              --backend NAME   glmakie (a window), wglmakie (a browser), or
                               cairomakie (a still file, no display at all)
              --record FILE    write a still image instead of opening a window
            """)
            exit(0)
        elseif a == "--csv";       c.csv = take()
        elseif a == "--dt";        c.dt = parse(Float64, take())
        elseif a == "--reference"; c.reference = take()
        elseif a == "--assume";    c.assume = take()
        elseif a == "--web";       c.backend = "wglmakie"
        elseif a == "--backend";   c.backend = lowercase(take())
        elseif a == "--record";    c.record_to = take()
        elseif !startswith(a, "--") && !positional_taken
            c.csv = a
            positional_taken = true
        else
            error("unknown option $a — try --help")
        end
        i += 1
    end
    c.assume in ("pfx", "induced", "total") ||
        error("--assume must be pfx, induced or total, not \"$(c.assume)\" — try --help")
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

"""The analytic-model comparison bars, from `pitch_metrics` on a named
`PITCH_TYPES` pitch — live, not a copied-in table."""
function analytic_reference(name::AbstractString; dt::Real = 1.0e-4)
    idx = findfirst(t -> t.name == name, PITCH_TYPES)
    idx === nothing && error("unknown --reference \"$name\" — one of: " *
                            join((t.name for t in PITCH_TYPES), ", "))
    spec = PITCH_TYPES[idx]
    props = BaseballProperties()
    aero = CoefficientAero(props; CD = 0.35, CL_slope = 1.0)
    s0 = BallState(; position = (2.0, 0.0, 1.75), velocity = (spec.speed, 0.0, 0.0),
                   spin = spin_from_rpm(spec.axis, spec.rpm))
    return pitch_metrics(s0, props, aero; dt = dt)
end

function main(c::BreakPlotConfig)
    isfile(c.csv) || error("no such file: $(c.csv)")
    data = read_pitch_csv(c.csv)
    mb = measure_break(data; dt = c.dt)
    pm = analytic_reference(c.reference; dt = c.dt)

    cats = ["pfx", "induced", "total"]
    cfd_h = inches.((mb.pfx_horizontal, mb.induced_horizontal, mb.total_horizontal))
    cfd_v = inches.((mb.pfx_vertical, mb.induced_vertical, mb.total_vertical))
    ref_h = inches.((pm.pfx_horizontal, pm.induced_horizontal, pm.total_horizontal))
    ref_v = inches.((pm.pfx_vertical, pm.induced_vertical, pm.total_vertical))

    @printf("%s vs. %s analytic model, vs. reported (assumed ≡ %s):\n",
            c.csv, c.reference, c.assume)
    @printf("  %-10s %10s %10s   %10s %10s\n", "definition", "horiz(CFD)",
            "horiz(model)", "vert(CFD)", "vert(model)")
    for i in 1:3
        @printf("  %-10s %10.1f %10.1f   %10.1f %10.1f\n", cats[i], cfd_h[i], ref_h[i],
                cfd_v[i], ref_v[i])
    end
    println()

    fig = Figure(size = (1200, 650))

    x = Float64.(vcat(1:3, 1:3))
    dodge = vcat(fill(1, 3), fill(2, 3))
    colors = (:steelblue, :gray60)

    axh = Axis(fig[1, 1]; title = "horizontal break", ylabel = "inches",
              xticks = (1:3, cats))
    barplot!(axh, x, vcat(cfd_h, ref_h); dodge = dodge, n_dodge = 2,
            color = [colors[d] for d in dodge], width = 0.7)
    hlines!(axh, [REPORTED_HORIZONTAL_IN]; color = :firebrick, linewidth = 2,
           linestyle = :dash)
    text!(axh, 0.98, REPORTED_HORIZONTAL_IN;
         text = "reported $(REPORTED_HORIZONTAL_IN)\" (assumed ≡ $(c.assume))",
         align = (:right, :bottom), fontsize = 11, color = :firebrick,
         space = :data, offset = (0, 4))
    for (xi, h) in zip(x, vcat(cfd_h, ref_h))
        text!(axh, xi, h; text = @sprintf("%.1f", h), align = (:center, :bottom),
             fontsize = 10, offset = (0, h >= 0 ? 3 : -13))
    end

    axv = Axis(fig[1, 2]; title = "vertical break", ylabel = "inches",
              xticks = (1:3, cats))
    barplot!(axv, x, vcat(cfd_v, ref_v); dodge = dodge, n_dodge = 2,
            color = [colors[d] for d in dodge], width = 0.7)
    hlines!(axv, [REPORTED_VERTICAL_IN]; color = :firebrick, linewidth = 2,
           linestyle = :dash)
    text!(axv, 0.98, REPORTED_VERTICAL_IN;
         text = "reported $(REPORTED_VERTICAL_IN)\" (assumed ≡ $(c.assume))",
         align = (:right, :top), fontsize = 11, color = :firebrick,
         space = :data, offset = (0, -4))
    for (xi, v) in zip(x, vcat(cfd_v, ref_v))
        text!(axv, xi, v; text = @sprintf("%.1f", v), align = (:center, :bottom),
             fontsize = 10, offset = (0, v >= 0 ? 3 : -13))
    end

    steelblue_swatch = PolyElement(color = :steelblue)
    gray_swatch = PolyElement(color = :gray60)
    Legend(fig[2, 1:2], [steelblue_swatch, gray_swatch],
          ["this run (CFD, measured C_D)", "analytic model ($(c.reference))"];
          orientation = :horizontal, framevisible = false)

    if !isempty(c.record_to)
        save(c.record_to, fig)
        println("wrote ", c.record_to)
        return
    end

    display(fig)
    println("Press Enter to close.")
    readline()
end

main(CFG)
