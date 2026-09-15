#!/usr/bin/env julia
#
# Render the parametric seam as an SVG so its shape can be eyeballed against
# photographs of a real ball (docs/design/DESIGN.md §4.1 calls for exactly this
# calibration). Writes three orthographic views; the near half of the curve is
# drawn solid and the far half dashed.
#
#   julia --project=. scripts/plot_seam.jl [output.svg] [amplitude]

using BreakingBallLBM

const PANEL = 260      # px per panel
const MARGIN = 26
const RBALL = 96       # px

project(p, view) = view === :z ? (p[1], -p[2], p[3]) :
                   view === :x ? (p[2], -p[3], p[1]) :
                                 (p[1], -p[3], p[2])

function panel(io, seam, view, title, subtitle, x0, n)
    cx, cy = x0 + PANEL / 2, MARGIN + PANEL / 2
    scale = RBALL / seam.radius

    println(io, """<circle cx="$cx" cy="$cy" r="$RBALL" fill="#f4f1ea" stroke="#c9c2b4" stroke-width="1.5"/>""")

    pts = [project(seam_point(seam, 2π * (i - 1) / n), view) for i in 1:n]
    push!(pts, pts[1])
    px(v) = round(v, digits = 2)
    screen(p) = "$(px(cx + p[1] * scale)),$(px(cy + p[2] * scale))"

    # Emit one polyline per run of segments on the same side of the ball, so the
    # near half can be drawn solid and the far half dashed.
    run_start = 1
    near(i) = (pts[i][3] + pts[i+1][3]) / 2 > 0
    for i in 1:n
        if i == n || near(i) != near(i + 1)
            coords = join((screen(pts[k]) for k in run_start:(i+1)), " ")
            println(io, """<polyline points="$coords" class="$(near(i) ? "near" : "far")"/>""")
            run_start = i + 1
        end
    end

    ty = MARGIN + PANEL + 24
    println(io, """<text x="$cx" y="$ty" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#2b2b2b">$title</text>""")
    println(io, """<text x="$cx" y="$(ty + 19)" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#6b6b6b">$subtitle</text>""")
end

function main()
    out = length(ARGS) >= 1 ? ARGS[1] : "seam.svg"
    amplitude = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.7
    seam = BaseballSeam(amplitude = amplitude)

    width = 3 * PANEL + 4 * MARGIN
    height = PANEL + 3 * MARGIN + 30
    open(out, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<style>
          polyline { fill: none; }
          .near { stroke: #b3271e; stroke-width: 5; stroke-linecap: round; stroke-linejoin: round; }
          .far  { stroke: #d99b95; stroke-width: 3; stroke-dasharray: 4 4; }
        </style>""")
        println(io, """<rect width="$width" height="$height" fill="#ffffff"/>""")
        panel(io, seam, :z, "view along z", "spin about z: 4 seams/rev", MARGIN, 600)
        panel(io, seam, :x, "view along x", "spin about x: 2 seams/rev", 2MARGIN + PANEL, 600)
        panel(io, seam, :y, "view along y", "spin about y: 2 seams/rev", 3MARGIN + 2PANEL, 600)
        println(io, "</svg>")
    end

    L = seam_length(seam)
    println("wrote $out  (amplitude=$amplitude, radius=$(seam.radius) m, seam length=$(round(L * 1000, digits = 1)) mm)")
end

main()
