#!/usr/bin/env julia
#
# What a grid choice costs and what it actually resolves (§6.5).
#
# The design document originally claimed that 40 points per diameter would catch
# the 0.79 mm seam ridge with "two or three points". It does not: Δx is 1.87 mm
# there, so the ridge is 0.42 of one cell. This script exists because that
# mistake would drive the wrong resolution choice, and because the trade it hides
# — seam resolution against domain width, at the third power — is the decision
# that most affects whether the answer means anything.
#
#   julia --project=. scripts/resolution_budget.jl

using BreakingBallLBM
using Printf

const VRAM_GIB = 6.5        # what is usable of the 3060 Ti's 8 GiB
const MLUPS = 2300          # measured, wall-bounded FP32 (scripts/benchmark_gpu.jl)

function table(resolutions, domains)
    @printf("%-6s %-5s %-7s %-8s %-9s %-10s %-9s %-10s %-8s\n",
            "N/D", "L/D", "edge", "VRAM", "dx (mm)", "seam/dx", "BL/dx", "blockage", "hours")
    for L in domains, N in resolutions
        b = grid_budget(; nodes_per_diameter = N, domain_diameters = L, mlups = MLUPS)
        fits = b.gib <= VRAM_GIB
        @printf("%-6d %-5.1f %-7d %-8s %-9.3f %-10.2f %-9.2f %-10.3f %s\n",
                N, L, b.edge, fits ? @sprintf("%.2f G", b.gib) : "over",
                b.dx_mm, b.seam_cells, b.boundary_layer_cells, b.blockage,
                fits ? @sprintf("%-8.1f", b.hours) : "-")
    end
end

function main()
    h = 0.00079
    δ = boundary_layer_thickness()
    @printf("Seam ridge          %.2f mm\n", 1000h)
    @printf("Boundary layer      %.2f mm  (laminar, at the equator, 39 m/s)\n", 1000δ)
    println()
    println("These are the same number to within 5%. A roughness element one boundary")
    println("layer tall is exactly what trips transition, which is why a baseball has no")
    println("sharp drag crisis (§1.3) — and it means resolving the seam and resolving the")
    println("boundary layer are one requirement, not two. Both columns below move together.")
    println()

    println("Uniform grid, cubic domain of L diameters")
    table((40, 50, 60, 80, 100, 120), (8, 6, 4, 3))
    println()
    @printf("Nothing here reaches one cell across the seam without shrinking the domain to\n")
    @printf("three diameters, where the sphere occupies a sixth of the box — the blockage\n")
    @printf("V&V-2 measured against Hasimoto, i.e. a periodic array of baseballs rather\n")
    @printf("than one. And note the hours column: at 8D/40 the run is under an hour on\n")
    @printf("about half the VRAM. **Memory is the binding constraint, not speed.**\n\n")

    # What refinement buys, at the same memory.
    println("Two levels of 2x refinement over a box of F diameters around the ball")
    @printf("%-8s %-6s %-12s %-10s %-9s %-10s %-8s\n",
            "coarse", "F/D", "fine N/D", "fine dx", "seam/dx", "VRAM", "hours")
    for F in (1.5, 2.0, 3.0), N in (32, 40, 48)
        fine_N = 4N
        coarse_nodes = (8N)^3
        fine_nodes = round(Int, (F * fine_N)^3)
        nodes = coarse_nodes + fine_nodes
        gib = nodes * 112 / 2^30
        dxf = 0.0748 / fine_N
        # Acoustic scaling: each level halves the step, so the fine region is
        # stepped four times per coarse step.
        updates = coarse_nodes + 4 * fine_nodes
        base = grid_budget(; nodes_per_diameter = N, domain_diameters = 8, mlups = MLUPS)
        hours = updates * base.steps / (MLUPS * 1e6) / 3600
        @printf("%-8d %-6.1f %-12d %-10.3f %-9.2f %-10s %-8.1f\n",
                N, F, fine_N, 1000dxf, 0.00079 / dxf,
                gib <= VRAM_GIB ? @sprintf("%.2f G", gib) : @sprintf("over (%.1f G)", gib),
                hours)
    end
    println()
    println("Refinement is the only way to hold an eight-diameter domain and a seam worth")
    println("the name at the same time. The best row that fits is 40 points per diameter")
    println("coarse with 160 over a box of one and a half diameters: 1.69 cells across the")
    println("ridge in 4.9 GiB and two hours, against 0.42 cells for the uniform grid of the")
    println("same domain. That moves local grid refinement out of the optimisation column")
    println("and into the requirements.")
end

main()
