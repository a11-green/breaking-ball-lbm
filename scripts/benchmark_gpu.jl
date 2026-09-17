#!/usr/bin/env julia
#
# P2: check the CUDA backend against the CPU reference, then measure how fast it
# actually runs, so the planning numbers in docs/design/DESIGN.md §6.5-6.6 can be
# replaced with measurements.
#
# This is the script to run on the RTX 3060 Ti. Development happened on a machine
# without a GPU, so the correctness checks here are the first time the kernels
# execute at all — run it before trusting any timing it prints.
#
# CUDA and StaticArrays are weak dependencies, so install them into the default
# environment, not the project one — adding them with --project=. would put them
# in [deps] as well as [weakdeps], which Pkg rejects.
#
#   julia -e 'using Pkg; Pkg.add(["CUDA", "StaticArrays"])'
#   julia --project=. scripts/benchmark_gpu.jl

using BreakingBallLBM
using CUDA
using StaticArrays
using Printf

const BBL = BreakingBallLBM

"""
Free device memory in bytes.

CUDA.jl has spelled this differently across versions, and the benchmark should
not fall over because of a name, so try what exists and fall back to the total.
"""
function free_memory()
    for name in (:available_memory, :free_memory)
        isdefined(CUDA, name) && return Int(getproperty(CUDA, name)())
    end
    return Int(CUDA.totalmem(CUDA.device()))
end

"""A Taylor-Green state in cube-slot order, on the host."""
function host_state(::Type{T}, n::Int, τ::Real; nz::Int = n) where {T}
    tg = TaylorGreen(u0 = 0.02, n = n, ν = viscosity_from_tau(τ))
    s = LBMState{T}(n, n, nz, τ; lattice = D3Q27())
    BBL.init!(s, tg)
    return to_cube_order!(similar(s.f), s.f), tg
end

"""Do the device kernels reproduce the CPU reference?"""
function check_against_cpu(::Type{T}; n = 16, steps = 10, operator = :central_moment) where {T}
    g_cpu, _ = host_state(T, n, 0.8; nz = 8)
    g_gpu = CuArray(copy(g_cpu))

    force = (T(2e-5), T(0), T(0))
    aa_run!(g_cpu, steps, T(0.8); force = force, operator = operator)
    gpu_run!(g_gpu, steps, T(0.8); force = force, operator = operator)

    got = Array(g_gpu)
    err = maximum(abs.(got .- g_cpu)) / maximum(abs.(g_cpu))
    tol = T === Float32 ? 1e-4 : 1e-11
    ok = err < tol
    @printf("  %-8s %-16s max relative difference vs CPU: %.3e  %s\n",
            T, operator, err, ok ? "OK" : "FAILED")
    return ok
end

"""Does the device reproduce the analytic Taylor-Green decay rate?"""
function check_physics(::Type{T}; n = 32, steps = 200, operator = :central_moment) where {T}
    τ = 0.6
    ν = viscosity_from_tau(τ)
    g, tg = host_state(T, n, τ; nz = 4)
    d_g = CuArray(g)

    energy(h) = begin
        s = LBMState{T}(n, n, 4, τ; lattice = D3Q27())
        from_cube_order!(s.f, h)
        total_kinetic_energy(s)
    end

    e0 = energy(g)
    gpu_run!(d_g, steps, τ; operator = operator)
    e1 = energy(Array(d_g))

    ν_measured = -log(e1 / e0) / (4 * tg.k^2 * steps)
    ok = abs(ν_measured / ν - 1) < 0.05
    @printf("  %-8s %-16s viscosity from decay: %.5e vs %.5e (%.2f%%)  %s\n",
            T, operator, ν_measured, ν, 100 * (ν_measured / ν - 1), ok ? "OK" : "FAILED")
    return ok
end

"""Does the wall path reproduce the CPU reference, forces included?"""
function check_walls(::Type{T}; n = 20, radius = 5.0, steps = 6,
                     operator = :central_moment, rule = :interpolated_local) where {T}
    dims = (n, n, n)
    τ = T(0.8)
    ϕ = sphere_sdf_field(T, dims, radius)
    wall = build_wall_field(ϕ; sdf_fn = sphere_sdf_fn(dims, radius))

    s = LBMState{T}(dims..., τ; lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0 + 0.002sin(i + 2j), 0.01cos(i), -0.008sin(j), 0.0))
    g_cpu = to_cube_order!(similar(s.f), s.f)
    g_gpu = CuArray(copy(g_cpu))

    force = (T(2e-5), T(0), T(0))
    spin = (T(0), T(0), T(1.5e-3))

    Fcpu, Tcpu = aa_run_walls!(g_cpu, wall, steps, τ; force = force, spin = spin,
                               operator = operator, rule = rule)

    dwall = gpu_wall(wall)
    contrib = CUDA.zeros(T, 6, length(wall))
    Fgpu, Tgpu = gpu_run_walls!(g_gpu, dwall, contrib, steps, τ; force = force,
                                spin = spin, operator = operator, rule = rule)

    solid = solid_mask(ϕ)
    got = Array(g_gpu)
    ferr = 0.0
    scale = 0.0
    for k in 1:n, j in 1:n, i in 1:n, q in 1:27
        solid[i, j, k] && continue
        ferr = max(ferr, abs(got[i, j, k, q] - g_cpu[i, j, k, q]))
        scale = max(scale, abs(g_cpu[i, j, k, q]))
    end
    rel = ferr / scale
    force_err = maximum(abs.(collect(Fgpu) .- collect(Fcpu))) / maximum(abs.(collect(Fcpu)))
    torque_err = maximum(abs.(collect(Tgpu) .- collect(Tcpu))) / maximum(abs.(collect(Tcpu)))

    tol = T === Float32 ? 1e-3 : 1e-10
    ok = rel < tol && force_err < tol && torque_err < tol
    @printf("  %-8s %-16s populations %.2e, force %.2e, torque %.2e  %s\n",
            T, rule, rel, force_err, torque_err, ok ? "OK" : "FAILED")
    return ok
end

"""
Does the whole coupled loop agree between host and device?

This is the strongest check in the file, because it exercises everything at
once: the wall kernel's accumulating reduction, the box-mean velocity, the
controller, the unit conversions and the 6DOF step. Two implementations that
share only the physics have to produce the same ball.
"""
function check_coupling(::Type{T}; n = 20, radius = 3.5, cycles = 6,
                        substeps = 10) where {T}
    dims = (n, n, n)
    units = LatticeUnits(T; nodes_per_diameter = 7, speed = 39.0,
                         lattice_speed = 0.05, ν = 39.0 * 0.0748 / 40)
    props = BaseballProperties(T)
    ϕ = sphere_sdf_field(T, dims, radius)
    wall = build_wall_field(ϕ; sdf_fn = sphere_sdf_fn(dims, radius))

    s = LBMState{T}(dims..., units.τ; lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
    g_cpu = to_cube_order!(similar(s.f), s.f)
    g_gpu = CuArray(copy(g_cpu))
    dflow = gpu_flow(wall)

    zero3 = (T(0), T(0), T(0))
    ρc, uc = mean_fluid_velocity(g_cpu, wall, zero3)
    ρd, ud = flow_mean_velocity(g_gpu, dflow, zero3)
    mean_err = max(abs(ρd / ρc - 1), maximum(abs.(collect(ud) .- collect(uc))) / 0.05)

    ball = BallState(T; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                     spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
    run = PitchRun(units, props; substeps = substeps, control_time = 20 * substeps)
    st_cpu = PitchState(ball)
    st_gpu = PitchState(ball)
    for _ in 1:cycles
        couple_step!(g_cpu, run, st_cpu, wall)
        couple_step!(g_gpu, run, st_gpu, dflow)
    end

    rel(a, b, scale) = maximum(abs.(collect(a) .- collect(b))) / scale
    force_err = rel(st_gpu.force, st_cpu.force, maximum(abs.(collect(st_cpu.force))))
    vel_err = rel(st_gpu.ball.v, st_cpu.ball.v, 39.0)
    ctl_err = rel(st_gpu.control, st_cpu.control, maximum(abs.(collect(st_cpu.control))))

    # The weighted region mean, on the developed field rather than the uniform
    # one it started from — a uniform field would agree whatever the reduction
    # did. Both sides read the *same* host data, so this is the kernel on its
    # own and not the two runs having drifted apart. The drag sweep divides a
    # coefficient by what this returns, so it answers to the same bar.
    disc = zeros(T, dims)
    c = T(n + 1) / 2
    plane = n - 1
    @inbounds for k in 1:n, j in 1:n
        (T(j) - c)^2 + (T(k) - c)^2 <= T(radius)^2 && (disc[plane, j, k] = one(T))
    end
    _, ur_c = region_mean_velocity(g_cpu, disc, st_cpu.control)
    _, ur_d = region_mean_velocity(CuArray(g_cpu), CuArray(disc), st_cpu.control)
    region_err = rel(ur_d, ur_c, 0.05)

    tol = T === Float32 ? 1e-2 : 1e-9
    ok = mean_err < tol && force_err < tol && vel_err < tol && ctl_err < tol &&
         region_err < tol
    @printf("  %-8s coupled loop: mean %.2e, region %.2e, force %.2e, control %.2e, velocity %.2e  %s\n",
            T, mean_err, region_err, force_err, ctl_err, vel_err, ok ? "OK" : "FAILED")
    return ok
end

"""
Does the device re-cut the seam the way the host does?

The re-cut is the one piece where host and device run genuinely different code —
two kernel launches against a pair of loops — so this compares the whole result:
which nodes came out solid, every wall fraction, how many nodes were refilled,
what they were refilled with, and finally the force a short run produces.

**Wall fractions cannot be expected to match to round-off, and it would be wrong
to ask.** Each δ is the output of a bisection, so it carries the resolution of
that bisection and no more: `2^-iterations`, or 9.5e-7 at the default depth of
twenty. Host and device evaluate `sin`, `cos` and `atan` with different last
bits, and near the root the distance being tested is nearly zero, so the final
comparison can fall either way. In Float64 that costs exactly one bin — the
smallest disagreement the algorithm can produce.

In Float32 it costs several, and that is not a worse result, it is the same
result seen through coarser arithmetic: `eps(Float32)` is 1.2e-7, so a bin at
this depth is eight ulps wide. The interval endpoints, the midpoints and the
distance being tested are all rounded at a size comparable to the bin, so the
two bisections wander a few bins apart on most links. Twenty halvings is already
past what single precision can resolve.

So the bar is not counted in bins at all. δ is measured in lattice spacings, and
what matters is that the wall ends up in the same place: agreement to 1e-4 of a
cell is one part in four thousand of the seam ridge itself at production
resolution. Bins are reported alongside, to tell a noisy bisection (many links,
a few bins each) from a real geometry difference (few links, δ wrong by a lot),
and the force check is the final word on whether any of it matters.
"""
function check_recut(::Type{T}; N = 20, cycles = 6, dims = (44, 44, 44),
                     iterations = 20) where {T}
    geom = BaseballGeometry(; diameter = T(0.0748), seam_height = T(0.00079),
                            seam_amplitude = T(0.7))
    dx = geom.radius * 2 / N
    host = RotatingWall(geom, dims, dx)
    dev = gpu_rotating_flow(host)

    s = LBMState{T}(dims..., T(0.55); lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
    gh = to_cube_order!(similar(s.f), s.f)
    gd = CuArray(copy(gh))
    spin = (T(0), T(0), T(2e-3))

    kind_diff = 0
    worst_δ = 0.0
    far = 0
    fresh_h = 0
    fresh_d = 0
    for m in 1:cycles
        q = quat_from_axis_angle((T(0), T(0), T(1)), T(0.12) * m)
        fresh_h += recut!(host, q, gh, spin; iterations = iterations)
        fresh_d += recut!(dev, q, gd, spin; iterations = iterations)
        kind_diff += count(Array(dev.flow.wall.kind) .!= host.wall.kind)
        d = abs.(Array(dev.flow.wall.deltas) .- host.wall.deltas)
        worst_δ = max(worst_δ, maximum(d))
        far += count(>(2.0^-iterations * 2), d)
    end

    pop_err = maximum(abs.(Array(gd) .- gh))

    # What the wall fractions are for: run the flow against them and compare the
    # force. A δ difference that mattered would show here as something far
    # larger than the precision of the populations themselves.
    Fh, Mh = aa_run_walls!(gh, host.wall, 20, T(0.55); spin = spin, reduction = :mean)
    Fd, Md = gpu_run_walls!(gd, dev.flow.wall, dev.flow.contrib, 20, T(0.55);
                            spin = spin, reduction = :mean)
    fscale = maximum(abs.(collect(Fh)))
    force_err = maximum(abs.(collect(Fd) .- collect(Fh))) / fscale
    torque_err = maximum(abs.(collect(Md) .- collect(Mh))) / max(maximum(abs.(collect(Mh))), eps(T))

    bins = worst_δ * 2.0^iterations
    # The refill is exact arithmetic on an equilibrium, so it is held to
    # round-off. The force is not: a bisection bin of δ propagates through
    # Bouzidi's rule into every link it touches, so the honest bar there is
    # "far smaller than anything that would matter", not "round-off".
    δ_tol = 1e-4                       # lattice spacings of wall position
    pop_tol = T === Float32 ? 1e-4 : 1e-9
    force_tol = T === Float32 ? 1e-2 : 1e-4
    # fresh_h > 0 matters: with no node flips the refill path is not tested at
    # all, and the comparison would pass by doing nothing.
    ok = kind_diff == 0 && worst_δ < δ_tol && fresh_d == fresh_h && fresh_h > 0 &&
         dev.nfluid == host.nfluid && pop_err < pop_tol &&
         force_err < force_tol && torque_err < force_tol
    @printf("  %-8s re-cut: kind %d, delta %.1e cells (%.0f bins, %d links), refilled %d/%d, force %.2e  %s\n",
            T, kind_diff, worst_δ, bins, far, fresh_d, fresh_h, force_err,
            ok ? "OK" : "FAILED")
    return ok
end

"""
Does the device refine the way the host does?

Both levels, the force, and the mean the controller reads. The transfers are the
only place where a value crosses between grids, so a mistake there shows up in
the coarse level first and in the force soon after.
"""
function check_refine(::Type{T}; N = 4, cycles = 12) where {T}
    geom = BaseballGeometry(; diameter = T(0.0748), seam_height = T(0.00079),
                            seam_amplitude = T(0.7))
    dxf = geom.radius / N
    τc = T(0.7)
    cdims = (24, 24, 24)
    lo, hi = (5, 5, 5), (20, 20, 20)

    rg = TwoGrid(T, cdims, lo, hi, τc)
    wall = RotatingWall(geom, size(rg.fine)[1:3], dxf)
    rf = RefinedFlow(rg, wall)
    init_refined_flow!(rf, (x, y, z) -> (1.0, -0.05, 0.0, 0.0))

    drf = gpu_refined_flow(rf)
    dg = drf.grid
    spin = (T(0), T(0), T(5e-3))
    force = (T(1e-6), T(0), T(0))

    Fh = (T(0), T(0), T(0)); Fd = Fh
    Mh = Fh; Md = Fh
    for _ in 1:cycles
        Fh, Mh = advance_flow!(rg, rf, 2, τc; force = force, spin = spin, operator = :bgk)
        Fd, Md = advance_flow!(dg, drf, 2, τc; force = force, spin = spin, operator = :bgk)
    end

    scale = maximum(abs.(rg.coarse))
    coarse_err = maximum(abs.(Array(dg.coarse) .- rg.coarse)) / scale
    fine_err = maximum(abs.(Array(dg.fine) .- rg.fine)) / maximum(abs.(rg.fine))
    force_err = maximum(abs.(collect(Fd) .- collect(Fh))) / maximum(abs.(collect(Fh)))
    torque_err = maximum(abs.(collect(Md) .- collect(Mh))) / maximum(abs.(collect(Mh)))

    ρh, ūh = flow_mean_velocity(rg, rf, force)
    ρd, ūd = flow_mean_velocity(dg, drf, force)
    mean_err = maximum(abs.(collect(ūd) .- collect(ūh))) / 0.05
    count_ok = flow_fluid_count(drf) == flow_fluid_count(rf)

    tol = T === Float32 ? 1e-3 : 1e-10
    ok = coarse_err < tol && fine_err < tol && force_err < tol && torque_err < tol &&
         mean_err < tol && count_ok
    @printf("  %-8s refine: coarse %.2e, fine %.2e, force %.2e, mean %.2e, nfluid %s  %s\n",
            T, coarse_err, fine_err, force_err, mean_err, count_ok ? "ok" : "DIFFER",
            ok ? "OK" : "FAILED")
    return ok
end

"""Milliseconds per re-cut on the device, at production resolution."""
function recut_ms(::Type{T}; N = 40, repeats = 20) where {T}
    geom = BaseballGeometry(; diameter = T(0.0748), seam_height = T(0.00079),
                            seam_amplitude = T(0.7))
    units = LatticeUnits(T; nodes_per_diameter = N, speed = 39.0)
    edge = 2 * (N + 8)
    host = RotatingWall(geom, (edge, edge, edge), units.dx)
    dev = gpu_rotating_flow(host)
    recut!(dev, quat_from_axis_angle((T(0), T(0), T(1)), T(0.01)))
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for m in 1:repeats
            recut!(dev, quat_from_axis_angle((T(0), T(0), T(1)), T(0.01) * m))
        end
    end
    return 1000 * t / repeats, length(host.shell), count(>(Int32(0)), host.wall.kind)
end

"""Node updates per second, in millions, for the wall-bounded kernel."""
function wall_mlups(::Type{T}, n::Int; radius = nothing, threads = 128,
                    target_seconds = 2.0) where {T}
    r = radius === nothing ? n / 8 : radius
    dims = (n, n, n)
    ϕ = sphere_sdf_field(T, dims, r)
    wall = build_wall_field(ϕ)
    s = LBMState{T}(dims..., 0.8; lattice = D3Q27())
    init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
    d_g = CuArray(to_cube_order!(similar(s.f), s.f))
    dwall = gpu_wall(wall)
    contrib = CUDA.zeros(T, 6, max(length(wall), 1))

    gpu_run_walls!(d_g, dwall, contrib, 2, 0.8; threads = threads)
    CUDA.synchronize()
    steps = 20
    while true
        t = CUDA.@elapsed gpu_run_walls!(d_g, dwall, contrib, steps, 0.8; threads = threads)
        t > target_seconds / 4 && return n^3 * steps / t / 1e6
        steps *= 4
        steps > 100_000 && return n^3 * steps / t / 1e6
    end
end

"""Node updates per second, in millions."""
function mlups(::Type{T}, n::Int, operator::Symbol; target_seconds = 2.0,
               threads = 128) where {T}
    g, _ = host_state(T, n, 0.8)
    d_g = CuArray(g)

    gpu_run!(d_g, 2, 0.8; operator = operator, threads = threads)   # warm up / compile
    CUDA.synchronize()

    steps = 20
    while true
        t = CUDA.@elapsed gpu_run!(d_g, steps, 0.8; operator = operator, threads = threads)
        t > target_seconds / 4 && return n^3 * steps / t / 1e6, steps, t
        steps *= 4
        steps > 100_000 && return n^3 * steps / t / 1e6, steps, t
    end
end

function main()
    if !gpu_backend_loaded()
        error("the CUDA extension did not load — check that CUDA.jl and StaticArrays are installed")
    end
    dev = CUDA.device()
    total = CUDA.totalmem(dev) / 2^30
    @printf("Device: %s\n", CUDA.name(dev))
    @printf("Memory: %.1f GiB total, %.1f GiB free\n", total, free_memory() / 2^30)
    println()

    println("Correctness (kernels must match the CPU reference before timings mean anything)")
    ok = true
    for T in (Float64, Float32), operator in (:bgk, :central_moment)
        ok &= check_against_cpu(T; operator = operator)
    end
    for T in (Float64, Float32)
        ok &= check_physics(T; operator = :central_moment)
    end
    for T in (Float64, Float32), rule in (:halfway, :interpolated_local)
        ok &= check_walls(T; rule = rule)
    end
    for T in (Float64, Float32)
        ok &= check_coupling(T)
    end
    for T in (Float64, Float32)
        ok &= check_recut(T)
    end
    for T in (Float64, Float32)
        ok &= check_refine(T)
    end
    ok || error("correctness checks failed — do not trust the timings below")
    println()

    # Measure the ceiling rather than trusting a datasheet number: the 3060 Ti
    # ships with both GDDR6 (448 GB/s) and GDDR6X (608 GB/s) memory.
    bw = gpu_copy_bandwidth(Float32)
    @printf("Measured copy bandwidth: %.0f GB/s\n\n", bw / 1e9)

    println("Throughput")
    @printf("%-9s %-16s %-6s %-12s %-10s %-12s\n",
            "precision", "operator", "n", "MLUPS", "roofline", "of roofline")
    # 27 loads + 27 stores per node update.
    for T in (Float32, Float64), operator in (:bgk, :central_moment), n in (64, 128, 192)
        bytes = 54 * sizeof(T)
        roof = bw / bytes / 1e6
        try
            m, steps, t = mlups(T, n, operator)
            @printf("%-9s %-16s %-6d %-12.1f %-10.0f %-12.1f%%\n",
                    T, operator, n, m, roof, 100 * m / roof)
        catch err
            @printf("%-9s %-16s %-6d skipped (%s)\n", T, operator, n,
                    first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()

    println("Wall-bounded kernel (sphere of radius n/8)")
    @printf("%-9s %-6s %-12s\n", "precision", "n", "MLUPS")
    for T in (Float32, Float64), n in (64, 128, 192)
        try
            @printf("%-9s %-6d %-12.1f\n", T, n, wall_mlups(T, n))
        catch err
            @printf("%-9s %-6d skipped (%s)\n", T, n,
                    first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()

    # Re-cutting the seam. On the CPU this costs several times the whole flow
    # solve (scripts/recut_cost.jl), which is why it is here at all.
    println("Seam re-cut (production geometry, one cut)")
    @printf("%-9s %-6s %-9s %-10s %-11s %-12s\n",
            "precision", "N/D", "shell", "boundary", "ms/cut", "h per pitch")
    for T in (Float32, Float64), n in (20, 40, 60)
        try
            ms, shell, boundary = recut_ms(T; N = n)
            units = LatticeUnits(T; nodes_per_diameter = n, speed = 39.0)
            cuts = (0.445 / units.dt) / 18          # a cut every ~18 steps (§4.1.1)
            @printf("%-9s %-6d %-9d %-10d %-11.2f %-12.3f\n",
                    T, n, shell, boundary, ms, cuts * ms / 1000 / 3600)
        catch err
            @printf("%-9s %-6d skipped (%s)\n", T, n,
                    first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()

    # Block size: worth a look, since the kernel keeps 27 values per thread and
    # occupancy is the first thing that costs.
    println("Block size (Float32, central moment, n = 128)")
    @printf("%-9s %-12s\n", "threads", "MLUPS")
    for threads in (64, 128, 256, 512)
        try
            m, _, _ = mlups(Float32, 128, :central_moment; threads = threads)
            @printf("%-9d %-12.1f\n", threads, m)
        catch err
            @printf("%-9d skipped (%s)\n", threads, first(split(sprint(showerror, err), '\n')))
        end
        flush(stdout)
    end
    println()

    # What the production run can afford.
    for T in (Float32, Float64)
        per_node = 27 * sizeof(T)
        nodes = 0.85 * free_memory() / per_node
        edge = floor(Int, cbrt(nodes))
        @printf("%-8s %d B/node → about %.0f M nodes, i.e. a cube of %d³\n",
                T, per_node, nodes / 1e6, edge)
    end
end

main()
