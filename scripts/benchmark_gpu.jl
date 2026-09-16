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
