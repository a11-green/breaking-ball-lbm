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
    @printf("Memory: %.1f GiB total, %.1f GiB free\n", total, CUDA.available_memory() / 2^30)
    println()

    println("Correctness (kernels must match the CPU reference before timings mean anything)")
    ok = true
    for T in (Float64, Float32), operator in (:bgk, :central_moment)
        ok &= check_against_cpu(T; operator = operator)
    end
    for T in (Float64, Float32)
        ok &= check_physics(T; operator = :central_moment)
    end
    ok || error("correctness checks failed — do not trust the timings below")
    println()

    println("Throughput")
    @printf("%-9s %-16s %-6s %-12s %-10s %-12s\n",
            "precision", "operator", "n", "MLUPS", "roofline", "of roofline")
    # 27 loads + 27 stores per node update; the card's own bandwidth sets the ceiling.
    bw = 448e9    # RTX 3060 Ti, GB/s
    for T in (Float32, Float64), operator in (:bgk, :central_moment), n in (64, 128, 192)
        bytes = 54 * sizeof(T)
        roof = bw / bytes / 1e6
        try
            m, steps, t = mlups(T, n, operator)
            @printf("%-9s %-16s %-6d %-12.1f %-10.0f %-12.1f%%\n",
                    T, operator, n, m, roof, 100 * m / roof)
        catch err
            @printf("%-9s %-16s %-6d skipped (%s)\n", T, operator, n,
                    err isa OutOfGPUMemoryError ? "out of memory" : sprint(showerror, err))
        end
        flush(stdout)
    end
    println()

    # What the production run can afford.
    for T in (Float32, Float64)
        per_node = 27 * sizeof(T)
        nodes = 0.85 * CUDA.available_memory() / per_node
        edge = floor(Int, cbrt(nodes))
        @printf("%-8s %d B/node → about %.0f M nodes, i.e. a cube of %d³\n",
                T, per_node, nodes / 1e6, edge)
    end
end

main()
