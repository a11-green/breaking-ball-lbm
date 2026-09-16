"""
CUDA backend.

The kernel does no physics of its own: it maps a thread to a node and calls
`aa_step_node!`, the same function the CPU driver calls and the same one the
analytic benchmarks exercise. Only the launch geometry and the per-thread buffer
type are new here, which keeps the untested surface as small as a GPU-less
development machine allows.
"""
module BreakingBallLBMCUDAExt

using BreakingBallLBM
using CUDA
using StaticArrays

const BBL = BreakingBallLBM

function aa_kernel!(g, even::Bool, nx::Int, ny::Int, nz::Int, τ::T,
                    force::NTuple{3,T}, op::Val, ωb::T, ωh::T) where {T}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        t = Int(idx) - 1
        i = t % nx + 1
        j = (t ÷ nx) % ny + 1
        k = t ÷ (nx * ny) + 1
        buf = MVector{27,T}(undef)
        BBL.aa_step_node!(g, buf, even, i, j, k, nx, ny, nz, τ, force, op, ωb, ωh)
    end
    return nothing
end

function BreakingBallLBM.gpu_run!(g::CuArray{T,4}, nsteps::Integer, τ::Real;
                                  force::NTuple{3,<:Real} = (0, 0, 0),
                                  operator::Symbol = :central_moment,
                                  omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                                  threads::Int = 128) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    size(g, 4) == 27 || throw(ArgumentError("expected 27 cube slots, got $(size(g, 4))"))

    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    n = nx * ny * nz
    blocks = cld(n, threads)
    F = T.(force)
    op = Val(operator)
    τT, ωbT, ωhT = T(τ), T(omega_bulk), T(omega_higher)

    for step in 1:nsteps
        even = isodd(step)
        @cuda threads = threads blocks = blocks aa_kernel!(g, even, nx, ny, nz, τT,
                                                           F, op, ωbT, ωhT)
    end
    return g
end

"""
Plain copy, for measuring what bandwidth this card actually delivers.

A grid-stride loop rather than one element per thread: each thread then has
several loads in flight, which is what it takes to saturate the memory system.
One element per thread measured about 5% under what the LBM kernel itself
sustains, i.e. it was reporting a ceiling below the floor.
"""
function copy_kernel!(dst, src)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds while i <= length(dst)
        dst[i] = src[i]
        i += stride
    end
    return nothing
end

function BreakingBallLBM.gpu_copy_bandwidth(::Type{T} = Float32; n = 64_000_000,
                                            repeats = 20) where {T}
    src = CUDA.rand(T, n)      # not zeros: those can compress in flight
    dst = CUDA.zeros(T, n)
    threads = 256
    blocks = min(cld(n, threads), 8192)
    @cuda threads = threads blocks = blocks copy_kernel!(dst, src)   # warm up
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:repeats
            @cuda threads = threads blocks = blocks copy_kernel!(dst, src)
        end
    end
    CUDA.unsafe_free!(src)
    CUDA.unsafe_free!(dst)
    return 2 * sizeof(T) * n * repeats / t     # bytes read + written, per second
end

end # module
