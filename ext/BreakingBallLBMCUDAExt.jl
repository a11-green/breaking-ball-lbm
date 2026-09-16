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


"""
Wall-bounded step.

Same shape as `aa_kernel!` — one thread per node, calling the function the CPU
driver and the tests already exercise — with one addition: a boundary node
writes the force and torque it hands the body into its own column of `contrib`.

The reduction needs no atomics because the geometry already numbers the boundary
nodes. `kind[i, j, k]` is that number, so column `b` belongs to exactly one
thread, in this step and in every other, and the totals are a plain sum
afterwards.

`Val{ACC}` picks what a step does to its column. `false` overwrites it, so the
call reports the last step and needs no clearing pass; `true` adds to it, which
is what the coupled loop wants — the momentum-exchange force on a body in
turbulent flow fluctuates by more than the mean it is fluctuating about, so a
single step is a poor sample to feed a trajectory. Accumulating is still
race-free for the same reason overwriting is: one thread owns the column.
"""
function aa_wall_kernel!(g, contrib, even::Bool, nx::Int, ny::Int, nz::Int, τ::T,
                         force::NTuple{3,T}, op::Val, ωb::T, ωh::T,
                         kind, deltas, center::NTuple{3,T}, spin::NTuple{3,T},
                         halfway::Bool, ::Val{ACC}) where {T,ACC}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        t = Int(idx) - 1
        i = t % nx + 1
        j = (t ÷ nx) % ny + 1
        k = t ÷ (nx * ny) + 1
        buf = MVector{27,T}(undef)
        F, M = BBL.aa_step_node_walls!(g, buf, even, i, j, k, nx, ny, nz, τ, force,
                                       op, ωb, ωh, kind, deltas, center, spin, halfway)
        @inbounds b = kind[i, j, k]
        if b > Int32(0)
            if ACC
                @inbounds contrib[1, b] += F[1]
                @inbounds contrib[2, b] += F[2]
                @inbounds contrib[3, b] += F[3]
                @inbounds contrib[4, b] += M[1]
                @inbounds contrib[5, b] += M[2]
                @inbounds contrib[6, b] += M[3]
            else
                @inbounds contrib[1, b] = F[1]
                @inbounds contrib[2, b] = F[2]
                @inbounds contrib[3, b] = F[3]
                @inbounds contrib[4, b] = M[1]
                @inbounds contrib[5, b] = M[2]
                @inbounds contrib[6, b] = M[3]
            end
        end
    end
    return nothing
end

"""Copy the wall geometry to the device; the centre stays a plain tuple."""
function BreakingBallLBM.gpu_wall(wall::BBL.WallField)
    return BBL.WallField(CuArray(wall.kind), CuArray(wall.deltas), wall.center)
end

function BreakingBallLBM.gpu_run_walls!(g::CuArray{T,4}, wall::BBL.WallField,
                                        contrib::CuArray{T,2}, nsteps::Integer, τ::Real;
                                        force::NTuple{3,<:Real} = (0, 0, 0),
                                        spin::NTuple{3,<:Real} = (0, 0, 0),
                                        operator::Symbol = :central_moment,
                                        rule::Symbol = :interpolated_local,
                                        reduction::Symbol = :last,
                                        omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                                        threads::Int = 128) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    size(g, 4) == 27 || throw(ArgumentError("expected 27 cube slots, got $(size(g, 4))"))
    rule in (:interpolated_local, :halfway) ||
        throw(ArgumentError("rule must be :interpolated_local or :halfway, got $rule"))
    reduction in (:last, :mean) ||
        throw(ArgumentError("reduction must be :last or :mean, got $reduction"))
    size(contrib, 1) == 6 && size(contrib, 2) >= length(wall) ||
        throw(ArgumentError("contrib must be (6, $(length(wall))), got $(size(contrib))"))

    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    blocks = cld(nx * ny * nz, threads)
    F = T.(force)
    ω = T.(spin)
    op = Val(operator)
    halfway = rule === :halfway
    τT, ωbT, ωhT = T(τ), T(omega_bulk), T(omega_higher)
    center = T.(wall.center)
    acc = Val(reduction === :mean)
    reduction === :mean && fill!(contrib, zero(T))

    for step in 1:nsteps
        even = isodd(step)
        @cuda threads = threads blocks = blocks aa_wall_kernel!(
            g, contrib, even, nx, ny, nz, τT, F, op, ωbT, ωhT,
            wall.kind, wall.deltas, center, ω, halfway, acc)
    end

    total = Array(sum(contrib; dims = 2))
    reduction === :mean && (total ./= nsteps)
    return (total[1], total[2], total[3]), (total[4], total[5], total[6])
end

"""
A coupled run's device-side state: the wall geometry, the scratch its force
reduction writes into, and the fluid mask the box-mean velocity needs.

The mask is stored rather than derived per call because the mean is taken every
sub-cycle and `kind` is `Int32`, so comparing it on the fly would mean either a
type conversion inside the reduction or a second kernel.
"""
struct DeviceFlow{T<:AbstractFloat,W,C,M}
    wall::W
    contrib::C
    fmask::M
    nfluid::Int
end

"""
    gpu_flow(wall)

Move a [`WallField`](@ref) to the device along with everything the coupled loop
needs alongside it.
"""
function BreakingBallLBM.gpu_flow(wall::BBL.WallField{T}) where {T}
    dwall = BreakingBallLBM.gpu_wall(wall)
    nb = max(length(wall), 1)
    contrib = CUDA.zeros(T, 6, nb)
    mask = T.(wall.kind .!= BBL.SOLID_NODE)
    nfluid = count(!=(BBL.SOLID_NODE), wall.kind)
    dmask = CuArray(mask)
    return DeviceFlow{T,typeof(dwall),typeof(contrib),typeof(dmask)}(
        dwall, contrib, dmask, nfluid)
end

BreakingBallLBM.flow_fluid_count(f::DeviceFlow) = f.nfluid

function BreakingBallLBM.advance_flow!(g::CuArray{T,4}, f::DeviceFlow{T},
                                       nsteps::Integer, τ::Real; kwargs...) where {T}
    return BreakingBallLBM.gpu_run_walls!(g, f.wall, f.contrib, nsteps, τ;
                                          reduction = :mean, kwargs...)
end

"""
Mass-averaged density and velocity over the fluid nodes, on the device.

Twenty-seven masked reductions, one per cube slot, because the momentum is a
fixed linear combination of the slot sums: `ρu_α = Σ_s c_α(s) Σ_x g[x, s]`. Each
reduction covers a twenty-seventh of the array, so the whole thing is a single
pass over the populations — a fraction of a percent of a sub-cycle's work —
without a hand-written block reduction to get wrong.

The accumulator is `Float64` even for a Float32 solver, for the reason spelled
out on `mean_fluid_velocity`: the controller acts on a difference that is a
fraction of a percent of these sums, and single-precision rounding over tens of
millions of nodes is of that same order.

Valid only in the even-step layout, which is where `gpu_run_walls!` always
leaves the state, since it insists on an even step count. The `force/2` term is
Guo's definition of momentum, matching `mean_fluid_velocity` on the host.
"""
function BreakingBallLBM.flow_mean_velocity(g::CuArray{T,4}, f::DeviceFlow{T},
                                            force::NTuple{3,<:Real}) where {T}
    n = size(g, 1) * size(g, 2) * size(g, 3)
    gr = reshape(g, n, 27)
    mask = reshape(f.fmask, n)

    ρ = 0.0; mx = 0.0; my = 0.0; mz = 0.0
    for s in 1:27
        cx, cy, cz = BBL.cube_velocity(s)
        total = mapreduce((a, b) -> Float64(a) * Float64(b), +,
                          view(gr, :, s), mask; init = 0.0)
        ρ += total
        cx != 0 && (mx += cx * total)
        cy != 0 && (my += cy * total)
        cz != 0 && (mz += cz * total)
    end
    ρ == 0 && return zero(T), (zero(T), zero(T), zero(T))
    half = 0.5 * f.nfluid
    return T(ρ / f.nfluid), (T((mx + half * Float64(force[1])) / ρ),
                             T((my + half * Float64(force[2])) / ρ),
                             T((mz + half * Float64(force[3])) / ρ))
end

end # module
