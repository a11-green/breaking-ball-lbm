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
                    force::NTuple{3,T}, op::Val, ωb::T, ωh::T, smag::T) where {T}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        t = Int(idx) - 1
        i = t % nx + 1
        j = (t ÷ nx) % ny + 1
        k = t ÷ (nx * ny) + 1
        buf = MVector{27,T}(undef)
        BBL.aa_step_node!(g, buf, even, i, j, k, nx, ny, nz, τ, force, op, ωb, ωh, smag)
    end
    return nothing
end

function BreakingBallLBM.gpu_run!(g::CuArray{T,4}, nsteps::Integer, τ::Real;
                                  force::NTuple{3,<:Real} = (0, 0, 0),
                                  operator::Symbol = :central_moment,
                                  smagorinsky::Real = 0.0,
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
    smagT = T(smagorinsky)

    for step in 1:nsteps
        even = isodd(step)
        @cuda threads = threads blocks = blocks aa_kernel!(g, even, nx, ny, nz, τT,
                                                           F, op, ωbT, ωhT, smagT)
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
                         halfway::Bool, smag::T, ::Val{ACC}) where {T,ACC}
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        t = Int(idx) - 1
        i = t % nx + 1
        j = (t ÷ nx) % ny + 1
        k = t ÷ (nx * ny) + 1
        buf = MVector{27,T}(undef)
        F, M = BBL.aa_step_node_walls!(g, buf, even, i, j, k, nx, ny, nz, τ, force,
                                       op, ωb, ωh, kind, deltas, center, spin, halfway,
                                       smag)
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
                                        smagorinsky::Real = 0.0,
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
            wall.kind, wall.deltas, center, ω, halfway, T(smagorinsky), acc)
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
mutable struct DeviceFlow{T<:AbstractFloat,W,C,M}
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

"""
Device state for geometry that turns with the ball.

Wraps the static flow handle and adds what a re-cut needs: the permanent shell
numbering, scratch for the signed distance and the solid flags, and the shape
itself — which travels as `BallShape` rather than `BaseballGeometry` because the
latter carries a sampled polyline and a `Vector` cannot go to a kernel.
"""
mutable struct DeviceRotatingFlow{T<:AbstractFloat,F,S,V,P,B}
    flow::F
    slot::S
    shell::V
    phi::P
    solid::B
    was_solid::B
    shape::BBL.BallShape{T}
    dx::T
    center::NTuple{3,T}
    radius_nodes::T
    orientation::BBL.Quat{T}
    fixed_solid::Int
    nfluid::Int
    recuts::Int
end

"""
    gpu_rotating_flow(rw)

Move a [`RotatingWall`](@ref) and everything the coupled loop needs onto the
device. The host object stays valid and unaltered, which is what lets the two
be re-cut side by side and compared.
"""
function BreakingBallLBM.gpu_rotating_flow(rw::BBL.RotatingWall{T}) where {T}
    flow = BreakingBallLBM.gpu_flow(rw.wall)
    return DeviceRotatingFlow{T,typeof(flow),typeof(CuArray(rw.slot)),
                              typeof(CuArray(rw.shell)),typeof(CuArray(rw.phi)),
                              typeof(CuArray(rw.solid))}(
        flow, CuArray(rw.slot), CuArray(rw.shell), CuArray(rw.phi),
        CuArray(rw.solid), CuArray(rw.was_solid),
        BBL.BallShape(rw.geom), rw.dx, rw.center, rw.radius_nodes,
        rw.orientation, rw.fixed_solid, rw.nfluid, rw.recuts)
end

function recut_phi_kernel!(phi, solid, shell, shape::BBL.BallShape{T},
                           back::BBL.Quat{T}, center::NTuple{3,T}, dx::T) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= length(shell)
        @inbounds ijk = shell[n]
        φ = BBL.recut_distance(shape, back, center, dx, ijk[1], ijk[2], ijk[3])
        @inbounds phi[n] = φ
        @inbounds solid[n] = φ < 0
    end
    return nothing
end

function recut_column_kernel!(kind, deltas, slot, solid, phi, fmask, shell,
                              shape::BBL.BallShape{T}, back::BBL.Quat{T},
                              center::NTuple{3,T}, dx::T, nx::Int, ny::Int, nz::Int,
                              iterations::Int) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= length(shell)
        @inbounds ijk = shell[n]
        i, j, k = Int(ijk[1]), Int(ijk[2]), Int(ijk[3])
        is_solid = BBL.recut_column!(kind, deltas, slot, solid, phi, Int(n), i, j, k,
                                     shape, back, center, dx, nx, ny, nz, iterations)
        # The box-mean reduction reads this mask, so it has to move with the cut.
        @inbounds fmask[i, j, k] = is_solid ? zero(T) : one(T)
    end
    return nothing
end

function refill_kernel!(g, kind, slot, solid, was_solid, shell,
                        center::NTuple{3,T}, spin::NTuple{3,T},
                        nx::Int, ny::Int, nz::Int) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= length(shell)
        @inbounds fresh = was_solid[n] && !solid[n]
        if fresh
            @inbounds ijk = shell[n]
            BBL.refill_node!(g, kind, slot, solid, was_solid,
                             Int(ijk[1]), Int(ijk[2]), Int(ijk[3]),
                             center, spin, nx, ny, nz)
        end
    end
    return nothing
end

"""
    recut!(drw, q; iterations, threads)

Re-cut the geometry on the device. Two launches, because a node's wall fractions
depend on whether its neighbours came out solid, and that has to be settled
across the whole shell before any of it is used.
"""
function BreakingBallLBM.recut!(drw::DeviceRotatingFlow{T}, q::BBL.Quat{T};
                                iterations::Integer = 20, threads::Int = 128) where {T}
    back = conj(q)
    drw.was_solid, drw.solid = drw.solid, drw.was_solid
    n = length(drw.shell)
    n == 0 && return drw
    blocks = cld(n, threads)
    wall = drw.flow.wall
    nx, ny, nz = size(drw.slot)

    @cuda threads = threads blocks = blocks recut_phi_kernel!(
        drw.phi, drw.solid, drw.shell, drw.shape, back, drw.center, drw.dx)
    @cuda threads = threads blocks = blocks recut_column_kernel!(
        wall.kind, wall.deltas, drw.slot, drw.solid, drw.phi, drw.flow.fmask,
        drw.shell, drw.shape, back, drw.center, drw.dx, nx, ny, nz, Int(iterations))

    drw.orientation = q
    drw.recuts += 1
    drw.nfluid = prod(size(drw.slot)) - drw.fixed_solid - Int(sum(drw.solid))
    return drw
end

function BreakingBallLBM.recut!(drw::DeviceRotatingFlow{T}, q::BBL.Quat{T},
                                g::CuArray{T,4}, spin::NTuple{3,<:Real};
                                iterations::Integer = 20, threads::Int = 128) where {T}
    BreakingBallLBM.recut!(drw, q; iterations = iterations, threads = threads)
    n = length(drw.shell)
    n == 0 && return 0
    nx, ny, nz = size(drw.slot)
    wall = drw.flow.wall
    @cuda threads = threads blocks = cld(n, threads) refill_kernel!(
        g, wall.kind, drw.slot, drw.solid, drw.was_solid, drw.shell,
        drw.center, T.(spin), nx, ny, nz)
    return Int(mapreduce((a, b) -> (a && !b) ? 1 : 0, +, drw.was_solid, drw.solid;
                         init = 0))
end

BreakingBallLBM.flow_fluid_count(drw::DeviceRotatingFlow) = drw.nfluid

BreakingBallLBM.advance_flow!(g::CuArray{T,4}, drw::DeviceRotatingFlow{T},
                              nsteps::Integer, τ::Real; kwargs...) where {T} =
    BreakingBallLBM.advance_flow!(g, drw.flow, nsteps, τ; kwargs...)

BreakingBallLBM.flow_mean_velocity(g::CuArray{T,4}, drw::DeviceRotatingFlow{T},
                                   force::NTuple{3,<:Real}) where {T} =
    BreakingBallLBM.flow_mean_velocity(g, drw.flow, force)

function BreakingBallLBM.maybe_recut!(g::CuArray{T,4}, drw::DeviceRotatingFlow{T},
                                      q::BBL.Quat{T}, spin::NTuple{3,<:Real},
                                      drift::Real) where {T}
    BBL.orientation_drift(drw.orientation, q, drw.radius_nodes) < drift && return 0
    return BreakingBallLBM.recut!(drw, q, g, spin)
end

# --- two-level refinement --------------------------------------------------
#
# The transfers are written against a (parent, child) pair, so the same kernels
# serve a deeper hierarchy if one is ever nested — only the host-side recursion
# would change.

"""Device copy of a [`TwoGrid`](@ref), with the interface node list precomputed."""
mutable struct DeviceTwoGrid{T<:AbstractFloat,A,P,V}
    coarse::A
    fine::A
    prev::P
    τc::T
    τf::T
    lo::NTuple{3,Int}
    hi::NTuple{3,Int}
    ratio::Int
    α::T
    edge::V
    layers::Int
end

"""
    gpu_two_grid(rg; layers = 3)

Move a [`TwoGrid`](@ref) to the device. The interface node list is built once on
the host: the patch never moves, and launching a thread per fine node would have
fifteen sixteenths of them exit immediately at production size.
"""
function BreakingBallLBM.gpu_two_grid(rg::BBL.TwoGrid{T}; layers::Integer = 3) where {T}
    edge = CuArray(BBL.interface_nodes(size(rg.fine)[1:3], layers))
    return DeviceTwoGrid{T,typeof(CuArray(rg.coarse)),typeof(CuArray(rg.prev)),typeof(edge)}(
        CuArray(rg.coarse), CuArray(rg.fine), CuArray(rg.prev),
        rg.τc, rg.τf, rg.lo, rg.hi, rg.ratio, rg.α, edge, Int(layers))
end

function interface_kernel!(fine, coarse, prev, lo::NTuple{3,Int}, α::T, θ::T, edge) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= length(edge)
        @inbounds abc = edge[n]
        neq_a = MVector{27,T}(undef)
        neq_b = MVector{27,T}(undef)
        acc = MVector{27,T}(undef)
        BBL.fill_interface_node!(fine, coarse, prev, lo, α, θ,
                                 Int(abc[1]), Int(abc[2]), Int(abc[3]),
                                 neq_a, neq_b, acc)
    end
    return nothing
end

function restrict_kernel!(coarse, fine, lo::NTuple{3,Int}, ext::NTuple{3,Int},
                          ratio::Int, invα::T) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= ext[1] * ext[2] * ext[3]
        t = Int(n) - 1
        i = lo[1] + 1 + t % ext[1]
        j = lo[2] + 1 + (t ÷ ext[1]) % ext[2]
        k = lo[3] + 1 + t ÷ (ext[1] * ext[2])
        a = ratio * (i - lo[1]) + 1
        b = ratio * (j - lo[2]) + 1
        c = ratio * (k - lo[3]) + 1
        acc = MVector{27,T}(undef)
        BBL.restrict_node!(coarse, fine, i, j, k, a, b, c, invα, acc)
    end
    return nothing
end

"""Snapshot the covered coarse box; a strided copy, so no kernel of its own."""
function BreakingBallLBM.save_coarse!(dg::DeviceTwoGrid)
    dg.prev .= view(dg.coarse, dg.lo[1]:dg.hi[1], dg.lo[2]:dg.hi[2], dg.lo[3]:dg.hi[3], :)
    return dg
end

function BreakingBallLBM.interface_fill!(dg::DeviceTwoGrid{T}, frac::Real;
                                         layers::Integer = dg.layers,
                                         threads::Int = 128, scratch = nothing) where {T}
    layers == dg.layers ||
        throw(ArgumentError("the device node list was built for $(dg.layers) layers, not $layers"))
    n = length(dg.edge)
    n == 0 && return dg
    @cuda threads = threads blocks = cld(n, threads) interface_kernel!(
        dg.fine, dg.coarse, dg.prev, dg.lo, dg.α, T(frac), dg.edge)
    return dg
end

function BreakingBallLBM.restrict!(dg::DeviceTwoGrid{T}; filtered::Bool = false,
                                   threads::Int = 128, scratch = nothing) where {T}
    filtered && throw(ArgumentError("the filtered restriction is host-only; it is off by " *
                                    "default because it measured worse (see restrict!)"))
    ext = ntuple(d -> dg.hi[d] - dg.lo[d] - 1, 3)
    n = prod(ext)
    n <= 0 && return dg
    @cuda threads = threads blocks = cld(n, threads) restrict_kernel!(
        dg.coarse, dg.fine, dg.lo, ext, dg.ratio, one(T) / dg.α)
    return dg
end

"""
Device state for a ball inside the refined patch.

`coarse_mask` is the coarse level's fluid mask — one where the box-mean should
count a node, zero inside the ball — and it has to be rebuilt whenever a re-cut
flips a node, or the controller spreads its momentum over the wrong volume.
"""
mutable struct DeviceRefinedFlow{T<:AbstractFloat,G,W,M}
    grid::G
    wall::W
    coarse_mask::M
    nfluid::Int
    cycles::Int
end

function BreakingBallLBM.gpu_refined_flow(rf::BBL.RefinedFlow{T}; layers::Integer = 3) where {T}
    dg = BreakingBallLBM.gpu_two_grid(rf.grid; layers = layers)
    dw = BreakingBallLBM.gpu_rotating_flow(rf.wall)
    mask = CuArray(T.(.!rf.coarse_solid))
    return DeviceRefinedFlow{T,typeof(dg),typeof(dw),typeof(mask)}(
        dg, dw, mask, rf.nfluid, rf.cycles)
end

function coarse_mask_kernel!(mask, kind, lo::NTuple{3,Int}, ext::NTuple{3,Int},
                             ratio::Int, ::Type{T}) where {T}
    n = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if n <= ext[1] * ext[2] * ext[3]
        t = Int(n) - 1
        i = lo[1] + t % ext[1]
        j = lo[2] + (t ÷ ext[1]) % ext[2]
        k = lo[3] + t ÷ (ext[1] * ext[2])
        a = ratio * (i - lo[1]) + 1
        b = ratio * (j - lo[2]) + 1
        c = ratio * (k - lo[3]) + 1
        @inbounds mask[i, j, k] = kind[a, b, c] == BBL.SOLID_NODE ? zero(T) : one(T)
    end
    return nothing
end

function BreakingBallLBM.refresh_coarse_solid!(drf::DeviceRefinedFlow{T};
                                               threads::Int = 128) where {T}
    dg = drf.grid
    ext = ntuple(d -> dg.hi[d] - dg.lo[d] + 1, 3)
    n = prod(ext)
    @cuda threads = threads blocks = cld(n, threads) coarse_mask_kernel!(
        drf.coarse_mask, drf.wall.flow.wall.kind, dg.lo, ext, dg.ratio, T)
    drf.nfluid = Int(sum(drf.coarse_mask))
    return drf
end

BreakingBallLBM.flow_fluid_count(drf::DeviceRefinedFlow) = drf.nfluid

"""Mass-averaged density and velocity over the coarse level, skipping the ball."""
function BreakingBallLBM.flow_mean_velocity(dg::DeviceTwoGrid{T},
                                            drf::DeviceRefinedFlow{T},
                                            force::NTuple{3,<:Real}) where {T}
    n = size(dg.coarse, 1) * size(dg.coarse, 2) * size(dg.coarse, 3)
    gr = reshape(dg.coarse, n, 27)
    mask = reshape(drf.coarse_mask, n)

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
    half = 0.5 * drf.nfluid
    return T(ρ / drf.nfluid), (T((mx + half * Float64(force[1])) / ρ),
                               T((my + half * Float64(force[2])) / ρ),
                               T((mz + half * Float64(force[3])) / ρ))
end

function BreakingBallLBM.advance_flow!(dg::DeviceTwoGrid{T}, drf::DeviceRefinedFlow{T},
                                       nsteps::Integer, τ::Real;
                                       force::NTuple{3,<:Real} = (0, 0, 0),
                                       spin::NTuple{3,<:Real} = (0, 0, 0),
                                       operator::Symbol = :central_moment,
                                       rule::Symbol = :interpolated_local,
                                       smagorinsky::Real = 0.0,
                                       omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                                       threads::Int = 128) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    m = dg.ratio
    Fc = T.(force)
    Ff = BBL.fine_force(Fc, m)
    ωf = T.(spin) ./ m
    dwall = drf.wall

    sF = (zero(T), zero(T), zero(T))
    sM = (zero(T), zero(T), zero(T))
    cycles = nsteps ÷ 2
    for _ in 1:cycles
        BreakingBallLBM.save_coarse!(dg)
        BreakingBallLBM.gpu_run!(dg.coarse, 2, dg.τc; force = Fc, operator = operator,
                                 smagorinsky = smagorinsky,
                                 omega_bulk = omega_bulk, omega_higher = omega_higher,
                                 threads = threads)
        for half in 1:2
            F, M = BreakingBallLBM.gpu_run_walls!(dg.fine, dwall.flow.wall,
                                                  dwall.flow.contrib, 2, dg.τf;
                                                  force = Ff, spin = ωf,
                                                  operator = operator, rule = rule,
                                                  reduction = :mean,
                                                  smagorinsky = smagorinsky,
                                                  omega_bulk = omega_bulk,
                                                  omega_higher = omega_higher,
                                                  threads = threads)
            sF = sF .+ F
            sM = sM .+ M
            BreakingBallLBM.interface_fill!(dg, half / 2; threads = threads)
        end
        BreakingBallLBM.restrict!(dg; threads = threads)
        drf.cycles += 1
    end
    w = one(T) / (2 * cycles)
    return BBL.coarse_force(sF .* w, m), BBL.coarse_torque(sM .* w, m)
end

function BreakingBallLBM.maybe_recut!(dg::DeviceTwoGrid{T}, drf::DeviceRefinedFlow{T},
                                      q::BBL.Quat{T}, spin::NTuple{3,<:Real},
                                      drift::Real) where {T}
    ωf = T.(spin) ./ dg.ratio
    fresh = BreakingBallLBM.maybe_recut!(dg.fine, drf.wall, q, ωf, drift)
    fresh == 0 && return 0
    BreakingBallLBM.refresh_coarse_solid!(drf)
    return fresh
end

BreakingBallLBM.max_substeps(drf::DeviceRefinedFlow, spin::NTuple{3,<:Real}, drift::Real) =
    max(2, 2 * floor(Int, drift /
        (BBL.surface_drift_per_step(drf.wall, spin ./ drf.grid.ratio) * drf.grid.ratio) / 2))

end # module
