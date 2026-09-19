"""
More than two levels, as a chain of the pairs that already exist.

**Why a chain rather than a new kind of grid.** Every transfer in
`two_grid.jl` — the interface fill, the restriction, the saved coarse box — is
written against a *(parent, child)* pair and touches nothing else. So a deeper
hierarchy is not a new mechanism; it is the same pair, repeated, with each
level's fine array serving as the next pair's coarse one. The arrays are shared
rather than copied, so a step taken on one level is already visible to the pair
below it, and not one line of the transfers changes.

**Why a second level is needed at all.** §6.5: the seam ridge is 0.79 mm and the
reachable uniform spacing is 1.87 mm, so the thing the whole project is about is
sub-cell. One level of refinement brings it to 0.84 cells; two brings it to 1.7,
which is the first configuration where the ridge is something the solver can see
rather than infer.

**The recursion.** Advancing a level by two of its own steps means, when it has
a child, running the child pair's cycle — which advances the child by four. Time
refinement follows the acoustic scaling of §6.5.1, so each level runs at half
its parent's step, and a chain of `n` levels advances the deepest by `2^n` steps
for every two of the base grid's. The interface is filled at the halves, exactly
as the two-level cycle does, because it *is* the two-level cycle.

**The ball lives on the deepest level only.** Every level above solves straight
through it and is overwritten by restriction, which is what the two-level code
already does with the coarse level (§6.5.1) — the cost is a few percent of the
level's nodes and it keeps every level a plain uniform grid.
"""

"""
    GridChain(T, base_dims, patches, τ_base; ratio = 2)

`patches[k]` is the `(lo, hi)` box of level `k`'s patch **in level `k - 1`'s
index space**, the same convention a single [`TwoGrid`](@ref) takes. One patch
gives exactly the two-level grid, and the cycle below then does exactly what
`refine_cycle!` does.
"""
struct GridChain{T<:AbstractFloat}
    levels::Vector{TwoGrid{T}}
end

function GridChain(::Type{T}, base_dims::NTuple{3,<:Integer},
                   patches::AbstractVector, τ_base::Real;
                   ratio::Integer = 2) where {T<:AbstractFloat}
    isempty(patches) && throw(ArgumentError("a chain needs at least one patch"))
    levels = TwoGrid{T}[]
    dims = NTuple{3,Int}(base_dims)
    τ = T(τ_base)
    for (n, (lo, hi)) in enumerate(patches)
        rg = isempty(levels) ? TwoGrid(T, dims, lo, hi, τ; ratio = ratio) :
                               TwoGrid(levels[end].fine, lo, hi, τ; ratio = ratio)
        push!(levels, rg)
        dims = NTuple{3,Int}(size(rg.fine)[1:3])
        τ = rg.τf
    end
    return GridChain{T}(levels)
end

"""The arrays, coarsest first: `n` pairs describe `n + 1` levels."""
levels(ch::GridChain) = vcat([ch.levels[1].coarse], [rg.fine for rg in ch.levels])

"""Relaxation time of each level, coarsest first."""
level_taus(ch::GridChain) = vcat([ch.levels[1].τc], [rg.τf for rg in ch.levels])

"""The base grid — the one the domain boundary belongs to."""
base_grid(ch::GridChain) = ch.levels[1].coarse

"""The deepest grid — the one the ball is cut into."""
finest_grid(ch::GridChain) = ch.levels[end].fine

Base.length(ch::GridChain) = length(ch.levels) + 1      # levels, not pairs

"""
    chain_sizes(ch)

Node counts per level and the work one base-step pair costs, in node updates.
Each level runs `2^k` steps for every two of the base grid's, so the deepest
level dominates long before its node count does.
"""
function chain_sizes(ch::GridChain)
    ns = [prod(size(a)[1:3]) for a in levels(ch)]
    steps = [2 * 2^(k - 1) for k in 1:length(ns)]
    return (nodes = ns, steps = steps, updates = sum(ns .* steps))
end

"""
    chain_cycle_walls!(ch, wall; force, spin, ...)

One base-level cycle — two steps of the coarsest grid — with the ball on the
deepest. Returns the `(force, torque)` averaged over the cycle, **in the
deepest level's lattice units**.

`spin` is the **base** level's, and is scaled down one factor of the ratio per
level, the same way the body force is. (`refine_cycle_walls!` takes its fine
level's spin already scaled; `advance_flow!` is what scales it there. Taking the
base level's is the convention that survives an arbitrary depth.)

`wall = nothing` runs the same cycle with no body in it, which is what a
convergence or fixed-point check wants.

The recursion is the whole implementation: *advance a level by two of its own
steps* is `aa_run!` when the level has no child, and the child pair's cycle when
it does. The pair's cycle is the two-level one, unchanged — it saves the covered
coarse box, advances its coarse level, fills the interface at each half, and
restricts. So a chain of one pair runs exactly what `refine_cycle_walls!` runs,
which is checked rather than asserted (`test/test_multi_grid.jl`).

`channel` reaches the base level only. The domain boundary belongs to the
coarsest grid; every patch is several diameters inside it (§4.4.2.1).
"""
function chain_cycle_walls!(ch::GridChain{T}, wall::Union{RotatingWall{T},Nothing};
                            force::NTuple{3,<:Real} = (0, 0, 0),
                            spin::NTuple{3,<:Real} = (0, 0, 0),
                            operator::Symbol = :central_moment,
                            rule::Symbol = :interpolated_local,
                            smagorinsky::Real = 0.0,
                            layers::Integer = 3, filtered::Bool = false,
                            omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                            channel = nothing, inlet::NTuple{3,<:Real} = (0, 0, 0),
                            scratch = ntuple(_ -> Vector{T}(undef, 27), 3)) where {T}
    npairs = length(ch.levels)
    arrays = levels(ch)
    τs = level_taus(ch)
    m = ch.levels[1].ratio

    F0 = T.(force)
    ω0 = T.(spin)
    sF = Ref((zero(T), zero(T), zero(T)))
    sM = Ref((zero(T), zero(T), zero(T)))
    visits = Ref(0)

    # Body acceleration and spin both go down by one factor of the ratio per
    # level: `a ∝ dt²/dx` and `ω ∝ dt`, and acoustic scaling halves both.
    scale(v, i) = v ./ T(m)^(i - 1)

    function two_steps!(i::Int)
        if i > npairs
            if wall === nothing
                aa_run!(arrays[i], 2, τs[i]; force = scale(F0, i), operator = operator,
                        smagorinsky = smagorinsky,
                        omega_bulk = omega_bulk, omega_higher = omega_higher)
            else
                F, M = aa_run_walls!(arrays[i], wall.wall, 2, τs[i];
                                     force = scale(F0, i), spin = scale(ω0, i),
                                     operator = operator, rule = rule, reduction = :mean,
                                     smagorinsky = smagorinsky,
                                     omega_bulk = omega_bulk, omega_higher = omega_higher)
                sF[] = sF[] .+ F
                sM[] = sM[] .+ M
            end
            visits[] += 1
            return nothing
        end
        rg = ch.levels[i]
        save_coarse!(rg)
        aa_run!(arrays[i], 2, τs[i]; force = scale(F0, i), operator = operator,
                smagorinsky = smagorinsky,
                channel = i == 1 ? channel : nothing, inlet = inlet,
                omega_bulk = omega_bulk, omega_higher = omega_higher)
        for half in 1:2
            two_steps!(i + 1)
            interface_fill!(rg, half / 2; layers = layers, scratch = scratch)
        end
        restrict!(rg; filtered = filtered, scratch = scratch)
        return nothing
    end

    two_steps!(1)
    w = one(T) / visits[]
    return sF[] .* w, sM[] .* w
end

"""
    chain_force(F, ratio, npairs)

A force measured on the deepest level, in the base level's units. Force carries
`ρ L⁴/T²`, so it gains `m²` per level climbed; torque carries `ρ L⁵/T²` and
gains `m³`.
"""
chain_force(F::NTuple{3,T}, ratio::Integer, npairs::Integer) where {T} =
    F ./ T(ratio)^(2 * npairs)
chain_torque(M::NTuple{3,T}, ratio::Integer, npairs::Integer) where {T} =
    M ./ T(ratio)^(3 * npairs)

"""
    level_origin(ch, k)

Where level `k` sits in the **base** grid's index units: the position of its
first node, and the spacing between its nodes. Level 1 is the base grid itself,
so it returns `((1,1,1), 1)`.

The initialiser and anything checking a level against an analytic solution both
need this mapping, and they must not each work it out: a level filled with one
arithmetic and judged by another agrees only where the two happen to.
"""
function level_origin(ch::GridChain{T}, k::Integer) where {T}
    1 <= k <= length(ch) || throw(BoundsError(ch, k))
    offset = (one(T), one(T), one(T))
    h = one(T)
    for n in 1:(k - 1)
        rg = ch.levels[n]
        offset = offset .+ (T.(rg.lo) .- one(T)) .* h
        h /= T(rg.ratio)
    end
    return offset, h
end

"""
    init_chain!(ch, field)

Fill every level from `field(x, y, z) -> (ρ, ux, uy, uz)`, positions in **base**
index units, so all levels describe the same flow.
"""
function init_chain!(ch::GridChain{T}, field) where {T}
    for k in 1:length(ch)
        a = levels(ch)[k]
        origin, h = level_origin(ch, k)
        nx, ny, nz = size(a, 1), size(a, 2), size(a, 3)
        @inbounds for c in 1:nz, b in 1:ny, i in 1:nx
            ρ, ux, uy, uz = T.(field(origin[1] + T(i - 1) * h,
                                     origin[2] + T(b - 1) * h,
                                     origin[3] + T(c - 1) * h))
            for s in 1:27
                a[i, b, c, s] = cube_equilibrium(s, ρ, ux, uy, uz)
            end
        end
    end
    return ch
end
