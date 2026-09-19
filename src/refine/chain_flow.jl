"""
A chain of levels the coupled loop can fly a ball through.

The same arrangement as [`RefinedFlow`](@ref), one level deeper or more: the
ball lives on the deepest grid and only there — the walls, the re-cut, the
momentum-exchange sum — and every level above solves straight through the region
it occupies and has its answer there replaced by restriction.

**What changes with depth, and what does not.** The unit conversions compose:
force gains `m²` and torque `m³` per level climbed, and the body acceleration
and spin lose one factor of `m` per level descended. The mean velocity is still
taken on the **base** level over the whole box, since that is already the
composite average — a base cell is `m^(3n)` deepest cells, so counting base
nodes weights the regions by volume however deep the chain goes. What it has to
exclude is still the ball, and no level above the deepest has wall geometry of
its own, hence the footprint below.

**The sub-cycle gets shorter with every level.** The deepest grid takes `m^n`
steps for each of the base grid's, so the seam travels `m^n` times as far per
base step as the ratio-one case; `max_substeps` accounts for it, and at two
levels the sub-cycle is a quarter of the uniform-grid one.
"""

"""
    ChainFlow(chain, wall; margin = 2)

Bind a [`RotatingWall`](@ref) on the deepest level of a [`GridChain`](@ref) into
something [`couple_step!`](@ref) can drive.

`margin` is how many nodes of the deepest *pair's* coarse level the ball's
footprint must keep clear of the restricted region's edge — the same guarantee
`RefinedFlow` makes, applied where the ball actually is.
"""
mutable struct ChainFlow{T<:AbstractFloat}
    chain::GridChain{T}
    wall::RotatingWall{T}
    base_solid::Array{Bool,3}
    nfluid::Int
    cycles::Int
end

"""
    deepest_index(ch, i, j, k)

The deepest level's index coincident with base node `(i, j, k)`, or `nothing`
when that node is outside the deepest patch. Base nodes inside it always
coincide with one, since every ratio divides the spacing exactly.
"""
function deepest_index(ch::GridChain{T}, i::Integer, j::Integer, k::Integer) where {T}
    origin, h = level_origin(ch, length(ch))
    dims = size(finest_grid(ch))[1:3]
    idx = ntuple(3) do d
        x = (d == 1 ? i : d == 2 ? j : k)
        t = (T(x) - origin[d]) / h + 1
        r = round(Int, t)
        (abs(t - r) > 1e-9 || r < 1 || r > dims[d]) ? 0 : r
    end
    return any(==(0), idx) ? nothing : idx
end

function ChainFlow(chain::GridChain{T}, wall::RotatingWall{T};
                   margin::Integer = 2) where {T}
    size(wall.wall.kind) == size(finest_grid(chain))[1:3] ||
        throw(ArgumentError("the wall is sized for $(size(wall.wall.kind)) but the " *
                            "deepest level is $(size(finest_grid(chain))[1:3])"))

    nbx, nby, nbz = size(base_grid(chain))[1:3]
    solid = falses(nbx, nby, nbz)
    cf = ChainFlow{T}(chain, wall, solid, 0, 0)
    refresh_base_solid!(cf)

    # The margin is checked where the ball is: against the deepest pair's own
    # restricted region, in that pair's coarse indices.
    deep = chain.levels[end]
    kind = wall.wall.kind
    idx = findall(kind .== SOLID_NODE)
    if !isempty(idx)
        # A deepest node `a` sits at its pair's coarse index lo + (a - 1)/m.
        lo_s = ntuple(d -> deep.lo[d] + (minimum(x -> x[d], idx) - 1) ÷ deep.ratio, 3)
        hi_s = ntuple(d -> deep.lo[d] + (maximum(x -> x[d], idx) - 1) ÷ deep.ratio, 3)
        all(lo_s .>= deep.lo .+ (1 + margin)) && all(hi_s .<= deep.hi .- (1 + margin)) ||
            throw(ArgumentError("the ball reaches $lo_s..$hi_s on the level above it, " *
                                "too close to the restricted region " *
                                "$(deep.lo .+ 1)..$(deep.hi .- 1); enlarge the patch"))
    end
    return cf
end

"""Re-derive the ball's footprint on the base level after a re-cut."""
function refresh_base_solid!(cf::ChainFlow)
    ch = cf.chain
    nbx, nby, nbz = size(base_grid(ch))[1:3]
    kind = cf.wall.wall.kind
    @inbounds for k in 1:nbz, j in 1:nby, i in 1:nbx
        idx = deepest_index(ch, i, j, k)
        cf.base_solid[i, j, k] = idx === nothing ? false :
                                 kind[idx[1], idx[2], idx[3]] == SOLID_NODE
    end
    cf.nfluid = count(!, cf.base_solid)
    return cf
end

# --- the coupled loop's operations -----------------------------------------

flow_fluid_count(cf::ChainFlow) = cf.nfluid
flow_wall(cf::ChainFlow) = flow_wall(cf.wall)
flow_recuts(cf::ChainFlow) = flow_recuts(cf.wall)
flow_coarse_solid(cf::ChainFlow) = cf.base_solid

"""Mass-averaged density and velocity over the base level, skipping the ball."""
function flow_mean_velocity(ch::GridChain{T}, cf::ChainFlow{T},
                            force::NTuple{3,<:Real}) where {T}
    g = base_grid(ch)
    nx, ny, nz = size(g)[1:3]
    Fx, Fy, Fz = Float64(force[1]) / 2, Float64(force[2]) / 2, Float64(force[3]) / 2
    ρtot = 0.0
    mx = 0.0; my = 0.0; mz = 0.0
    n = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        cf.base_solid[i, j, k] && continue
        ρ, ux, uy, uz = node_macroscopic(g, i, j, k, T)
        ρtot += ρ
        mx += Float64(ρ * ux) + Fx
        my += Float64(ρ * uy) + Fy
        mz += Float64(ρ * uz) + Fz
        n += 1
    end
    n == 0 && return zero(T), (zero(T), zero(T), zero(T))
    return T(ρtot / n), (T(mx / ρtot), T(my / ρtot), T(mz / ρtot))
end

"""
    advance_flow!(ch, cf, nsteps, τ; ...)

`nsteps` **base** steps, with the force and torque averaged over them and
returned in base lattice units, so the rest of the loop needs to know nothing
about how deep the chain is.
"""
function advance_flow!(ch::GridChain{T}, cf::ChainFlow{T}, nsteps::Integer, τ::Real;
                       force::NTuple{3,<:Real} = (0, 0, 0),
                       spin::NTuple{3,<:Real} = (0, 0, 0),
                       operator::Symbol = :central_moment,
                       rule::Symbol = :interpolated_local, smagorinsky::Real = 0.0,
                       omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                       layers::Integer = 3, filtered::Bool = false,
                       channel = nothing,
                       inlet::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    npairs = length(ch.levels)
    m = ch.levels[1].ratio
    scratch = ntuple(_ -> Vector{T}(undef, 27), 3)

    sF = (zero(T), zero(T), zero(T))
    sM = (zero(T), zero(T), zero(T))
    cycles = nsteps ÷ 2
    for _ in 1:cycles
        F, M = chain_cycle_walls!(ch, cf.wall; force = force, spin = spin,
                                  operator = operator, rule = rule,
                                  smagorinsky = smagorinsky,
                                  layers = layers, filtered = filtered,
                                  channel = channel, inlet = inlet,
                                  omega_bulk = omega_bulk, omega_higher = omega_higher,
                                  scratch = scratch)
        sF = sF .+ F
        sM = sM .+ M
        cf.cycles += 1
    end
    w = one(T) / cycles
    return chain_force(sF .* w, m, npairs), chain_torque(sM .* w, m, npairs)
end

"""
    maybe_recut!(ch, cf, q, spin, drift)

Re-cut the seam on the deepest level. `spin` arrives in **base** units and loses
one factor of the ratio per level, as the body force does.
"""
function maybe_recut!(ch::GridChain{T}, cf::ChainFlow{T}, q::Quat{T},
                      spin::NTuple{3,<:Real}, drift::Real) where {T}
    npairs = length(ch.levels)
    ωd = T.(spin) ./ T(ch.levels[1].ratio)^npairs
    fresh = maybe_recut!(finest_grid(ch), cf.wall, q, ωd, drift)
    fresh == 0 && return 0
    refresh_base_solid!(cf)
    return fresh
end

"""
    max_substeps(cf, spin, drift)

The longest sub-cycle in **base** steps before the seam outruns its re-cut
threshold. The deepest level takes `m^n` steps per base step, so this shortens
by that factor with every level added.
"""
function max_substeps(cf::ChainFlow, spin::NTuple{3,<:Real}, drift::Real)
    ch = cf.chain
    npairs = length(ch.levels)
    m = ch.levels[1].ratio
    ωd = spin ./ m^npairs
    per = surface_drift_per_step(cf.wall, ωd)
    per == 0 && return typemax(Int)
    return max(2, 2 * floor(Int, drift / (per * m^npairs) / 2))
end

"""Fill every level from a macroscopic field and take the ball's footprint."""
function init_chain_flow!(cf::ChainFlow, field)
    init_chain!(cf.chain, field)
    refresh_base_solid!(cf)
    return cf
end
