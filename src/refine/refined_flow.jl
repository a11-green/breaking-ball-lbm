"""
A refined grid the coupled loop can fly a ball through.

The ball lives on the fine level and only there: the walls, the re-cut, the
momentum-exchange sum. The coarse level is plain fluid that happens to be solved
through the region the ball occupies, and whose answer there is thrown away and
replaced by the fine one every cycle.

**Units between the levels.** Everything the loop exchanges has to be stated in
one level's units, and the coarse level's are the ones the rest of the code
already speaks. Force carries `ρ L⁴/T²` and torque `ρ L⁵/T²`, so under acoustic
scaling with ratio `m`

    F_coarse = F_fine / m²        T_coarse = T_fine / m³
    ω_fine   = ω_coarse / m       a_fine   = a_coarse / m

and `to_physical_force` on the coarse units then works unchanged. Getting `m³`
where `m²` belongs would put the torque out by a factor of two, which is exactly
the size of a real spin-decay signal — so the conversions are tested against the
same physical force expressed on both levels rather than asserted.

**The mean velocity is taken on the coarse level, over the whole box.** That is
already the composite average: the covered region carries the restricted fine
solution, and a coarse cell is `m³` fine cells, so counting coarse nodes weights
the two regions by volume automatically. What it has to exclude is the ball, and
the coarse level has no wall geometry of its own — hence `coarse_solid`, the
ball's footprint on the coarse grid, used for nothing else.
"""

"""
    RefinedFlow(grid, wall; margin = 2)

Bind a [`RotatingWall`](@ref) on the fine level of a [`TwoGrid`](@ref) into
something [`couple_step!`](@ref) can drive.

`margin` is how many coarse nodes the ball's footprint must keep clear of the
restricted region's edge. The coarse level is solved straight through the ball
as if it were fluid, so the values it produces there are meaningless; they are
overwritten by the restriction every cycle, and the margin is what guarantees
they cannot reach the interface before that happens.
"""
mutable struct RefinedFlow{T<:AbstractFloat}
    grid::TwoGrid{T}
    wall::RotatingWall{T}
    coarse_solid::Array{Bool,3}
    nfluid::Int
    cycles::Int
end

function RefinedFlow(grid::TwoGrid{T}, wall::RotatingWall{T}; margin::Integer = 2) where {T}
    size(wall.wall.kind) == size(grid.fine)[1:3] ||
        throw(ArgumentError("the wall is sized for $(size(wall.wall.kind)) but the " *
                            "fine level is $(size(grid.fine)[1:3])"))

    ncx, ncy, ncz = size(grid.coarse)[1:3]
    solid = falses(ncx, ncy, ncz)
    # The ball's footprint on the coarse grid, taken from the fine classification
    # so the two levels cannot disagree about where it is.
    for k in 1:ncz, j in 1:ncy, i in 1:ncx
        all((i, j, k) .>= grid.lo) && all((i, j, k) .<= grid.hi) || continue
        a = grid.ratio * (i - grid.lo[1]) + 1
        b = grid.ratio * (j - grid.lo[2]) + 1
        c = grid.ratio * (k - grid.lo[3]) + 1
        solid[i, j, k] = wall.wall.kind[a, b, c] == SOLID_NODE
    end

    if any(solid)
        idx = findall(solid)
        lo_s = ntuple(d -> minimum(x -> x[d], idx), 3)
        hi_s = ntuple(d -> maximum(x -> x[d], idx), 3)
        all(lo_s .>= grid.lo .+ (1 + margin)) && all(hi_s .<= grid.hi .- (1 + margin)) ||
            throw(ArgumentError("the ball reaches coarse $lo_s..$hi_s, too close to the " *
                                "restricted region $(grid.lo .+ 1)..$(grid.hi .- 1); " *
                                "enlarge the fine patch"))
    end

    return RefinedFlow{T}(grid, wall, solid, count(!, solid), 0)
end

"""The ball's solid footprint as a `WallField`-shaped mask, for the coarse mean."""
struct CoarseMask{A<:AbstractArray{Bool,3}}
    solid::A
end

# --- the coupled loop's three operations -----------------------------------

flow_fluid_count(rf::RefinedFlow) = rf.nfluid

"""
Mass-averaged density and velocity over the coarse level, skipping the ball.

Double accumulation, for the reason given on `mean_fluid_velocity`: what the
controller acts on is a difference two or three orders below the mean.
"""
function flow_mean_velocity(rg::TwoGrid{T}, rf::RefinedFlow{T},
                            force::NTuple{3,<:Real}) where {T}
    g = rg.coarse
    nx, ny, nz = size(g)[1:3]
    Fx, Fy, Fz = Float64(force[1]) / 2, Float64(force[2]) / 2, Float64(force[3]) / 2
    ρtot = 0.0
    mx = 0.0; my = 0.0; mz = 0.0
    n = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        rf.coarse_solid[i, j, k] && continue
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

"""The wall field the fine level is solving against."""
flow_wall(rf::RefinedFlow) = flow_wall(rf.wall)
flow_recuts(rf::RefinedFlow) = flow_recuts(rf.wall)

"""
    refine_cycle_walls!(rg, wall, ...)

One cycle — two coarse steps, four fine ones — with the ball on the fine level.
Returns the `(force, torque)` averaged over the cycle, in *fine* lattice units.
"""
function refine_cycle_walls!(rg::TwoGrid{T}, wall::RotatingWall{T};
                             force::NTuple{3,<:Real} = (0, 0, 0),
                             spin::NTuple{3,<:Real} = (0, 0, 0),
                             operator::Symbol = :central_moment,
                             rule::Symbol = :interpolated_local,
                             smagorinsky::Real = 0.0,
                             layers::Integer = 3, filtered::Bool = false,
                             omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                             channel = nothing, inlet::NTuple{3,<:Real} = (0, 0, 0),
                             scratch = ntuple(_ -> Vector{T}(undef, 27), 3)) where {T}
    Fc = T.(force)
    Ff = fine_force(Fc, rg.ratio)
    ωf = T.(spin)

    # The faces are the coarse level's (see `refine_cycle!`), and a cycle
    # advances it by the pair the buffer is imposed on.
    save_coarse!(rg)
    aa_run!(rg.coarse, 2, rg.τc; force = Fc, operator = operator,
            smagorinsky = smagorinsky, channel = channel, inlet = inlet,
            omega_bulk = omega_bulk, omega_higher = omega_higher)

    sF = (zero(T), zero(T), zero(T))
    sM = (zero(T), zero(T), zero(T))
    for half in 1:2
        F, M = aa_run_walls!(rg.fine, wall.wall, 2, rg.τf; force = Ff, spin = ωf,
                             operator = operator, rule = rule, reduction = :mean,
                             smagorinsky = smagorinsky,
                             omega_bulk = omega_bulk, omega_higher = omega_higher)
        sF = sF .+ F
        sM = sM .+ M
        interface_fill!(rg, half / 2; layers = layers, scratch = scratch)
    end

    restrict!(rg; filtered = filtered, scratch = scratch)
    return sF ./ 2, sM ./ 2
end

"""
    advance_flow!(rg, rf, nsteps, τ; ...)

`nsteps` coarse steps — that is, `nsteps ÷ 2` cycles — with the returned force
and torque averaged over them and expressed in *coarse* lattice units, so the
rest of the loop needs to know nothing about the refinement.
"""
function advance_flow!(rg::TwoGrid{T}, rf::RefinedFlow{T}, nsteps::Integer, τ::Real;
                       force::NTuple{3,<:Real} = (0, 0, 0),
                       spin::NTuple{3,<:Real} = (0, 0, 0),
                       operator::Symbol = :central_moment,
                       rule::Symbol = :interpolated_local, smagorinsky::Real = 0.0,
                       omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                       layers::Integer = 3, filtered::Bool = false,
                       channel = nothing,
                       inlet::NTuple{3,<:Real} = (0, 0, 0)) where {T}
    iseven(nsteps) || throw(ArgumentError("nsteps must be even, got $nsteps"))
    m = rg.ratio
    ωf = T.(spin) ./ m                      # ω_fine = ω_coarse / m
    scratch = ntuple(_ -> Vector{T}(undef, 27), 3)

    sF = (zero(T), zero(T), zero(T))
    sM = (zero(T), zero(T), zero(T))
    cycles = nsteps ÷ 2
    for _ in 1:cycles
        F, M = refine_cycle_walls!(rg, rf.wall; force = force, spin = ωf,
                                   operator = operator, rule = rule,
                                   smagorinsky = smagorinsky,
                                   layers = layers, filtered = filtered,
                                   channel = channel, inlet = inlet,
                                   omega_bulk = omega_bulk, omega_higher = omega_higher,
                                   scratch = scratch)
        sF = sF .+ F
        sM = sM .+ M
        rf.cycles += 1
    end
    w = one(T) / cycles
    return coarse_force(sF .* w, m), coarse_torque(sM .* w, m)
end

"""Fine-level force in coarse lattice units: force carries `ρ L⁴/T²`."""
coarse_force(F::NTuple{3,T}, m::Integer) where {T} = F ./ T(m)^2

"""Fine-level torque in coarse lattice units: torque carries `ρ L⁵/T²`."""
coarse_torque(M::NTuple{3,T}, m::Integer) where {T} = M ./ T(m)^3

"""
    maybe_recut!(rg, rf, q, spin, drift)

Re-cut the seam on the fine level. `spin` arrives in coarse lattice units and is
converted; the surface travel per step works out the same on both levels, since
the fine level's angle per step is `m` times smaller and its radius `m` times
larger.
"""
function maybe_recut!(rg::TwoGrid{T}, rf::RefinedFlow{T}, q::Quat{T},
                      spin::NTuple{3,<:Real}, drift::Real) where {T}
    ωf = T.(spin) ./ rg.ratio
    fresh = maybe_recut!(rg.fine, rf.wall, q, ωf, drift)
    fresh == 0 && return 0
    # Nodes flipped, so the footprint the coarse mean skips has moved with them.
    refresh_coarse_solid!(rf)
    return fresh
end

"""Re-derive the ball's coarse footprint after a re-cut."""
function refresh_coarse_solid!(rf::RefinedFlow)
    g = rf.grid
    ncx, ncy, ncz = size(g.coarse)[1:3]
    @inbounds for k in 1:ncz, j in 1:ncy, i in 1:ncx
        all((i, j, k) .>= g.lo) && all((i, j, k) .<= g.hi) || continue
        a = g.ratio * (i - g.lo[1]) + 1
        b = g.ratio * (j - g.lo[2]) + 1
        c = g.ratio * (k - g.lo[3]) + 1
        rf.coarse_solid[i, j, k] = rf.wall.wall.kind[a, b, c] == SOLID_NODE
    end
    rf.nfluid = count(!, rf.coarse_solid)
    return rf
end

"""
    max_substeps(rf, spin, drift)

The longest sub-cycle, in *coarse* steps, before the seam outruns its re-cut
threshold. Half what a uniform grid of the coarse spacing would allow, because a
cycle advances the fine level twice as far.
"""
function max_substeps(rf::RefinedFlow, spin::NTuple{3,<:Real}, drift::Real)
    ωf = spin ./ rf.grid.ratio
    per = surface_drift_per_step(rf.wall, ωf)
    per == 0 && return typemax(Int)
    return max(2, 2 * floor(Int, drift / (per * rf.grid.ratio) / 2))
end

"""
    init_refined_flow!(rf, field)

Fill both levels from a macroscopic field, as [`init_refined!`](@ref) does, and
take the ball's coarse footprint from the geometry as it stands.
"""
function init_refined_flow!(rf::RefinedFlow, field)
    init_refined!(rf.grid, field)
    refresh_coarse_solid!(rf)
    return rf
end
