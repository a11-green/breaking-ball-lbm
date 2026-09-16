"""
Geometry that turns with the ball.

The rotating boundary condition already imposes `u_wall = ω × r`, which is
everything a smooth sphere needs — a sphere is the same shape at every
orientation. A seamed ball is not. The ridge has to physically move, and since
it is sub-cell at every resolution this hardware can reach (§6.5), "moving" means
moving the `δ` values of interpolated bounce-back: those *are* the seam, as far
as the solver is concerned. Leave them fixed and the ball is a smooth sphere
wearing a stationary pattern.

**What makes re-cutting cheap is that the ball never translates in its own
frame, and that the sphere is rotationally symmetric.** Both together mean the
set of nodes whose classification can *ever* change is a fixed thin shell, known
once at setup:

  * `φ_sphere < 0` — inside the sphere, so solid at every orientation, since the
    seam tube is unioned on and can only add solid;
  * `φ_sphere > h + √3` — no neighbour can be solid, because the tube reaches at
    most `h` above the sphere and a link spans at most √3;
  * in between — the shell, and the only part that is re-cut.

So the boundary columns are numbered once and belong to their node permanently.
A re-cut rewrites values in place: no reallocation, no prefix sum to compact a
new column list, and `WallField` stays exactly the structure the solver and the
CUDA kernel already read.

**Fresh nodes.** When the ridge sweeps past, a node flips from solid to fluid
and its populations are whatever AA-pattern left there as scratch. They are
refilled at equilibrium, with the density averaged over the fluid neighbours and
the velocity of the wall that has just uncovered the node. There are few of
them — the sphere contributes none, and the tube's surface is only a few hundred
cells — but garbage left in even one node is a divergence waiting to happen.
"""

"""
    RotatingWall(geom, dims, dx; center, orientation, margin)

Wall geometry for a seamed ball at a given orientation, re-cuttable in place.

`margin` is how far beyond the ridge the shell reaches, in lattice spacings; it
must exceed √3 so that no node outside the shell can ever have a solid
neighbour. [`shell_is_sufficient`](@ref) checks this against the geometry rather
than trusting the argument.
"""
mutable struct RotatingWall{T<:AbstractFloat}
    wall::WallField{T,Array{Int32,3},Matrix{T}}
    slot::Array{Int32,3}            # permanent column index, 0 outside the shell
    shell::Vector{NTuple{3,Int32}}  # the node each column belongs to
    phi::Vector{T}                  # signed distance, lattice units, this cut
    solid::Vector{Bool}
    was_solid::Vector{Bool}
    geom::BaseballGeometry{T}
    dx::T                           # metres per lattice spacing
    center::NTuple{3,T}
    radius_nodes::T
    orientation::Quat{T}
    fixed_solid::Int          # nodes inside the sphere: solid at every orientation
    nfluid::Int
    recuts::Int
end

function RotatingWall(geom::BaseballGeometry{T}, dims::NTuple{3,<:Integer}, dx::Real;
                      center::NTuple{3,<:Real} = (dims .+ 1) ./ 2,
                      orientation::Quat = one(Quat{T}), margin::Real = 2.0) where {T}
    Δ = T(dx)
    margin > sqrt(3) ||
        throw(ArgumentError("margin must exceed √3 ≈ 1.733 so that no node outside " *
                            "the shell can have a solid neighbour, got $margin"))
    R = geom.radius / Δ
    h = geom.seam_height / Δ
    c = T.(center)

    kind = zeros(Int32, dims)
    slot = zeros(Int32, dims)
    shell = NTuple{3,Int32}[]
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        d = sqrt((T(i) - c[1])^2 + (T(j) - c[2])^2 + (T(k) - c[3])^2) - R
        if d < 0
            kind[i, j, k] = SOLID_NODE          # inside the sphere: always solid
        elseif d <= h + T(margin)
            push!(shell, (Int32(i), Int32(j), Int32(k)))
            slot[i, j, k] = Int32(length(shell))
        end                                      # else: always fluid, never a boundary
    end

    n = length(shell)
    fixed_solid = count(==(SOLID_NODE), kind)
    rw = RotatingWall{T}(WallField(kind, zeros(T, 27, n), c), slot, shell,
                         zeros(T, n), fill(true, n), fill(true, n),
                         geom, Δ, c, R,
                         Quat{T}(orientation.w, orientation.x, orientation.y, orientation.z),
                         fixed_solid, 0, 0)
    recut!(rw, rw.orientation)
    return rw
end

"""
    body_sdf(rw, q)

The signed distance to the ball at orientation `q`, in lattice units, as a
function of a (possibly fractional) node position.

The geometry is fixed in the ball's frame, so the query point is rotated back
rather than the ball rotated forward — the seam polyline is never rebuilt.
"""
function body_sdf(rw::RotatingWall{T}, q::Quat{T}) where {T}
    back = conj(q)
    Δ = rw.dx
    c = rw.center
    return p -> sdf(rw.geom, rotate(back, ((T(p[1]) - c[1]) * Δ,
                                           (T(p[2]) - c[2]) * Δ,
                                           (T(p[3]) - c[3]) * Δ))) / Δ
end

"""
    recut!(rw, q)
    recut!(rw, q, g, spin)

Re-classify the shell at orientation `q`, rewriting `rw.wall` in place. The
four-argument form also refills the nodes the ridge has just uncovered, and
needs `g` in the even-step layout and `spin` in radians per step.

`iterations` is the bisection depth for each wall fraction, and 20 rather than
`refine_delta_cube`'s default 40 because it is the whole cost of a re-cut:
roughly seven solid links on each of the boundary nodes, against a couple of
thousand plain distance evaluations for the classification itself. Twenty halvings
put δ within 1e-6, on a quantity the solver clamps at 1e-3.
"""
function recut!(rw::RotatingWall{T}, q::Quat{T}; iterations::Integer = 20) where {T}
    at = body_sdf(rw, q)
    nx, ny, nz = size(rw.slot)
    kind = rw.wall.kind
    deltas = rw.wall.deltas
    rw.was_solid, rw.solid = rw.solid, rw.was_solid

    @inbounds for n in eachindex(rw.shell)
        i, j, k = rw.shell[n]
        φ = T(at((i, j, k)))
        rw.phi[n] = φ
        rw.solid[n] = φ < 0
    end

    nsolid = 0
    @inbounds for n in eachindex(rw.shell)
        i, j, k = Int(rw.shell[n][1]), Int(rw.shell[n][2]), Int(rw.shell[n][3])
        if rw.solid[n]
            kind[i, j, k] = SOLID_NODE
            nsolid += 1
            continue
        end
        φf = rw.phi[n]
        touches = false
        for s in 1:27
            cx, cy, cz = cube_velocity(s)
            if cx == 0 && cy == 0 && cz == 0
                deltas[s, n] = zero(T)
                continue
            end
            is = shift_periodic(i, cx, nx)
            js = shift_periodic(j, cy, ny)
            ks = shift_periodic(k, cz, nz)
            m = rw.slot[is, js, ks]
            # A neighbour outside the shell is permanently classified, and its
            # `kind` still holds that classification — only shell entries are
            # rewritten here, so this never reads a half-updated value.
            nb_solid = m > 0 ? rw.solid[m] : kind[is, js, ks] == SOLID_NODE
            if !nb_solid
                deltas[s, n] = zero(T)
                continue
            end
            φs = m > 0 ? rw.phi[m] : T(at((is, js, ks)))
            δ = refine_delta_cube(at, i, j, k, s, φf / (φf - φs); iterations = iterations)
            deltas[s, n] = clamp(δ, T(1e-3), one(T))
            touches = true
        end
        kind[i, j, k] = touches ? Int32(n) : Int32(0)
    end

    rw.orientation = q
    rw.recuts += 1
    # Counted from the shell rather than swept from the grid: at production the
    # grid is 3e7 nodes and this runs every eighteen steps.
    rw.nfluid = prod(size(rw.slot)) - rw.fixed_solid - nsolid
    return rw
end

function recut!(rw::RotatingWall{T}, q::Quat{T}, g::Array{T,4},
                spin::NTuple{3,<:Real}) where {T}
    recut!(rw, q)
    return refill_fresh!(g, rw, spin)
end

"""
    refill_fresh!(g, rw, spin)

Give populations to the nodes the last re-cut turned from solid to fluid, and
return how many there were.

Equilibrium at the wall's own velocity, with the density averaged over whichever
neighbours are fluid. That is the simplest defensible choice: the node has just
emerged from inside the body, so the fluid there is moving with the surface, and
there is no non-equilibrium history to extrapolate from that is not itself
invented. It leaves a small pressure transient, which is the honest cost of a
moving boundary on a fixed grid.
"""
function refill_fresh!(g::Array{T,4}, rw::RotatingWall{T},
                       spin::NTuple{3,<:Real}) where {T}
    nx, ny, nz = size(rw.slot)
    ω = T.(spin)
    c = rw.center
    fresh = 0
    @inbounds for n in eachindex(rw.shell)
        (rw.was_solid[n] && !rw.solid[n]) || continue
        i, j, k = Int(rw.shell[n][1]), Int(rw.shell[n][2]), Int(rw.shell[n][3])

        ρsum = zero(T)
        found = 0
        for s in 1:27
            cx, cy, cz = cube_velocity(s)
            (cx == 0 && cy == 0 && cz == 0) && continue
            is = shift_periodic(i, cx, nx)
            js = shift_periodic(j, cy, ny)
            ks = shift_periodic(k, cz, nz)
            rw.wall.kind[is, js, ks] == SOLID_NODE && continue
            ρn = zero(T)
            for t in 1:27
                ρn += g[is, js, ks, t]
            end
            ρsum += ρn
            found += 1
        end
        ρ = found > 0 ? ρsum / found : one(T)

        rx = T(i) - c[1]; ry = T(j) - c[2]; rz = T(k) - c[3]
        ux = ω[2] * rz - ω[3] * ry
        uy = ω[3] * rx - ω[1] * rz
        uz = ω[1] * ry - ω[2] * rx
        for s in 1:27
            g[i, j, k, s] = cube_equilibrium(s, ρ, ux, uy, uz)
        end
        fresh += 1
    end
    return fresh
end

"""
    shell_is_sufficient(rw; samples)

Check the shell really does contain every node whose classification can change,
by re-cutting at `samples` orientations spread over a full turn and confirming
that no node outside the shell ever acquires a solid neighbour.

The margin argument is a claim about the geometry; this is the check of it.
"""
function shell_is_sufficient(rw::RotatingWall{T}; samples::Integer = 24,
                             axis::NTuple{3,<:Real} = (0.3, -1.0, 0.5)) where {T}
    nx, ny, nz = size(rw.slot)
    saved = rw.orientation
    ok = true
    for m in 0:(samples-1)
        recut!(rw, quat_from_axis_angle(axis, 2π * m / samples))
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            rw.slot[i, j, k] == 0 || continue
            rw.wall.kind[i, j, k] == SOLID_NODE && continue   # permanently solid
            for s in 1:27
                cx, cy, cz = cube_velocity(s)
                (cx == 0 && cy == 0 && cz == 0) && continue
                is = shift_periodic(i, cx, nx)
                js = shift_periodic(j, cy, ny)
                ks = shift_periodic(k, cz, nz)
                if rw.wall.kind[is, js, ks] == SOLID_NODE
                    ok = false
                    break
                end
            end
            ok || break
        end
        ok || break
    end
    recut!(rw, saved)
    return ok
end

# --- the coupled loop's view of a rotating wall -----------------------------

flow_fluid_count(rw::RotatingWall) = rw.nfluid

advance_flow!(g::Array{T,4}, rw::RotatingWall{T}, nsteps::Integer, τ::Real;
              kwargs...) where {T} =
    aa_run_walls!(g, rw.wall, nsteps, τ; reduction = :mean, kwargs...)

flow_mean_velocity(g::Array{T,4}, rw::RotatingWall{T},
                   force::NTuple{3,<:Real}) where {T} =
    mean_fluid_velocity(g, rw.wall, force)

"""
    maybe_recut!(g, flow, orientation, spin, drift)

Re-cut the geometry if the surface has turned far enough since the last cut, and
return how many nodes were refilled. Static geometry does nothing.

`drift` is in lattice spacings of travel at the equator. At production a
2,700 rpm ball moves its surface 0.014 spacings per step, so a threshold of a
quarter spacing asks for a cut every eighteen steps or so — which is why the
coupled loop's sub-cycle has to be at least that short, not because the
trajectory needs it.
"""
maybe_recut!(g, flow, q, spin, drift) = 0

function maybe_recut!(g::Array{T,4}, rw::RotatingWall{T}, q::Quat{T},
                      spin::NTuple{3,<:Real}, drift::Real) where {T}
    orientation_drift(rw.orientation, q, rw.radius_nodes) < drift && return 0
    return recut!(rw, q, g, spin)
end

"""
    max_substeps(rw, spin, drift)

The longest sub-cycle the coupled loop may use before the seam outruns its
re-cut threshold.

The geometry is re-cut once per sub-cycle, so this is a hard bound rather than
advice: run longer and the ridge is in the wrong place for the tail of the
sub-cycle. At production it works out near twenty steps, which is short — but a
sub-cycle costs one extra pass over the populations for the box mean, so twenty
steps carries only a few percent of overhead. It is the re-cut, not the
sub-cycle, that has to be made cheap.
"""
function max_substeps(rw::RotatingWall, spin::NTuple{3,<:Real}, drift::Real)
    per = surface_drift_per_step(rw, spin)
    per == 0 && return typemax(Int)
    return max(2, 2 * floor(Int, drift / per / 2))       # even, as AA-pattern requires
end

"""
    surface_drift_per_step(rw, spin)

How far the surface travels at the equator in one lattice step, in spacings —
the number that sets how often the geometry has to be re-cut.
"""
surface_drift_per_step(rw::RotatingWall, spin::NTuple{3,<:Real}) =
    sqrt(sum(abs2, spin)) * rw.radius_nodes
