"""
Boundary links: the lattice directions that cross the solid surface.

For every fluid node with a solid neighbour along direction `q`, one link
records where the wall sits between them. The wall fraction

    δ = φ_f / (φ_f - φ_s)  ∈ (0, 1]

comes from linear interpolation of the signed-distance field along the link,
which is exact for a plane and second-order for a curved surface. `δ = 1/2` is
the halfway case that plain bounce-back assumes.

Stored as a struct of arrays: the GPU port walks these as a flat list.
"""

struct BounceBackLinks{T<:AbstractFloat,L<:Lattice}
    lattice::L
    i::Vector{Int32}
    j::Vector{Int32}
    k::Vector{Int32}
    q::Vector{Int32}
    δ::Vector{T}
    xw::Vector{NTuple{3,T}}     # wall intersection, relative to `center`, lattice units
    second_fluid::Vector{Bool}  # is the node one further from the wall also fluid?
    center::NTuple{3,T}
end

Base.length(l::BounceBackLinks) = length(l.q)

"""
    solid_mask(ϕ)

Nodes where the signed-distance field is negative.
"""
solid_mask(ϕ::Array{T,3}) where {T} = ϕ .< 0

"""The solid nodes of a wall field, host or device, as a Bool array."""
solid_mask(w) = solid_mask_of(w)

"""
    refine_delta(sdf_fn, i, j, k, q, δ0; iterations = 40)

Locate the wall along a link by bisecting the true signed distance, starting
from the linear-interpolation guess `δ0`.

Linear interpolation of φ along a link is exact only for a flat wall. On a
sphere of radius `R` the curvature leaves an O(1/R) error in δ, which at the
handful-of-cells radii these grids afford is a percent-level error in the drag,
so it is worth removing wherever the geometry can be evaluated analytically.
"""
function refine_delta(sdf_fn, lat::Lattice, i::Integer, j::Integer, k::Integer,
                      q::Integer, δ0::T; iterations::Integer = 40) where {T}
    cx, cy, cz = cxs(lat)[q], cys(lat)[q], czs(lat)[q]
    at(t) = sdf_fn((T(i) + t * cx, T(j) + t * cy, T(k) + t * cz))
    lo, hi = zero(T), one(T)
    at(lo) > 0 && at(hi) < 0 || return δ0        # not bracketed: keep the guess
    for _ in 1:iterations
        mid = (lo + hi) / 2
        at(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"""
    build_links(ϕ; center = domain centre, sdf_fn = nothing)

Collect the bounce-back links implied by the signed-distance field `ϕ`, which
must be given in lattice units (one grid spacing = 1). Neighbours wrap
periodically, matching [`stream!`](@ref). `center` is the point torques are
taken about, in node-index units.

`sdf_fn` optionally evaluates the signed distance at an arbitrary (fractional)
node position, in which case each wall fraction is refined against it rather
than linearly interpolated between the two nodes.
"""
function build_links(ϕ::Array{T,3};
                     center::NTuple{3,<:Real} = (size(ϕ) .+ 1) ./ 2,
                     sdf_fn = nothing, lattice::Lattice = D3Q19()) where {T}
    nx, ny, nz = size(ϕ)
    solid = solid_mask(ϕ)
    c = T.(center)

    li = Int32[]; lj = Int32[]; lk = Int32[]; lq = Int32[]
    lδ = T[]; lxw = NTuple{3,T}[]; lsecond = Bool[]

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        solid[i, j, k] && continue
        φf = ϕ[i, j, k]
        for q in 2:nvelocities(lattice)
            cx, cy, cz = cxs(lattice)[q], cys(lattice)[q], czs(lattice)[q]
            is, js, ks = mod1(i + cx, nx), mod1(j + cy, ny), mod1(k + cz, nz)
            solid[is, js, ks] || continue

            φs = ϕ[is, js, ks]
            δ = φf / (φf - φs)                      # φf > 0 > φs, so δ ∈ (0, 1)
            if sdf_fn !== nothing
                δ = refine_delta(sdf_fn, lattice, i, j, k, q, δ)
            end
            δ = clamp(δ, T(1e-3), one(T))

            # The node one step further from the wall, used by the δ < 1/2 branch.
            ib, jb, kb = mod1(i - cx, nx), mod1(j - cy, ny), mod1(k - cz, nz)

            push!(li, i); push!(lj, j); push!(lk, k); push!(lq, q)
            push!(lδ, δ)
            push!(lxw, (T(i) - c[1] + δ * cx, T(j) - c[2] + δ * cy, T(k) - c[3] + δ * cz))
            push!(lsecond, !solid[ib, jb, kb])
        end
    end
    return BounceBackLinks(lattice, li, lj, lk, lq, lδ, lxw, lsecond, c)
end
