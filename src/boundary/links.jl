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

struct BounceBackLinks{T<:AbstractFloat}
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

"""
    build_links(ϕ; center = domain centre)

Collect the bounce-back links implied by the signed-distance field `ϕ`, which
must be given in lattice units (one grid spacing = 1). Neighbours wrap
periodically, matching [`stream!`](@ref). `center` is the point torques are
taken about, in node-index units.
"""
function build_links(ϕ::Array{T,3};
                     center::NTuple{3,<:Real} = (size(ϕ) .+ 1) ./ 2) where {T}
    nx, ny, nz = size(ϕ)
    solid = solid_mask(ϕ)
    c = T.(center)

    li = Int32[]; lj = Int32[]; lk = Int32[]; lq = Int32[]
    lδ = T[]; lxw = NTuple{3,T}[]; lsecond = Bool[]

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        solid[i, j, k] && continue
        φf = ϕ[i, j, k]
        for q in 2:Q19
            cx, cy, cz = CX19[q], CY19[q], CZ19[q]
            is, js, ks = mod1(i + cx, nx), mod1(j + cy, ny), mod1(k + cz, nz)
            solid[is, js, ks] || continue

            φs = ϕ[is, js, ks]
            δ = φf / (φf - φs)                      # φf > 0 > φs, so δ ∈ (0, 1)
            δ = clamp(δ, T(1e-3), one(T))

            # The node one step further from the wall, used by the δ < 1/2 branch.
            ib, jb, kb = mod1(i - cx, nx), mod1(j - cy, ny), mod1(k - cz, nz)

            push!(li, i); push!(lj, j); push!(lk, k); push!(lq, q)
            push!(lδ, δ)
            push!(lxw, (T(i) - c[1] + δ * cx, T(j) - c[2] + δ * cy, T(k) - c[3] + δ * cz))
            push!(lsecond, !solid[ib, jb, kb])
        end
    end
    return BounceBackLinks{T}(li, lj, lk, lq, lδ, lxw, lsecond, c)
end
