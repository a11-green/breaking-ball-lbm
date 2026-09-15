"""
Simulation state: the distribution functions plus the grid metadata.

Populations are stored as `f[x, y, z, q]`. Julia is column-major, so for a fixed
direction `q` consecutive `x` are contiguous in memory — the layout the GPU port
(P2) needs for coalesced access.
"""

mutable struct LBMState{T<:AbstractFloat}
    const nx::Int
    const ny::Int
    const nz::Int
    const τ::T
    f::Array{T,4}
    fnew::Array{T,4}
end

function LBMState{T}(nx::Integer, ny::Integer, nz::Integer, τ::Real) where {T<:AbstractFloat}
    τ > 0.5 || throw(ArgumentError("τ must exceed 0.5 for a positive viscosity (got $τ)"))
    f = Array{T,4}(undef, nx, ny, nz, Q19)
    return LBMState{T}(nx, ny, nz, T(τ), f, similar(f))
end

LBMState(nx::Integer, ny::Integer, nz::Integer, τ::Real) = LBMState{Float64}(nx, ny, nz, τ)

Base.eltype(::LBMState{T}) where {T} = T
Base.size(s::LBMState) = (s.nx, s.ny, s.nz)
viscosity(s::LBMState) = viscosity_from_tau(s.τ)

"""
    macroscopic(s, i, j, k) -> (ρ, ux, uy, uz)

Zeroth and first moments of the populations at one node.
"""
@inline function macroscopic(s::LBMState{T}, i::Integer, j::Integer, k::Integer) where {T}
    ρ = zero(T)
    mx = zero(T)
    my = zero(T)
    mz = zero(T)
    @inbounds for q in 1:Q19
        fq = s.f[i, j, k, q]
        ρ += fq
        mx += T(CX19[q]) * fq
        my += T(CY19[q]) * fq
        mz += T(CZ19[q]) * fq
    end
    return ρ, mx / ρ, my / ρ, mz / ρ
end

"""
    macroscopic!(ρ, ux, uy, uz, s)

Fill preallocated 3D arrays with the macroscopic fields.
"""
function macroscopic!(ρ::Array{T,3}, ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                      s::LBMState{T}) where {T}
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        ρijk, uxijk, uyijk, uzijk = macroscopic(s, i, j, k)
        ρ[i, j, k] = ρijk
        ux[i, j, k] = uxijk
        uy[i, j, k] = uyijk
        uz[i, j, k] = uzijk
    end
    return ρ, ux, uy, uz
end

"""
    macroscopic_fields(s)

Allocate and return `(ρ, ux, uy, uz)` for the whole domain.
"""
function macroscopic_fields(s::LBMState{T}) where {T}
    dims = (s.nx, s.ny, s.nz)
    return macroscopic!(Array{T,3}(undef, dims), Array{T,3}(undef, dims),
                        Array{T,3}(undef, dims), Array{T,3}(undef, dims), s)
end

"""
    total_kinetic_energy(s)

``\\sum \\tfrac{1}{2} ρ |u|^2`` over the domain, in lattice units.
"""
function total_kinetic_energy(s::LBMState{T}) where {T}
    e = zero(T)
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        ρ, ux, uy, uz = macroscopic(s, i, j, k)
        e += ρ * (ux * ux + uy * uy + uz * uz) / 2
    end
    return e
end

"""
    init_equilibrium!(s, field)

Initialise every node from `field(i, j, k) -> (ρ, ux, uy, uz)` at equilibrium.
"""
function init_equilibrium!(s::LBMState{T}, field) where {T}
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        ρ, ux, uy, uz = field(i, j, k)
        for q in 1:Q19
            s.f[i, j, k, q] = equilibrium(q, T(ρ), T(ux), T(uy), T(uz))
        end
    end
    return s
end

"""
    init_with_gradients!(s, field, gradient)

Initialise from `field(i, j, k) -> (ρ, ux, uy, uz)` including the first-order
off-equilibrium part built from `gradient(i, j, k) -> ∇u` (a 3×3 matrix with
`∇u[α, β] = ∂u_β/∂x_α`). Removes the initial acoustic transient that a
pure-equilibrium start would otherwise introduce.
"""
function init_with_gradients!(s::LBMState{T}, field, gradient) where {T}
    ∇u = Matrix{T}(undef, 3, 3)
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        ρ, ux, uy, uz = field(i, j, k)
        ∇u .= gradient(i, j, k)
        for q in 1:Q19
            s.f[i, j, k, q] = equilibrium(q, T(ρ), T(ux), T(uy), T(uz)) +
                              nonequilibrium(q, T(ρ), s.τ, ∇u)
        end
    end
    return s
end
