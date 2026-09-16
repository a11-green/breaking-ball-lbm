"""
Discrete velocity sets.

Both lattices index the rest velocity first and then list opposite pairs
`(2,3), (4,5), …`, so `opposite(q)` is a cheap index flip either way.

D3Q19 is the lean choice for verification and low-Reynolds work. D3Q27 is the
full tensor product `{-1,0,1}³`, which the production runs need: it is the only
one of the two whose moments factorise direction by direction, and that
factorisation is what the central-moment collision operator is built on.
"""

abstract type Lattice end

struct D3Q19 <: Lattice end
struct D3Q27 <: Lattice end

const CX19 = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const CY19 = (0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1)
const CZ19 = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1)

const W19 = (
    1 / 3,
    1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
    1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
    1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
)

# D3Q19 plus the eight corners.
const CX27 = (CX19..., 1, -1, 1, -1, 1, -1, -1, 1)
const CY27 = (CY19..., 1, -1, 1, -1, -1, 1, 1, -1)
const CZ27 = (CZ19..., 1, -1, -1, 1, 1, -1, 1, -1)

const W27 = (
    8 / 27,
    2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27,
    1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
    1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
    1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216,
)

@inline nvelocities(::D3Q19) = 19
@inline nvelocities(::D3Q27) = 27

@inline cxs(::D3Q19) = CX19
@inline cys(::D3Q19) = CY19
@inline czs(::D3Q19) = CZ19
@inline weights(::D3Q19) = W19

@inline cxs(::D3Q27) = CX27
@inline cys(::D3Q27) = CY27
@inline czs(::D3Q27) = CZ27
@inline weights(::D3Q27) = W27

"""Squared lattice speed of sound, ``c_s^2 = 1/3`` in lattice units."""
const CS2 = 1 / 3

"""Index of the velocity pointing opposite to `q`, for either lattice."""
@inline opposite(q::Integer) = q == 1 ? 1 : (iseven(q) ? q + 1 : q - 1)

"""Kinematic viscosity in lattice units for BGK relaxation time `τ`."""
@inline viscosity_from_tau(τ::Real) = CS2 * (τ - 0.5)

"""BGK relaxation time reproducing kinematic viscosity `ν` in lattice units."""
@inline tau_from_viscosity(ν::Real) = ν / CS2 + 0.5

"""
    equilibrium(lat, q, ρ, ux, uy, uz)

Second-order (low-Mach) Maxwell-Boltzmann equilibrium for direction `q`.
"""
@inline function equilibrium(lat::Lattice, q::Integer, ρ::T, ux::T, uy::T, uz::T) where {T<:AbstractFloat}
    cu = T(cxs(lat)[q]) * ux + T(cys(lat)[q]) * uy + T(czs(lat)[q]) * uz
    usq = ux * ux + uy * uy + uz * uz
    # Exact reciprocals of c_s²: dividing costs a real division (see aa_pattern.jl).
    return T(weights(lat)[q]) * ρ *
           (one(T) - T(1.5) * usq + T(3) * cu + T(4.5) * cu * cu)
end

"""
    nonequilibrium(lat, q, ρ, τ, ∇u)

First-order Chapman-Enskog estimate of the off-equilibrium populations from the
velocity gradient tensor `∇u[α, β] = ∂u_β/∂x_α`. Used to initialise a flow field
without the spurious acoustic transient that a pure-equilibrium start produces.
"""
@inline function nonequilibrium(lat::Lattice, q::Integer, ρ::T, τ::T,
                                ∇u::AbstractMatrix{T}) where {T<:AbstractFloat}
    c = (T(cxs(lat)[q]), T(cys(lat)[q]), T(czs(lat)[q]))
    qs = zero(T)
    @inbounds for β in 1:3, α in 1:3
        sαβ = (∇u[α, β] + ∇u[β, α]) / 2
        qαβ = c[α] * c[β] - (α == β ? T(CS2) : zero(T))
        qs += qαβ * sαβ
    end
    return -T(weights(lat)[q]) * ρ * τ / T(CS2) * qs
end
