"""
D3Q19 lattice.

Velocity ordering: index 1 is the rest velocity, the remaining 18 form
opposite pairs `(2,3), (4,5), ...`, so `opposite(q)` is a cheap index flip.
"""

const Q19 = 19

const CX19 = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const CY19 = (0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1)
const CZ19 = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1)

const W19 = (
    1 / 3,
    1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
    1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
    1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
)

"""Squared lattice speed of sound, ``c_s^2 = 1/3`` in lattice units."""
const CS2 = 1 / 3

"""Index of the velocity pointing opposite to `q`."""
@inline opposite(q::Integer) = q == 1 ? 1 : (iseven(q) ? q + 1 : q - 1)

"""Kinematic viscosity in lattice units for BGK relaxation time `τ`."""
@inline viscosity_from_tau(τ::Real) = CS2 * (τ - 0.5)

"""BGK relaxation time reproducing kinematic viscosity `ν` in lattice units."""
@inline tau_from_viscosity(ν::Real) = ν / CS2 + 0.5

"""
    equilibrium(q, ρ, ux, uy, uz)

Second-order (low-Mach) Maxwell-Boltzmann equilibrium for direction `q`.
"""
@inline function equilibrium(q::Integer, ρ::T, ux::T, uy::T, uz::T) where {T<:AbstractFloat}
    cu = T(CX19[q]) * ux + T(CY19[q]) * uy + T(CZ19[q]) * uz
    usq = ux * ux + uy * uy + uz * uz
    return T(W19[q]) * ρ * (one(T) + cu / T(CS2) + cu * cu / (2 * T(CS2)^2) - usq / (2 * T(CS2)))
end

"""
    nonequilibrium(q, ρ, τ, ∇u)

First-order Chapman-Enskog estimate of the off-equilibrium populations from the
velocity gradient tensor `∇u[α, β] = ∂u_β/∂x_α`. Used to initialise a flow field
without the spurious acoustic transient that a pure-equilibrium start produces.
"""
@inline function nonequilibrium(q::Integer, ρ::T, τ::T, ∇u::AbstractMatrix{T}) where {T<:AbstractFloat}
    c = (T(CX19[q]), T(CY19[q]), T(CZ19[q]))
    qs = zero(T)
    @inbounds for β in 1:3, α in 1:3
        sαβ = (∇u[α, β] + ∇u[β, α]) / 2
        qαβ = c[α] * c[β] - (α == β ? T(CS2) : zero(T))
        qs += qαβ * sαβ
    end
    return -T(W19[q]) * ρ * τ / T(CS2) * qs
end
