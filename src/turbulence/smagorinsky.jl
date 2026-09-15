"""
Smagorinsky subgrid model, coupled to LBM through the relaxation time.

The strain rate is read straight off the non-equilibrium populations rather than
from finite differences, which keeps the model node-local — the property that
makes it cheap on a GPU:

    S_αβ = -Π_αβ / (2 ρ c_s² τ_tot),   Π_αβ = Σ_q c_qα c_qβ (f_q - f_q^eq)

Because `τ_tot` appears on both sides, the closure `ν_t = (C_s Δ)² |S|` with
`|S| = √(2 S_αβ S_αβ)` is a quadratic in `τ_tot` with the closed-form root

    τ_tot = ½ (τ₀ + √(τ₀² + 2√2 (C_s Δ)² ‖Π‖ / (ρ c_s⁴)))

where `‖Π‖ = √(Π_αβ Π_αβ)`.

WALE, which damps the eddy viscosity near walls, is not implemented yet: it
needs the full velocity gradient tensor and so is not node-local.
"""

struct Smagorinsky{T<:AbstractFloat}
    cs::T
    Δ::T
end

"""
    Smagorinsky(; cs = 0.16, Δ = 1.0)

Smagorinsky model with constant `cs` and filter width `Δ` in lattice units.
"""
function Smagorinsky(; cs::Real = 0.16, Δ::Real = 1.0)
    cs >= 0 || throw(ArgumentError("cs must be non-negative, got $cs"))
    T = promote_type(typeof(float(cs)), typeof(float(Δ)))
    return Smagorinsky{T}(T(cs), T(Δ))
end

"""
    total_relaxation_time(model, τ0, ρ, Πnorm)

Relaxation time including the eddy viscosity, from the Frobenius norm `Πnorm`
of the non-equilibrium momentum flux.
"""
@inline function total_relaxation_time(model::Smagorinsky{T}, τ0::T, ρ::T, Πnorm::T) where {T}
    a = 2 * sqrt(T(2)) * (model.cs * model.Δ)^2 * Πnorm / (ρ * T(CS2)^2)
    return (τ0 + sqrt(τ0 * τ0 + a)) / 2
end

"""
    eddy_viscosity(model, τ0, ρ, Πnorm)

Subgrid viscosity implied by `total_relaxation_time`, in lattice units.
"""
@inline function eddy_viscosity(model::Smagorinsky{T}, τ0::T, ρ::T, Πnorm::T) where {T}
    return viscosity_from_tau(total_relaxation_time(model, τ0, ρ, Πnorm)) - viscosity_from_tau(τ0)
end

"""
    nonequilibrium_flux_norm(f, i, j, k, ρ, ux, uy, uz)

`‖Π‖ = √(Π_αβ Π_αβ)` at node `(i, j, k)`, where
`Π_αβ = Σ_q c_qα c_qβ (f_q - f_q^eq)`.
"""
@inline function nonequilibrium_flux_norm(f::Array{T,4}, i::Integer, j::Integer, k::Integer,
                                          ρ::T, ux::T, uy::T, uz::T) where {T}
    Πxx = Πyy = Πzz = Πxy = Πxz = Πyz = zero(T)
    @inbounds for q in 1:Q19
        fneq = f[i, j, k, q] - equilibrium(q, ρ, ux, uy, uz)
        cx, cy, cz = T(CX19[q]), T(CY19[q]), T(CZ19[q])
        Πxx += cx * cx * fneq
        Πyy += cy * cy * fneq
        Πzz += cz * cz * fneq
        Πxy += cx * cy * fneq
        Πxz += cx * cz * fneq
        Πyz += cy * cz * fneq
    end
    return sqrt(Πxx^2 + Πyy^2 + Πzz^2 + 2 * (Πxy^2 + Πxz^2 + Πyz^2))
end

"""
    strain_rate_magnitude(model, τ0, ρ, Πnorm)

`|S| = √(2 S_αβ S_αβ)`, consistent with the relaxation time the model picks.
"""
@inline function strain_rate_magnitude(model::Smagorinsky{T}, τ0::T, ρ::T, Πnorm::T) where {T}
    τtot = total_relaxation_time(model, τ0, ρ, Πnorm)
    return sqrt(T(2)) * Πnorm / (2 * ρ * T(CS2) * τtot)
end
