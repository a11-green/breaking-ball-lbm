"""
A closed-form aerodynamic model — not for production, but for testing the
things around it.

The whole point of this project is to get the force from the flow rather than
from a coefficient table (§1.4). But the 6DOF integrator, the break metrics and
the coupled driver all need *something* to hand them a force before the CFD is
wired in, and a model whose trajectory can be reasoned about independently is
worth more as a test fixture than a recorded CFD sample would be. It is also
what §4.4.3 would need if the loosely-coupled variant is ever revisited.

`C_L ≈ S` for `S < 0.4` is the linear fit of Nathan (2008); `C_D ≈ 0.35` is
representative of a seamed ball in the pitching range, where the seams smear out
the drag crisis a smooth sphere would show (§1.3).
"""

struct CoefficientAero{T<:AbstractFloat}
    props::BallProperties{T}
    CD::T
    CL_slope::T
    ρ::T
    spin_decay::T      # e-folding time of the spin, in seconds; Inf for none
end

function CoefficientAero(p::BallProperties{T}; CD::Real = 0.35, CL_slope::Real = 1.0,
                         ρ::Real = AIR_DENSITY, spin_decay::Real = Inf) where {T}
    return CoefficientAero{T}(p, T(CD), T(CL_slope), T(ρ), T(spin_decay))
end

"""
    (model)(state) -> (force, torque)

Drag along −V̂, Magnus along ω̂ × V̂, both in newtons; torque only if a spin
decay time was given.
"""
function (m::CoefficientAero{T})(s::BallState{T}) where {T}
    U = speed(s)
    zero3 = (zero(T), zero(T), zero(T))
    U == 0 && return zero3, zero3
    û = s.v ./ U
    qdyn = T(0.5) * m.ρ * U^2 * m.props.area

    Fd = (-m.CD * qdyn) .* û

    # ω × V̂ drops the gyro component on its own, which is the physically right
    # thing: spin parallel to the flight direction makes no Magnus force.
    cx = s.ω[2] * û[3] - s.ω[3] * û[2]
    cy = s.ω[3] * û[1] - s.ω[1] * û[3]
    cz = s.ω[1] * û[2] - s.ω[2] * û[1]
    cn = sqrt(cx^2 + cy^2 + cz^2)

    Fl = if cn == 0
        zero3
    else
        S⊥ = cn * m.props.radius / U          # |ω × û| r / U = transverse spin parameter
        (m.CL_slope * S⊥ * qdyn / cn) .* (cx, cy, cz)
    end

    M = isfinite(m.spin_decay) ?
        (-m.props.inertia / m.spin_decay) .* s.ω : zero3
    return (Fd[1] + Fl[1], Fd[2] + Fl[2], Fd[3] + Fl[3]), M
end

"""A model with the Magnus term switched off — the reference for induced break."""
no_magnus(m::CoefficientAero{T}) where {T} =
    CoefficientAero{T}(m.props, m.CD, zero(T), m.ρ, m.spin_decay)

"""A model with no aerodynamics at all — the reference for total movement."""
struct BallisticAero end
(::BallisticAero)(s::BallState{T}) where {T} =
    ((zero(T), zero(T), zero(T)), (zero(T), zero(T), zero(T)))
