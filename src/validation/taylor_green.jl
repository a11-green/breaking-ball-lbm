"""
Decaying Taylor-Green vortex — the analytic reference case for V&V-1.

The 2D vortex is embedded in the 3D lattice (uniform along z), so it exercises
the full D3Q19 machinery while still having an exact incompressible solution:

    u_x = -u₀ cos(kx) sin(ky) e^{-2νk²t}
    u_y =  u₀ sin(kx) cos(ky) e^{-2νk²t}
    p   = -ρ₀u₀²/4 (cos 2kx + cos 2ky) e^{-4νk²t}

All quantities are in lattice units. Node `i` sits at `x = i - 1`, so a domain of
`n` cells holds exactly one wavelength for `k = 2π/n`.
"""

struct TaylorGreen{T<:AbstractFloat}
    u0::T
    k::T
    ρ0::T
    ν::T
end

"""
    TaylorGreen(; u0, n, ν, ρ0 = 1.0)

Vortex with peak velocity `u0` and one wavelength across `n` cells.
"""
function TaylorGreen(; u0::Real, n::Integer, ν::Real, ρ0::Real = 1.0)
    T = promote_type(typeof(float(u0)), typeof(float(ν)), typeof(float(ρ0)))
    return TaylorGreen{T}(T(u0), T(2π / n), T(ρ0), T(ν))
end

"""Velocity decay factor at time `t` (in lattice time steps)."""
@inline decay(tg::TaylorGreen, t::Real) = exp(-2 * tg.ν * tg.k^2 * t)

"""Analytic velocity `(ux, uy, uz)` at position `(x, y)` and time `t`."""
@inline function velocity(tg::TaylorGreen{T}, x::Real, y::Real, t::Real) where {T}
    e = T(decay(tg, t))
    kx = tg.k * T(x)
    ky = tg.k * T(y)
    return (-tg.u0 * cos(kx) * sin(ky) * e, tg.u0 * sin(kx) * cos(ky) * e, zero(T))
end

"""Analytic density at position `(x, y)` and time `t` (via `p = ρ c_s²`)."""
@inline function density(tg::TaylorGreen{T}, x::Real, y::Real, t::Real) where {T}
    e = T(decay(tg, t))^2
    p = -tg.ρ0 * tg.u0^2 / 4 * (cos(2 * tg.k * T(x)) + cos(2 * tg.k * T(y))) * e
    return tg.ρ0 + p / T(CS2)
end

"""Analytic velocity gradient `∇u[α, β] = ∂u_β/∂x_α` at `(x, y)` and time `t`."""
function velocity_gradient(tg::TaylorGreen{T}, x::Real, y::Real, t::Real) where {T}
    e = T(decay(tg, t))
    kx = tg.k * T(x)
    ky = tg.k * T(y)
    a = tg.u0 * tg.k * e
    ∇u = zeros(T, 3, 3)
    ∇u[1, 1] = a * sin(kx) * sin(ky)
    ∇u[2, 1] = -a * cos(kx) * cos(ky)
    ∇u[1, 2] = a * cos(kx) * cos(ky)
    ∇u[2, 2] = -a * sin(kx) * sin(ky)
    return ∇u
end

"""
    init!(s, tg)

Initialise the state from the analytic solution at `t = 0`, including the
first-order off-equilibrium part.
"""
function init!(s::LBMState{T}, tg::TaylorGreen) where {T}
    field = (i, j, k) -> begin
        x, y = i - 1, j - 1
        ux, uy, uz = velocity(tg, x, y, 0)
        (density(tg, x, y, 0), ux, uy, uz)
    end
    gradient = (i, j, k) -> velocity_gradient(tg, i - 1, j - 1, 0)
    return init_with_gradients!(s, field, gradient)
end

"""
    l2_velocity_error(s, tg, t) -> (relative_error, absolute_error)

L2 norm of the difference between the simulated and analytic velocity fields at
time `t`, both absolute and relative to the analytic field.
"""
function l2_velocity_error(s::LBMState{T}, tg::TaylorGreen, t::Real) where {T}
    err = zero(T)
    ref = zero(T)
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        _, ux, uy, uz = macroscopic(s, i, j, k)
        ex, ey, ez = velocity(tg, i - 1, j - 1, t)
        err += (ux - ex)^2 + (uy - ey)^2 + (uz - ez)^2
        ref += ex^2 + ey^2 + ez^2
    end
    return sqrt(err / ref), sqrt(err / (s.nx * s.ny * s.nz))
end
