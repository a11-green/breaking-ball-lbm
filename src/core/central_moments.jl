"""
Central-moment ("cascaded") collision on D3Q27, with the cumulants of order
three and above relaxed to zero.

BGK gives every population the same relaxation rate, so driving the viscosity
down (τ → 1/2) drives *everything* to the edge of over-relaxation, and the
scheme comes apart at the Reynolds numbers this project needs. Here the moments
are taken in the frame moving with the fluid and relaxed by what they mean:

  * the deviatoric second-order moments carry the shear viscosity, at rate ω;
  * their trace is the acoustic mode and relaxes at its own rate `ωb ≈ 1`, so
    the bulk viscosity stays well damped however small ω⁻¹ - 1/2 becomes. This
    is what buys the stability margin — on a doubly periodic shear layer BGK
    fails by τ = 0.501 while this operator still runs at τ = 0.5001;
  * everything above second order has its cumulant set to zero, meaning those
    moments are rebuilt from the *current* second-order ones through Isserlis'
    theorem rather than frozen at their isotropic equilibrium values. Freezing
    them instead leaves an error proportional to the strain rate that costs
    about a quarter of an order of grid convergence.

The transform is exact and cheap because D3Q27 is the tensor product
`{-1,0,1}³`: the 27 populations factorise into three one-dimensional transforms,
each mapping three populations to the moments of order 0, 1 and 2 about `u`,

    κ₀ = f₋ + f₀ + f₊
    κ₁ = (f₊ - f₋) - u κ₀
    κ₂ = (f₊ + f₋) - 2u (f₊ - f₋) + u² κ₀

which invert in closed form.

A body force needs no special scheme here. In the co-moving frame it contributes
to the first-order moments and nothing else, so the momentum increment is
applied exactly by setting `κ₁ = F/2` after relaxation (it was `-F/2` before,
since `u` already carries the half-force correction).

What this does *not* fix is the lattice's own cubic defect: `c³ = c` for
`c ∈ {-1,0,1}` pins the third central moment at `-ρu³` where a Maxwellian would
have zero, so the residual Galilean error is comparable to BGK's. Geier's
cumulant method adds explicit correction terms built from velocity gradients to
cancel it; those are not implemented here.
"""

# Position of each lattice direction inside the 3×3×3 cube, strides (1, 3, 9).
const CUBE27 = ntuple(q -> 1 + (CX27[q] + 1) + 3 * (CY27[q] + 1) + 9 * (CZ27[q] + 1), 27)

"""
    gaussian_moment(m, n, o, A, B, C, D, E, F)

`E[xᵐ yⁿ z^o]` for a zero-mean Gaussian with covariance
`[A D E; D B F; E F C]`, from Isserlis' theorem. Setting the cumulants of order
three and above to zero is the same as giving those moments these values, so
this is what the higher moments relax to.

Only the multi-indices D3Q27 can represent are covered: every exponent is at
most two, so the even cases are order four — permutations of `(2,2,0)` and
`(2,1,1)` — and the single order-six term `(2,2,2)`. Odd orders vanish.
"""
@inline function gaussian_moment(m::Int, n::Int, o::Int, A::T, B::T, C::T,
                                 D::T, E::T, F::T) where {T}
    isodd(m + n + o) && return zero(T)
    if (m, n, o) == (2, 2, 0)
        return A * B + 2 * D * D
    elseif (m, n, o) == (2, 0, 2)
        return A * C + 2 * E * E
    elseif (m, n, o) == (0, 2, 2)
        return B * C + 2 * F * F
    elseif (m, n, o) == (2, 1, 1)
        return A * F + 2 * D * E
    elseif (m, n, o) == (1, 2, 1)
        return B * E + 2 * D * F
    elseif (m, n, o) == (1, 1, 2)
        return C * D + 2 * E * F
    elseif (m, n, o) == (2, 2, 2)
        return A * B * C + 2 * A * F * F + 2 * B * E * E + 2 * C * D * D + 8 * D * E * F
    end
    return zero(T)   # unreachable for exponents ≤ 2 with even order ≥ 3
end

"""Forward one-dimensional transform: three populations → moments about `u`."""
@inline function to_moments_1d!(buf::AbstractVector{T}, base::Int, stride::Int, u::T) where {T}
    @inbounds begin
        fm = buf[base]                 # c = -1
        f0 = buf[base+stride]          # c =  0
        fp = buf[base+2stride]         # c = +1
        k0 = fm + f0 + fp
        d = fp - fm
        buf[base] = k0
        buf[base+stride] = d - u * k0
        buf[base+2stride] = (fp + fm) - 2 * u * d + u * u * k0
    end
end

"""Inverse of [`to_moments_1d!`](@ref)."""
@inline function to_populations_1d!(buf::AbstractVector{T}, base::Int, stride::Int, u::T) where {T}
    @inbounds begin
        k0 = buf[base]
        k1 = buf[base+stride]
        k2 = buf[base+2stride]
        d = k1 + u * k0                # f₊ - f₋
        sum2 = k2 + 2 * u * d - u * u * k0   # f₊ + f₋
        buf[base] = (sum2 - d) / 2
        buf[base+stride] = k0 - sum2
        buf[base+2stride] = (sum2 + d) / 2
    end
end

"""Transform the whole cube in place, populations → central moments about `u`."""
@inline function to_moments!(buf::AbstractVector{T}, ux::T, uy::T, uz::T) where {T}
    for c in 0:2, b in 0:2
        to_moments_1d!(buf, 1 + 3b + 9c, 1, ux)
    end
    for c in 0:2, a in 0:2
        to_moments_1d!(buf, 1 + a + 9c, 3, uy)
    end
    for b in 0:2, a in 0:2
        to_moments_1d!(buf, 1 + a + 3b, 9, uz)
    end
end

"""Inverse of [`to_moments!`](@ref)."""
@inline function to_populations!(buf::AbstractVector{T}, ux::T, uy::T, uz::T) where {T}
    for b in 0:2, a in 0:2
        to_populations_1d!(buf, 1 + a + 3b, 9, uz)
    end
    for c in 0:2, a in 0:2
        to_populations_1d!(buf, 1 + a + 9c, 3, uy)
    end
    for c in 0:2, b in 0:2
        to_populations_1d!(buf, 1 + 3b + 9c, 1, ux)
    end
end

"""
    relax_moments!(buf, ρ, ω, force, ωb = 1, ωh = 1, ωo = ωh)

Relax central moments in place: mass is untouched, momentum takes the body
force, the deviatoric second-order moments relax at `ω` and their trace at `ωb`,
and everything of third order and above has its cumulant relaxed to zero — the
even orders at `ωh`, the odd ones at `ωo`.

**Why the odd orders get their own rate.** Bounce-back does not put the wall
where the geometry says it is. TRT says it puts it where the magic parameter
`Λ = (1/ω⁺ - 1/2)(1/ω⁻ - 1/2)` says, and that only `Λ = 3/16` puts it halfway.
`ω⁺` is the rate the even non-equilibrium moments relax at, which is the one
carrying the shear viscosity, so `1/ω⁺ - 1/2 = τ - 1/2` and it is not free. `ω⁻`
is the rate the odd ones relax at, and here those are the third- and fifth-order
cumulants — this argument, not `ω`. Under BGK the two rates are the same and `Λ`
collapses as `(τ - 1/2)²`; separating them makes it reachable.

**And the answer it gives is to leave `ωo` alone.** `test/test_boundary.jl` fits
the wall position out of a Poiseuille profile under this operator. At `ωo = 1`
the wall lands within 0.005 of where the geometry puts it at τ = 0.53, and the
error *shrinks* as τ → 1/2; forcing `Λ = 3/16` moves it further out at every τ
tried — four times as far at τ = 0.56, ten times at 0.53 — under both
bounce-back rules and with the wall on or off the midpoint. So 1.0 is at or near this operator's optimum, the wall condition gets
better at the production τ rather than worse, and the 7% the sphere validation's
τ ladder moves at Re = 100 is something else. That is what this argument exists
to have established; it is not a knob production is expected to turn.

The even orders keep `ωh` regardless, because that is what the stability rests
on: they are rebuilt from the *current* second-order moments through Isserlis'
theorem, and under-relaxing them leaves the fourth- and sixth-order content
lagging behind the strain rate it is supposed to follow. Driving both rates down
together diverges at Re = 100 inside two hundred steps, which is why the rate
had to be split before the question could be asked at all.

Odd cumulants relax towards zero, which is also their Gaussian target, so the
odd branch is plain under-relaxation and the two branches agree when
`ωo == ωh`. Both default to 1, which is the operator as it was before the rate
was split.
"""
@inline function relax_moments!(buf::AbstractVector{T}, ρ::T, ω::T, force::NTuple{3,T},
                                ωb::T = one(T), ωh::T = one(T),
                                ωo::T = ωh) where {T}
    κeq2 = ρ * T(CS2)
    invρ = one(T) / ρ
    third = one(T) / 3
    @inbounds begin
        # Momentum: in the co-moving frame the body force is its only content.
        buf[1+1] = force[1] / 2
        buf[1+3] = force[2] / 2
        buf[1+9] = force[3] / 2

        # Second order carries the viscosity. The trace is the acoustic (bulk)
        # mode and is relaxed separately: only the deviatoric part sets the shear
        # viscosity, so the trace need not be driven at ω → 2, and damping it at
        # ωb ≈ 1 instead is what buys the stability margin at low τ.
        tr = buf[3] + buf[7] + buf[19]
        tr_new = (tr + ωb * (3 * κeq2 - tr)) * third
        tr_third = tr * third
        buf[3] = tr_new + (one(T) - ω) * (buf[3] - tr_third)
        buf[7] = tr_new + (one(T) - ω) * (buf[7] - tr_third)
        buf[19] = tr_new + (one(T) - ω) * (buf[19] - tr_third)
        buf[5] *= (one(T) - ω)                 # κ110, equilibrium 0
        buf[11] *= (one(T) - ω)                # κ101
        buf[13] *= (one(T) - ω)                # κ011

        # Covariance implied by the relaxed second moments, per unit mass.
        A, B, C = buf[3] * invρ, buf[7] * invρ, buf[19] * invρ
        D, E, F = buf[5] * invρ, buf[11] * invρ, buf[13] * invρ

        for o in 0:2, n in 0:2, m in 0:2
            ord = m + n + o
            ord >= 3 || continue
            idx = 1 + m + 3n + 9o
            if isodd(ord)
                buf[idx] *= (one(T) - ωo)
            else
                target = ρ * gaussian_moment(m, n, o, A, B, C, D, E, F)
                buf[idx] += ωh * (target - buf[idx])
            end
        end
    end
end

"""
    collide_central_moments!(s; force = nothing, solid = nothing, les = nothing,
                             omega_bulk = 1.0, omega_higher = 1.0,
                             omega_odd = omega_higher)

Central-moment collision over the whole domain. Requires a D3Q27 state.
`omega_bulk` relaxes the acoustic mode, `omega_higher` the even cumulants above
second order and `omega_odd` the odd ones; all default to full damping, which is
what the stability rests on. `omega_odd` is the wall's `ω⁻` — see
[`relax_moments!`](@ref) for what lowering it buys and costs.
"""
function collide_central_moments!(s::LBMState{T}; force = nothing, solid = nothing,
                                  les = nothing, omega_bulk = 1.0,
                                  omega_higher = 1.0,
                                  omega_odd = omega_higher) where {T}
    s.lattice isa D3Q27 ||
        throw(ArgumentError("the central-moment operator needs D3Q27, got $(s.lattice)"))
    f = s.f
    ω = one(T) / s.τ
    F = force === nothing ? (zero(T), zero(T), zero(T)) : T.(force)
    buf = Vector{T}(undef, 27)

    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        if solid !== nothing && solid[i, j, k]
            continue
        end

        ρ = zero(T)
        mx = zero(T)
        my = zero(T)
        mz = zero(T)
        for q in 1:27
            fq = f[i, j, k, q]
            buf[CUBE27[q]] = fq
            ρ += fq
            mx += T(CX27[q]) * fq
            my += T(CY27[q]) * fq
            mz += T(CZ27[q]) * fq
        end
        ux = (mx + F[1] / 2) / ρ
        uy = (my + F[2] / 2) / ρ
        uz = (mz + F[3] / 2) / ρ

        ωnode = ω
        if les !== nothing
            Πnorm = nonequilibrium_flux_norm(s, i, j, k, ρ, ux, uy, uz)
            ωnode = one(T) / total_relaxation_time(les, s.τ, ρ, Πnorm)
        end

        to_moments!(buf, ux, uy, uz)
        relax_moments!(buf, ρ, ωnode, F, T(omega_bulk), T(omega_higher), T(omega_odd))
        to_populations!(buf, ux, uy, uz)

        for q in 1:27
            f[i, j, k, q] = buf[CUBE27[q]]
        end
    end
    return s
end
