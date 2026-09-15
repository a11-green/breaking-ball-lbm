"""
BGK collision (single relaxation time).

MRT and cumulant operators are planned for the turbulent production runs (P3);
BGK is the reference operator used for verification against analytic solutions.
"""

"""
    collide!(s)

Relax every node towards local equilibrium, in place.
"""
function collide!(s::LBMState{T}) where {T}
    f = s.f
    ω = one(T) / s.τ
    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        ρ = zero(T)
        mx = zero(T)
        my = zero(T)
        mz = zero(T)
        for q in 1:Q19
            fq = f[i, j, k, q]
            ρ += fq
            mx += T(CX19[q]) * fq
            my += T(CY19[q]) * fq
            mz += T(CZ19[q]) * fq
        end
        ux = mx / ρ
        uy = my / ρ
        uz = mz / ρ
        for q in 1:Q19
            feq = equilibrium(q, ρ, ux, uy, uz)
            f[i, j, k, q] = f[i, j, k, q] - ω * (f[i, j, k, q] - feq)
        end
    end
    return s
end
