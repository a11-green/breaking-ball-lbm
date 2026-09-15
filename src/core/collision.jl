"""
BGK collision (single relaxation time), optionally with a uniform body force.

MRT and cumulant operators are planned for the turbulent production runs;
BGK is the reference operator used for verification against analytic solutions.

The body force uses Guo's forcing scheme: the velocity carries a half-force
correction and a source term is added after relaxation. The non-inertial frame
of the coupled trajectory solver needs exactly this machinery, which is why it
lives in the reference implementation rather than waiting for the GPU port.
"""

"""
    collide!(s; force = nothing, solid = nothing, les = nothing)

Relax every node towards local equilibrium, in place. `force` is a uniform body
force per unit volume in lattice units; `solid` is a mask of nodes to leave
untouched (their populations are replaced by the boundary treatment); `les` is a
subgrid model that raises the local relaxation time.
"""
function collide!(s::LBMState{T}; force = nothing, solid = nothing, les = nothing) where {T}
    f = s.f
    ω = one(T) / s.τ
    Fx, Fy, Fz = force === nothing ? (zero(T), zero(T), zero(T)) : T.(force)

    @inbounds for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
        if solid !== nothing && solid[i, j, k]
            continue
        end

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
        # Half-force correction: u is the velocity the fluid actually transports.
        ux = (mx + Fx / 2) / ρ
        uy = (my + Fy / 2) / ρ
        uz = (mz + Fz / 2) / ρ

        # Subgrid viscosity raises the relaxation time; τ is otherwise constant.
        ωnode = ω
        if les !== nothing
            Πnorm = nonequilibrium_flux_norm(f, i, j, k, ρ, ux, uy, uz)
            ωnode = one(T) / total_relaxation_time(les, s.τ, ρ, Πnorm)
        end
        fpre_node = force === nothing ? zero(T) : (one(T) - ωnode / 2)

        for q in 1:Q19
            cx, cy, cz = T(CX19[q]), T(CY19[q]), T(CZ19[q])
            fq = f[i, j, k, q]
            fq -= ωnode * (fq - equilibrium(q, ρ, ux, uy, uz))
            if force !== nothing
                cu = cx * ux + cy * uy + cz * uz
                sx = (cx - ux) / T(CS2) + cu * cx / T(CS2)^2
                sy = (cy - uy) / T(CS2) + cu * cy / T(CS2)^2
                sz = (cz - uz) / T(CS2) + cu * cz / T(CS2)^2
                fq += fpre_node * T(W19[q]) * (sx * Fx + sy * Fy + sz * Fz)
            end
            f[i, j, k, q] = fq
        end
    end
    return s
end

"""
    fluid_velocity(s, i, j, k, force)

Macroscopic velocity including the half-force correction, i.e. the velocity that
matches the Navier-Stokes solution when a body force is present.
"""
@inline function fluid_velocity(s::LBMState{T}, i::Integer, j::Integer, k::Integer,
                                force::NTuple{3,T}) where {T}
    ρ, ux, uy, uz = macroscopic(s, i, j, k)
    return ρ, ux + force[1] / (2ρ), uy + force[2] / (2ρ), uz + force[3] / (2ρ)
end
