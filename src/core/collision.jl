"""
BGK collision (single relaxation time), optionally with a uniform body force.

BGK is the reference operator: simple enough to verify against analytic
solutions, but it loses stability as τ approaches 1/2, which is where the
production Reynolds numbers sit. The central-moment operator in
`core/central_moments.jl` is the one those runs use.

The body force uses Guo's forcing scheme: the velocity carries a half-force
correction and a source term is added after relaxation. The non-inertial frame
of the coupled trajectory solver needs exactly this machinery, which is why it
lives in the reference implementation rather than waiting for the GPU port.
"""

"""
    collide!(s; force = nothing, solid = nothing, les = nothing, operator = :bgk)

Relax every node towards local equilibrium, in place. `force` is a uniform body
force per unit volume in lattice units; `solid` is a mask of nodes to leave
untouched (their populations are replaced by the boundary treatment); `les` is a
subgrid model that raises the local relaxation time. `operator` selects `:bgk` or
`:central_moment`.
"""
function collide!(s::LBMState{T}; force = nothing, solid = nothing, les = nothing,
                  operator::Symbol = :bgk, kwargs...) where {T}
    if operator === :central_moment
        return collide_central_moments!(s; force = force, solid = solid, les = les, kwargs...)
    end
    isempty(kwargs) || throw(ArgumentError("unexpected keyword arguments $(keys(kwargs))"))
    operator === :bgk || throw(ArgumentError("unknown operator $operator"))

    lat = s.lattice
    f = s.f
    cx, cy, cz, w = cxs(lat), cys(lat), czs(lat), weights(lat)
    nq = nvelocities(lat)
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
        for q in 1:nq
            fq = f[i, j, k, q]
            ρ += fq
            mx += T(cx[q]) * fq
            my += T(cy[q]) * fq
            mz += T(cz[q]) * fq
        end
        # Half-force correction: u is the velocity the fluid actually transports.
        ux = (mx + Fx / 2) / ρ
        uy = (my + Fy / 2) / ρ
        uz = (mz + Fz / 2) / ρ

        # Subgrid viscosity raises the relaxation time; τ is otherwise constant.
        ωnode = ω
        if les !== nothing
            Πnorm = nonequilibrium_flux_norm(s, i, j, k, ρ, ux, uy, uz)
            ωnode = one(T) / total_relaxation_time(les, s.τ, ρ, Πnorm)
        end
        fpre = force === nothing ? zero(T) : (one(T) - ωnode / 2)

        for q in 1:nq
            cqx, cqy, cqz = T(cx[q]), T(cy[q]), T(cz[q])
            fq = f[i, j, k, q]
            fq -= ωnode * (fq - equilibrium(lat, q, ρ, ux, uy, uz))
            if force !== nothing
                cu = cqx * ux + cqy * uy + cqz * uz
                sx = (cqx - ux) / T(CS2) + cu * cqx / T(CS2)^2
                sy = (cqy - uy) / T(CS2) + cu * cqy / T(CS2)^2
                sz = (cqz - uz) / T(CS2) + cu * cqz / T(CS2)^2
                fq += fpre * T(w[q]) * (sx * Fx + sy * Fy + sz * Fz)
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
