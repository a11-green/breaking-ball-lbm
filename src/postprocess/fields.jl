"""
Fields worth looking at, derived from the populations.

Everything a viewer draws is computed here rather than in the viewer, so it can
be tested against analytic flows without a window, a GPU or a graphics stack.
`scripts/view_flow.jl` is then thin enough to be obviously correct by reading.

**Vorticity has to come from finite differences, and that is not an oversight.**
The non-equilibrium populations carry the strain rate — `f^neq` is proportional
to the *symmetric* part of the velocity gradient — and nothing else. Rotation is
the antisymmetric part, which collision leaves alone precisely because it does
not dissipate. So the node-local trick that makes the Smagorinsky model cheap
(§3.1) does not extend to vorticity: it needs neighbours.
"""

"""
    velocity_field(g; mask = nothing)

Velocity at every node of an AA-pattern array in the even layout, as
`(nx, ny, nz, 3)`. Nodes where `mask` is true are `NaN`, which is what makes a
solid body render as a hole rather than as a region of nonsense.
"""
function velocity_field(g::Array{T,4}; mask = nothing) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    u = Array{T,4}(undef, nx, ny, nz, 3)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if mask !== nothing && mask[i, j, k]
            u[i, j, k, 1] = u[i, j, k, 2] = u[i, j, k, 3] = T(NaN)
            continue
        end
        _, ux, uy, uz = node_macroscopic(g, i, j, k, T)
        u[i, j, k, 1] = ux
        u[i, j, k, 2] = uy
        u[i, j, k, 3] = uz
    end
    return u
end

"""
    density_field(g; mask = nothing)

Density at every node. Pressure is `ρ c_s²`, so this is the pressure field up to
a constant and a factor.
"""
function density_field(g::Array{T,4}; mask = nothing) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    ρ = Array{T,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if mask !== nothing && mask[i, j, k]
            ρ[i, j, k] = T(NaN)
            continue
        end
        ρ[i, j, k], = node_macroscopic(g, i, j, k, T)
    end
    return ρ
end

"""Magnitude of a `(nx, ny, nz, 3)` vector field."""
function magnitude(u::Array{T,4}) where {T}
    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    m = Array{T,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        m[i, j, k] = sqrt(u[i, j, k, 1]^2 + u[i, j, k, 2]^2 + u[i, j, k, 3]^2)
    end
    return m
end

"""
    velocity_gradient(u, i, j, k)

`∂u_β/∂x_α` at one node by central differences, wrapping periodically.

Second-order accurate and one spacing wide, which matters near a wall: a node
next to the solid differences across it, so the gradient there is wrong and the
viewer masks the body rather than pretending otherwise.
"""
@inline function velocity_gradient(u::Array{T,4}, i::Int, j::Int, k::Int) where {T}
    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    ip, im = shift_periodic(i, 1, nx), shift_periodic(i, -1, nx)
    jp, jm = shift_periodic(j, 1, ny), shift_periodic(j, -1, ny)
    kp, km = shift_periodic(k, 1, nz), shift_periodic(k, -1, nz)
    half = T(0.5)
    return (
        (half * (u[ip, j, k, 1] - u[im, j, k, 1]),
         half * (u[ip, j, k, 2] - u[im, j, k, 2]),
         half * (u[ip, j, k, 3] - u[im, j, k, 3])),
        (half * (u[i, jp, k, 1] - u[i, jm, k, 1]),
         half * (u[i, jp, k, 2] - u[i, jm, k, 2]),
         half * (u[i, jp, k, 3] - u[i, jm, k, 3])),
        (half * (u[i, j, kp, 1] - u[i, j, km, 1]),
         half * (u[i, j, kp, 2] - u[i, j, km, 2]),
         half * (u[i, j, kp, 3] - u[i, j, km, 3])))
end

"""
    vorticity(u)

`∇ × u`, as `(nx, ny, nz, 3)`. `NaN` propagates from a masked velocity field, so
the body and the layer that differences across it both come out blank.
"""
function vorticity(u::Array{T,4}) where {T}
    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    ω = Array{T,4}(undef, nx, ny, nz, 3)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        dx, dy, dz = velocity_gradient(u, i, j, k)
        ω[i, j, k, 1] = dy[3] - dz[2]
        ω[i, j, k, 2] = dz[1] - dx[3]
        ω[i, j, k, 3] = dx[2] - dy[1]
    end
    return ω
end

"""
    q_criterion(u)

`Q = ½(‖Ω‖² − ‖S‖²)`, positive where rotation beats strain.

The standard way to see a vortex without picking a threshold on vorticity, which
cannot tell a vortex from a shear layer — and a boundary layer is a shear layer,
so on a ball the vorticity plot is mostly the surface and the Q plot is mostly
the wake.
"""
function q_criterion(u::Array{T,4}) where {T}
    nx, ny, nz = size(u, 1), size(u, 2), size(u, 3)
    q = Array{T,3}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        g = velocity_gradient(u, i, j, k)
        s = zero(T)
        r = zero(T)
        for α in 1:3, β in 1:3
            sym = T(0.5) * (g[α][β] + g[β][α])
            asym = T(0.5) * (g[α][β] - g[β][α])
            s += sym * sym
            r += asym * asym
        end
        q[i, j, k] = T(0.5) * (r - s)
    end
    return q
end

"""
    field_limits(a; quantile = 0.99)

Symmetric colour limits that ignore the tail, so one cell next to the wall does
not set the scale for the whole picture. `NaN`s are skipped.
"""
function field_limits(a::AbstractArray{T}; quantile::Real = 0.99) where {T}
    v = sort!([abs(x) for x in a if isfinite(x)])
    isempty(v) && return (-one(T), one(T))
    hi = v[clamp(ceil(Int, quantile * length(v)), 1, length(v))]
    hi == 0 && (hi = one(T))
    return (-hi, hi)
end

"""
    slice_field(a, axis, index)

One plane of a 3-D array, as a 2-D array, for a viewer that draws heatmaps.
"""
function slice_field(a::AbstractArray{T,3}, axis::Integer, index::Integer) where {T}
    axis == 1 && return a[index, :, :]
    axis == 2 && return a[:, index, :]
    axis == 3 && return a[:, :, index]
    throw(ArgumentError("axis must be 1, 2 or 3, got $axis"))
end

slice_field(a::AbstractArray{T,4}, axis::Integer, index::Integer, component::Integer) where {T} =
    slice_field(view(a, :, :, :, component), axis, index)
