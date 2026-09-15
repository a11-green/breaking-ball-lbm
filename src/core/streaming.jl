"""
Streaming with periodic wrap-around in all three directions.

Uses a second buffer (`fnew`) and swaps; the in-place AA-pattern that halves the
memory traffic is a GPU-side concern and lands with the CUDA port (P2).
"""

"""
    stream!(s)

Propagate `f[x, y, z, q]` to `x + c_q`, then swap buffers.
"""
function stream!(s::LBMState{T}) where {T}
    f = s.f
    fnew = s.fnew
    nx, ny, nz = s.nx, s.ny, s.nz
    @inbounds for q in 1:Q19
        cx, cy, cz = CX19[q], CY19[q], CZ19[q]
        for k in 1:nz
            kd = mod1(k + cz, nz)
            for j in 1:ny
                jd = mod1(j + cy, ny)
                for i in 1:nx
                    fnew[mod1(i + cx, nx), jd, kd, q] = f[i, j, k, q]
                end
            end
        end
    end
    s.f, s.fnew = s.fnew, s.f
    return s
end

"""
    step!(s)

One full LBM time step: collide, then stream.
"""
function step!(s::LBMState)
    collide!(s)
    stream!(s)
    return s
end

"""
    run!(s, nsteps; callback = nothing)

Advance `nsteps` time steps. `callback(s, step)` runs after each step when given.
"""
function run!(s::LBMState, nsteps::Integer; callback = nothing)
    for n in 1:nsteps
        step!(s)
        callback === nothing || callback(s, n)
    end
    return s
end
