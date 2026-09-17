"""
Inflow and outflow on the streamwise faces, replacing the periodic wrap there.

**Why this exists.** A fully periodic box has no way to say what the free stream
is, so §4.4 held it with a controller on the box mean. V&V-2 measured what that
costs (§8): the box mean is fixed by the mass flux through any plane and is
therefore blind to the wake, while the body sits on the axis in the retarded
part of the profile — so the controller holds a number that is not the velocity
the body sees, and the drag coefficient's denominator becomes ambiguous by
enough to swamp what is being measured. Worse, the wake leaves one face and
arrives at the other: the ball flies through its own turbulence, which is the
one thing a seam simulation must not get wrong, since the seam works by moving
the separation point and the separation point is set by the state of the
approaching boundary layer.

An inlet states the free stream instead of inferring it, and an outlet lets the
wake leave. The controller is then unnecessary — and in the ball-following frame
the identity `d(u∞)/dt = a_fluid` (§4.4) means the interior accelerates at
exactly the rate the inlet value changes, so the two stay consistent with no
feedback at all.

**What it does not fix.** The lateral faces stay periodic, so the transverse
blockage — the sphere still has periodic images to the sides — is untouched and
has to be bought with domain width. The outlet is not perfectly non-reflecting
either: LBM is weakly compressible, and an acoustic wave reaching the outlet
leaves a residue behind.

**The geometry.** The stream runs along −x throughout this project, so fluid
enters at the high-x face and leaves at the low-x face.

**Why a buffer of two planes.** AA-pattern alternates two access patterns, and a
node's populations only sit in its own slots on even steps, so the condition can
only be imposed there — once per pair of steps. In two steps a population
travels two cells, so the wrap carries the outlet's state two planes into the
inlet end before the next chance to overwrite it. A buffer of two planes at each
end is exactly deep enough: nothing wrapped ever reaches the third plane, which
is the one the condition reads from.
"""
struct OpenChannel{T<:AbstractFloat}
    depth::Int
    ρ_ref::T
end

"""
    OpenChannel{T}(; depth = 2, density = 1)

`depth` is the buffer thickness in planes at each end, which must be at least
two for the reason in the type's docstring; a deeper one is allowed and costs
almost nothing. `density` is the reference the outlet anchors the pressure to.
"""
function OpenChannel{T}(; depth::Integer = 2, density::Real = 1) where {T<:AbstractFloat}
    depth >= 2 || throw(ArgumentError("depth must be at least 2, got $depth"))
    density > 0 || throw(ArgumentError("reference density must be positive, got $density"))
    return OpenChannel{T}(Int(depth), T(density))
end

OpenChannel(; kwargs...) = OpenChannel{Float64}(; kwargs...)

"""
    open_node!(g, itarget, isource, j, k, T, ρ_ref, u_in, impose_velocity)

One buffer node, from the first interior plane. Shared by the host driver and
the CUDA kernel, which is what keeps the two from drifting apart.

This is Guo's non-equilibrium extrapolation, used at both ends with a different
half imposed:

  - the **inlet** imposes the velocity and takes the density from the interior,
    so the stream is stated and a pressure wave can still pass out through it;
  - the **outlet** imposes the density and takes the velocity from the interior,
    so the wake convects out and the pressure has somewhere to be anchored.

Without that anchor nothing pins the absolute density and the box drifts: an
inlet that also sets the density over-determines the problem and reflects
everything instead.

The source is a single interior plane for every target plane in the buffer, so
no node ever reads one that another writes and the pass is race-free by
construction rather than by ordering.
"""
@inline function open_node!(g, itarget::Int, isource::Int, j::Int, k::Int,
                            ::Type{T}, ρ_ref::T, u_in::NTuple{3,T},
                            impose_velocity::Bool) where {T}
    ρn, ux, uy, uz = node_macroscopic(g, isource, j, k, T)
    ρb = impose_velocity ? ρn : ρ_ref
    bx = impose_velocity ? u_in[1] : ux
    by = impose_velocity ? u_in[2] : uy
    bz = impose_velocity ? u_in[3] : uz
    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds neq_s = g[isource, j, k, s] - cube_equilibrium(s, ρn, ux, uy, uz)
        @inbounds g[itarget, j, k, s] = cube_equilibrium(s, ρb, bx, by, bz) + neq_s
    end
    return nothing
end

"""The plane the inlet buffer reads from, and the one the outlet reads from."""
inlet_source(ch::OpenChannel, nx::Integer) = nx - ch.depth
outlet_source(ch::OpenChannel, ::Integer) = 1 + ch.depth

"""
    open_is_clear(wall, ch)

Is the geometry far enough from both faces for the condition to mean anything?

The buffer planes and the two planes they read from are stated to be uniform
free stream and undisturbed outflow respectively, so a solid node in any of them
is a body sitting in a boundary condition. Checked rather than assumed, in the
manner of `shell_is_sufficient` (§4.1.1).
"""
function open_is_clear(wall::WallField, ch::OpenChannel)
    nx = size(wall.kind, 1)
    for i in vcat(collect(1:(ch.depth + 1)), collect((nx - ch.depth):nx))
        any(@view(wall.kind[i, :, :]) .== SOLID_NODE) && return false
    end
    return true
end

"""
    apply_open!(g, ch, u_in)

Impose the inlet and outlet buffers. Valid only in the even-step layout, which
is where a driver leaves the array after an even number of steps.

`u_in` is the free stream, which the coupled loop changes every sub-cycle as the
ball slows: the ball-following frame's stream is `−V_ball` in lattice units, so
it points along −x and shrinks over the flight.
"""
function apply_open!(g::Array{T,4}, ch::OpenChannel{T},
                     u_in::NTuple{3,<:Real}) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    nx >= 2 * ch.depth + 2 ||
        throw(ArgumentError("a box of $nx planes cannot hold two buffers of $(ch.depth)"))
    u_in[1] <= 0 ||
        throw(ArgumentError("the stream runs along -x, so u_in[1] must not be positive; got $(u_in[1])"))
    u = T.(u_in)
    isrc = inlet_source(ch, nx)
    osrc = outlet_source(ch, nx)
    @inbounds for k in 1:nz, j in 1:ny
        for m in 0:(ch.depth - 1)
            open_node!(g, nx - m, isrc, j, k, T, ch.ρ_ref, u, true)
            open_node!(g, 1 + m, osrc, j, k, T, ch.ρ_ref, u, false)
        end
    end
    return g
end
