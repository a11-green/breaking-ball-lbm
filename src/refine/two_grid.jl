"""
Two-level grid refinement, with the fine patch a centred cube inside the coarse one.

§6.5 forced this: on a uniform grid, holding an eight-diameter domain caps the
seam ridge at 0.42 of a cell, and reaching a whole cell means shrinking the box
until the ball fills a sixth of it — a periodic array of baseballs, not one.
Refinement is the only arrangement that buys both.

**Acoustic scaling.** `dt` halves with `dx`, so the lattice speed `u_lattice` is
the same on both levels and the physical speed of sound `dx/(dt√3)` is too. The
cheaper alternative — one time step everywhere — would make the sound speed
differ by two between the levels, and a moving boundary produces acoustic noise
continuously, which would then partially reflect at the interface and stay
trapped in the fine patch. Paying for temporal interpolation is what buys an
interface that sound passes through.

Under that scaling, with a refinement ratio `m`:

    ν_lattice doubles      ⇒  τ_f − 1/2 = m (τ_c − 1/2)
    body acceleration      ⇒  a_f = a_c / m        (a ∝ dt²/dx)
    strain rate            ⇒  S_f = S_c / m        (S ∝ dt)

**The rescaling of the non-equilibrium part, which is the thing to get right.**
Populations do not transfer unchanged. `f^eq` depends only on `ρ` and `u`, which
acoustic scaling leaves alone, so it carries across untouched; `f^neq` carries
the strain rate and does not. From Chapman-Enskog, `f^neq ∝ τ_lattice · S_lattice`,
so

    α ≡ f^neq_fine / f^neq_coarse = (τ_f / τ_c) · (1/m)

which equals `τ_f (τ_c − 1/2) / (τ_c (τ_f − 1/2))`, the form usually quoted.

**α is not a constant, and that is the trap.** It runs from `1/m` as τ → 1/2 to
`1` as τ grows, because `τ_f/τ_c` runs from 1 to m over that range. A
τ-independent formula — or the τ−1/2 prefactor written where τ belongs — is
wrong by a factor approaching m at exactly the relaxation times a pitch
Reynolds number forces, while at τ ≈ 1 the same mistake is only tens of percent
and easy to mistake for discretisation error. `test/test_refine.jl` pins the
value with a linear shear, whose `f^neq` is constant and exactly known, at both
ends of that range.

**What it costs, measured.** A Taylor-Green vortex with the interface cut
straight through its core — the hardest placement there is — converges at order
1.98 then 2.42 under acoustic refinement of the whole problem, so the interface
is second order as designed and the spatial interpolation, not the boundary's
one-step staleness, is what limits it. Under diffusive scaling, where the number
of cycles grows as n² while each cycle's interface error falls as n⁻², the
observed order degrades to about 1.3: the interface error partially accumulates,
and over a long enough run it can dominate the bulk discretisation. In the
application the interface sits several diameters from the ball in nearly uniform
flow, where `(k Δx)²` is small; the test puts it where the gradient is largest on
purpose.

**Layout.** The fine patch covers coarse indices `lo:hi` inclusive and has
`2(hi−lo)+1` nodes a side, so odd fine indices sit on coarse nodes and even ones
halfway between. Coarse-to-fine transfer therefore needs no general
interpolation, only an average over the one, two, four or eight coarse nodes a
fine node falls between.

**Timing.** A cycle is two coarse steps and four fine ones, which is what returns
both levels to AA-pattern's even layout together. The fine patch's outer two
node layers are imposed from the coarse solution after every second fine step —
every second, not every step, because in the odd layout a node's populations are
scattered across its neighbours and writing them means writing someone else's
memory. Two layers is the minimum that keeps the wrap of the fine array — `aa_run!` sees
it as periodic — from reaching the interior: after two steps the contamination
has moved two nodes and no further. Three is the default because it measured
better, the extra layer buffering the skipped step's staleness as well.
"""

"""
    TwoGrid(coarse_dims, lo, hi, τ_coarse; ratio = 2)

A coarse grid of `coarse_dims` with a refined cube covering coarse indices
`lo:hi`. Populations are in cube-slot order on both levels, as everything else
on the device path is.
"""
struct TwoGrid{T<:AbstractFloat}
    coarse::Array{T,4}
    fine::Array{T,4}
    τc::T
    τf::T
    lo::NTuple{3,Int}
    hi::NTuple{3,Int}
    ratio::Int
    α::T                    # f^neq, coarse → fine
    prev::Array{T,4}        # the covered coarse box at the start of a cycle
end

"""τ on the fine level: ν in lattice units grows with the refinement ratio."""
fine_tau(τc::Real, ratio::Integer = 2) = ratio * (τc - 1 / 2) + 1 / 2

"""
    neq_rescale(τc, τf, ratio)

`f^neq_fine / f^neq_coarse`. See the module docstring; the two forms below are
algebraically the same and both are computed so a regression cannot quietly
change one.
"""
neq_rescale(τc::T, τf::T, ratio::Integer) where {T} = (τf / τc) / ratio

function TwoGrid(::Type{T}, coarse_dims::NTuple{3,<:Integer},
                 lo::NTuple{3,<:Integer}, hi::NTuple{3,<:Integer}, τc::Real;
                 ratio::Integer = 2) where {T<:AbstractFloat}
    ratio == 2 || throw(ArgumentError("only a refinement ratio of 2 is implemented, got $ratio"))
    all(hi .> lo) || throw(ArgumentError("the fine patch must have positive extent"))
    all(lo .>= 2) && all(hi .<= coarse_dims .- 1) ||
        throw(ArgumentError("the fine patch must leave a coarse node on every side: " *
                            "lo = $lo, hi = $hi, dims = $coarse_dims"))

    fine_dims = ntuple(d -> ratio * (hi[d] - lo[d]) + 1, 3)
    τf = T(fine_tau(τc, ratio))
    return TwoGrid{T}(zeros(T, coarse_dims..., 27), zeros(T, fine_dims..., 27),
                      T(τc), τf, Int.(lo), Int.(hi), Int(ratio),
                      T(neq_rescale(T(τc), τf, ratio)),
                      zeros(T, (hi .- lo .+ 1)..., 27))
end

"""Node count on each level, and how much of the coarse grid the patch covers."""
function grid_sizes(rg::TwoGrid)
    nc = prod(size(rg.coarse)[1:3])
    nf = prod(size(rg.fine)[1:3])
    return (coarse = nc, fine = nf, covered = prod(rg.hi .- rg.lo .+ 1),
            fraction = prod(rg.hi .- rg.lo .+ 1) / nc)
end

"""Body acceleration on the fine level, given the coarse one: `a ∝ dt²/dx`."""
fine_force(force::NTuple{3,T}, ratio::Integer = 2) where {T} = force ./ T(ratio)

# --- node-level helpers ----------------------------------------------------

"""Density and velocity from a node's own 27 populations (even layout)."""
@inline function node_macroscopic(g, i::Int, j::Int, k::Int, ::Type{T}) where {T}
    ρ = zero(T); mx = zero(T); my = zero(T); mz = zero(T)
    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds f_s = g[i, j, k, s]
        cv_s = cube_velocity(s)
        ρ += f_s
        mx += T(cv_s[1]) * f_s
        my += T(cv_s[2]) * f_s
        mz += T(cv_s[3]) * f_s
    end
    inv = one(T) / ρ
    return ρ, mx * inv, my * inv, mz * inv
end

"""Non-equilibrium part into `buf`, returning the macroscopic state."""
@inline function node_nonequilibrium!(buf, g, i::Int, j::Int, k::Int, ::Type{T}) where {T}
    ρ, ux, uy, uz = node_macroscopic(g, i, j, k, T)
    Base.Cartesian.@nexprs 27 s -> begin
        @inbounds buf[s] = g[i, j, k, s] - cube_equilibrium(s, ρ, ux, uy, uz)
    end
    return ρ, ux, uy, uz
end

"""
The coarse nodes a fine index falls between: one when it lands on a coarse node,
two when it lands halfway.
"""
@inline function coarse_span(lo::Int, a::Int)
    h = a - 1
    c = lo + h ÷ 2
    return isodd(h) ? (c, c + 1) : (c, c)
end

covered_offset(rg::TwoGrid, i::Int, j::Int, k::Int) =
    (i - rg.lo[1] + 1, j - rg.lo[2] + 1, k - rg.lo[3] + 1)

# --- transfers -------------------------------------------------------------

"""
    interface_fill!(rg, frac; layers = 3, scratch)

Impose the fine patch's outer `layers` node shells from the coarse solution,
taken at time fraction `frac` through the cycle.

`ρ`, `u` and `f^neq` are interpolated — not the populations. The equilibrium is
quadratic in `u`, so averaging populations and averaging the state it came from
are different things, and only the second keeps the interpolated node on the
equilibrium manifold of its own macroscopic state. It is also the only form in
which the `f^neq` rescaling has anywhere to go.
"""
function interface_fill!(rg::TwoGrid{T}, frac::Real; layers::Integer = 3,
                         scratch = nothing) where {T}
    nfx, nfy, nfz = size(rg.fine, 1), size(rg.fine, 2), size(rg.fine, 3)
    L = Int(layers)
    θ = T(frac)
    neq_a = scratch === nothing ? Vector{T}(undef, 27) : scratch[1]
    neq_b = scratch === nothing ? Vector{T}(undef, 27) : scratch[2]
    acc = scratch === nothing ? Vector{T}(undef, 27) : scratch[3]

    @inbounds for c in 1:nfz, b in 1:nfy, a in 1:nfx
        edge = a <= L || a > nfx - L || b <= L || b > nfy - L || c <= L || c > nfz - L
        edge || continue

        i0, i1 = coarse_span(rg.lo[1], a)
        j0, j1 = coarse_span(rg.lo[2], b)
        k0, k1 = coarse_span(rg.lo[3], c)

        ρ = zero(T); ux = zero(T); uy = zero(T); uz = zero(T)
        fill!(acc, zero(T))
        n = 0
        for kk in k0:k1, jj in j0:j1, ii in i0:i1
            o = covered_offset(rg, ii, jj, kk)
            ρa, xa, ya, za = node_nonequilibrium!(neq_a, rg.prev, o[1], o[2], o[3], T)
            ρb, xb, yb, zb = node_nonequilibrium!(neq_b, rg.coarse, ii, jj, kk, T)
            ρ += ρa + θ * (ρb - ρa)
            ux += xa + θ * (xb - xa)
            uy += ya + θ * (yb - ya)
            uz += za + θ * (zb - za)
            for s in 1:27
                acc[s] += neq_a[s] + θ * (neq_b[s] - neq_a[s])
            end
            n += 1
        end
        w = one(T) / n
        ρ *= w; ux *= w; uy *= w; uz *= w

        for s in 1:27
            rg.fine[a, b, c, s] = cube_equilibrium(s, ρ, ux, uy, uz) + rg.α * acc[s] * w
        end
    end
    return rg
end

"""
    restrict!(rg; filtered = true)

Overwrite the coarse nodes the patch covers with the fine solution.

`filtered` low-passes the fine state with the lattice weights before handing it
over, on the argument that a coarse node cannot represent what the fine grid
resolves and copying it straight across aliases that content rather than
discarding it.

**It is off by default, because measurement disagreed with the argument.** The
27-point weighted mean is far too broad: on a Taylor-Green vortex it made the
error worse by a factor of 1.7 with two imposed layers and 3.9 with three, by
damping the resolved field along with the unresolved. It is exact on what the
coarse grid can carry — a symmetric mean passes constants and linear gradients —
but a vortex at sixty-four fine cells per wavelength loses about a third of a
percent per application, and it is applied every cycle. Kept as an option
because filtering of *some* kind is the usual remedy for aliasing at high
Reynolds number; a narrower stencil, or filtering `f^neq` alone, would be the
thing to try before switching this on.
"""
function restrict!(rg::TwoGrid{T}; filtered::Bool = false, scratch = nothing) where {T}
    nfx, nfy, nfz = size(rg.fine, 1), size(rg.fine, 2), size(rg.fine, 3)
    neq = scratch === nothing ? Vector{T}(undef, 27) : scratch[1]
    acc = scratch === nothing ? Vector{T}(undef, 27) : scratch[2]
    invα = one(T) / rg.α

    @inbounds for k in (rg.lo[3]+1):(rg.hi[3]-1),
                  j in (rg.lo[2]+1):(rg.hi[2]-1),
                  i in (rg.lo[1]+1):(rg.hi[1]-1)
        a = rg.ratio * (i - rg.lo[1]) + 1
        b = rg.ratio * (j - rg.lo[2]) + 1
        c = rg.ratio * (k - rg.lo[3]) + 1

        if filtered && 1 < a < nfx && 1 < b < nfy && 1 < c < nfz
            ρ = zero(T); ux = zero(T); uy = zero(T); uz = zero(T)
            fill!(acc, zero(T))
            for s in 1:27
                cx, cy, cz = cube_velocity(s)
                w = cube_weight(T, s)
                ρn, xn, yn, zn = node_nonequilibrium!(neq, rg.fine, a + cx, b + cy, c + cz, T)
                ρ += w * ρn; ux += w * xn; uy += w * yn; uz += w * zn
                for t in 1:27
                    acc[t] += w * neq[t]
                end
            end
        else
            ρ, ux, uy, uz = node_nonequilibrium!(acc, rg.fine, a, b, c, T)
        end

        for s in 1:27
            rg.coarse[i, j, k, s] = cube_equilibrium(s, ρ, ux, uy, uz) + invα * acc[s]
        end
    end
    return rg
end

"""Snapshot the covered coarse box, so the interface can interpolate in time."""
function save_coarse!(rg::TwoGrid)
    @inbounds for s in 1:27,
                  k in rg.lo[3]:rg.hi[3], j in rg.lo[2]:rg.hi[2], i in rg.lo[1]:rg.hi[1]
        o = covered_offset(rg, i, j, k)
        rg.prev[o[1], o[2], o[3], s] = rg.coarse[i, j, k, s]
    end
    return rg
end

"""
    refine_cycle!(rg; force, operator, layers, filtered, ...)

Two coarse steps and four fine ones — the unit that returns both levels to the
even layout together.

The coarse level is advanced first because the fine level's boundary needs the
coarse state at the end of the cycle to interpolate towards. That makes the
coupling explicit in time, which is what every refined LBM does; the alternative
is iterating the two levels to convergence within a cycle, for an error already
below the scheme's own.
"""
function refine_cycle!(rg::TwoGrid{T}; force::NTuple{3,<:Real} = (0, 0, 0),
                       operator::Symbol = :central_moment,
                       layers::Integer = 3, filtered::Bool = false,
                       omega_bulk::Real = 1.0, omega_higher::Real = 1.0,
                       scratch = ntuple(_ -> Vector{T}(undef, 27), 3)) where {T}
    Fc = T.(force)
    Ff = fine_force(Fc, rg.ratio)

    save_coarse!(rg)
    aa_run!(rg.coarse, 2, rg.τc; force = Fc, operator = operator,
            omega_bulk = omega_bulk, omega_higher = omega_higher)

    for half in 1:2
        aa_run!(rg.fine, 2, rg.τf; force = Ff, operator = operator,
                omega_bulk = omega_bulk, omega_higher = omega_higher)
        interface_fill!(rg, half / 2; layers = layers, scratch = scratch)
    end

    restrict!(rg; filtered = filtered, scratch = scratch)
    return rg
end

"""
    init_refined!(rg, field)

Fill both levels from `field(x, y, z) -> (ρ, ux, uy, uz)`, with positions in
coarse-index units so the two levels describe the same flow.
"""
function init_refined!(rg::TwoGrid{T}, field) where {T}
    ncx, ncy, ncz = size(rg.coarse, 1), size(rg.coarse, 2), size(rg.coarse, 3)
    @inbounds for k in 1:ncz, j in 1:ncy, i in 1:ncx
        ρ, ux, uy, uz = T.(field(T(i), T(j), T(k)))
        for s in 1:27
            rg.coarse[i, j, k, s] = cube_equilibrium(s, ρ, ux, uy, uz)
        end
    end
    nfx, nfy, nfz = size(rg.fine, 1), size(rg.fine, 2), size(rg.fine, 3)
    h = one(T) / rg.ratio
    @inbounds for c in 1:nfz, b in 1:nfy, a in 1:nfx
        x = T(rg.lo[1]) + (a - 1) * h
        y = T(rg.lo[2]) + (b - 1) * h
        z = T(rg.lo[3]) + (c - 1) * h
        ρ, ux, uy, uz = T.(field(x, y, z))
        for s in 1:27
            rg.fine[a, b, c, s] = cube_equilibrium(s, ρ, ux, uy, uz)
        end
    end
    return rg
end

"""Macroscopic fields of one level, for comparing against an analytic solution."""
function level_macroscopic(g::Array{T,4}) where {T}
    nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
    ρ = Array{T,3}(undef, nx, ny, nz)
    u = Array{T,4}(undef, nx, ny, nz, 3)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        r, ux, uy, uz = node_macroscopic(g, i, j, k, T)
        ρ[i, j, k] = r
        u[i, j, k, 1] = ux; u[i, j, k, 2] = uy; u[i, j, k, 3] = uz
    end
    return ρ, u
end
