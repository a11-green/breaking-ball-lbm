"""
The lattice unit system, and the conversions the coupled loop needs.

LBM has no free time step: once Δx and the lattice velocity of the free stream
are chosen, Δt follows, and the relaxation time follows from that. Everything
here is a consequence of three run-time choices — how many nodes across the
ball, how fast the free stream is allowed to be in lattice units, and the
physical air properties — so resolution stays a setting rather than something
baked into the design (§6.5).

The conversions matter because the coupled loop lives in both worlds at once:
the momentum-exchange force comes out in lattice units, the 6DOF equations want
newtons, and the resulting frame acceleration has to go back as a lattice body
force. Getting one exponent wrong here would show up as a trajectory that is
plausible but wrong, which is the failure mode worth the most care.
"""

const AIR_DENSITY = 1.204        # kg/m³ at 20 °C, sea level
const AIR_VISCOSITY = 1.516e-5   # m²/s, kinematic
const GRAVITY = (0.0, 0.0, -9.80665)

"""
    LatticeUnits(; diameter, nodes_per_diameter, speed, lattice_speed, ν, ρ)

Δx, Δt and τ for a run, plus the diagnostics that say whether the choice is
usable.

`lattice_speed` is the free-stream speed in lattice units — the knob that trades
accuracy for cost. The lattice Mach number is `lattice_speed / c_s`, and the
compressibility error of LBM is O(Ma²), so 0.05 (Ma ≈ 0.087) is a reasonable
default and anything above about 0.1 starts to show.
"""
struct LatticeUnits{T<:AbstractFloat}
    dx::T              # m per lattice spacing
    dt::T              # s per lattice step
    τ::T               # molecular relaxation time
    ν_lattice::T
    ρ_physical::T      # kg/m³, the density that lattice ρ = 1 stands for
    nodes_per_diameter::Int
    lattice_speed::T
end

function LatticeUnits(::Type{T} = Float64; diameter::Real = 0.0748,
                      nodes_per_diameter::Integer = 40, speed::Real = 39.0,
                      lattice_speed::Real = 0.05, ν::Real = AIR_VISCOSITY,
                      ρ::Real = AIR_DENSITY) where {T<:AbstractFloat}
    speed > 0 || throw(ArgumentError("speed must be positive, got $speed"))
    nodes_per_diameter > 0 ||
        throw(ArgumentError("nodes_per_diameter must be positive, got $nodes_per_diameter"))

    dx = T(diameter) / nodes_per_diameter
    dt = T(lattice_speed) * dx / T(speed)
    ν_lat = T(ν) * dt / dx^2
    τ = ν_lat / T(CS2) + T(0.5)
    return LatticeUnits{T}(dx, dt, τ, ν_lat, T(ρ), Int(nodes_per_diameter), T(lattice_speed))
end

"""Lattice Mach number of the free stream."""
mach_number(u::LatticeUnits) = u.lattice_speed / sqrt(CS2)

"""Reynolds number the lattice actually resolves — should equal the physical one."""
function lattice_reynolds(u::LatticeUnits)
    return u.lattice_speed * u.nodes_per_diameter / u.ν_lattice
end

# Physical → lattice.
to_lattice_length(u::LatticeUnits, L::Real) = L / u.dx
to_lattice_velocity(u::LatticeUnits, v::Real) = v * u.dt / u.dx
to_lattice_acceleration(u::LatticeUnits, a::Real) = a * u.dt^2 / u.dx
to_lattice_rate(u::LatticeUnits, ω::Real) = ω * u.dt          # rad per step
to_lattice_steps(u::LatticeUnits, t::Real) = t / u.dt

# Lattice → physical. ρ_lattice = 1 stands for ρ_physical, so force carries
# ρ L⁴/T² and torque ρ L⁵/T².
to_physical_length(u::LatticeUnits, L::Real) = L * u.dx
to_physical_velocity(u::LatticeUnits, v::Real) = v * u.dx / u.dt
to_physical_time(u::LatticeUnits, n::Real) = n * u.dt
to_physical_force(u::LatticeUnits, F::Real) = F * u.ρ_physical * u.dx^4 / u.dt^2
to_physical_torque(u::LatticeUnits, τq::Real) = τq * u.ρ_physical * u.dx^5 / u.dt^2

for f in (:to_lattice_length, :to_lattice_velocity, :to_lattice_acceleration,
          :to_lattice_rate, :to_physical_length, :to_physical_velocity,
          :to_physical_force, :to_physical_torque)
    @eval $f(u::LatticeUnits, v::NTuple{3,<:Real}) =
        ($f(u, v[1]), $f(u, v[2]), $f(u, v[3]))
end

"""
    resolution_report(units)

What the choice costs and whether it is safe, as a named tuple.

`tau_margin` is the one to watch. At pitch Reynolds numbers τ sits within a few
times 1e-5 of 1/2, where BGK is unconditionally unstable and the run depends
entirely on the central-moment operator and the subgrid viscosity (§2.3, §3.1).
That is the expected regime, not a mistake — but it is worth printing, because a
run that is accidentally *well* resolved is a run that is accidentally cheap and
wrong about which physics it is capturing.
"""
function resolution_report(u::LatticeUnits{T}) where {T}
    return (dx_mm = 1000 * u.dx, dt_µs = 1e6 * u.dt, tau = u.τ,
            tau_margin = u.τ - T(0.5), mach = mach_number(u),
            reynolds = lattice_reynolds(u),
            steps_per_second = 1 / u.dt)
end

"""
    boundary_layer_thickness(; diameter, speed, ν)

Laminar boundary-layer thickness at the sphere's equator, `5√(νx/U)` with `x`
the arc length from the stagnation point.

Worth having as a number rather than an intuition, because at pitching speeds it
comes out at 0.76 mm against a seam ridge of 0.79 mm. The seam is one boundary
layer tall — which is exactly the condition for a roughness element to trip
transition (§1.3), and also means that resolving the seam and resolving the
boundary layer are not two requirements but one.
"""
function boundary_layer_thickness(; diameter::Real = 0.0748, speed::Real = 39.0,
                                  ν::Real = AIR_VISCOSITY)
    x = π * diameter / 4          # arc from the stagnation point to the equator
    return 5 * sqrt(ν * x / speed)
end

"""
    grid_budget(; nodes_per_diameter, domain_diameters, ...)

What one grid choice costs and what it actually resolves.

The two columns that decide everything are `seam_cells` and `blockage`, and they
pull against each other: a cubic domain of `L` diameters at `N` points per
diameter is `(N L)³` nodes, so buying seam resolution costs domain width at the
third power. `bytes_per_node` defaults to the populations plus the one `Int32`
per node the wall geometry carries.
"""
function grid_budget(; nodes_per_diameter::Integer = 40, domain_diameters::Real = 8,
                     diameter::Real = 0.0748, seam_height::Real = 0.00079,
                     speed::Real = 39.0, lattice_speed::Real = 0.05,
                     ν::Real = AIR_VISCOSITY, bytes_per_node::Real = 27 * 4 + 4,
                     mlups::Real = 2300, flight_time::Real = 0.45)
    units = LatticeUnits(; diameter = diameter, nodes_per_diameter = nodes_per_diameter,
                         speed = speed, lattice_speed = lattice_speed, ν = ν)
    edge = round(Int, nodes_per_diameter * domain_diameters)
    nodes = Int128(edge)^3
    steps = flight_time / units.dt
    return (edge = edge, nodes = Int(nodes), gib = Float64(nodes) * bytes_per_node / 2^30,
            dx_mm = 1000 * units.dx,
            seam_cells = seam_height / units.dx,
            boundary_layer_cells =
                boundary_layer_thickness(; diameter = diameter, speed = speed, ν = ν) / units.dx,
            blockage = 0.5 / domain_diameters,     # sphere radius over box edge
            steps = steps,
            hours = Float64(nodes) * steps / (mlups * 1e6) / 3600)
end
