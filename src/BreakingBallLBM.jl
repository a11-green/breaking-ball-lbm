"""
    BreakingBallLBM

Lattice Boltzmann solver for the aerodynamics and trajectory of a spinning,
seamed baseball. See `docs/design/DESIGN.md` for the theory and the roadmap.

This is the P1 stage: a CPU reference implementation (D3Q19 + BGK, periodic
domain) verified against the decaying Taylor-Green vortex.
"""
module BreakingBallLBM

include("core/lattice.jl")
include("core/state.jl")
include("core/collision.jl")
include("core/streaming.jl")
include("turbulence/smagorinsky.jl")
include("boundary/links.jl")
include("boundary/bounceback.jl")
include("geometry/seam.jl")
include("geometry/sdf.jl")
include("validation/taylor_green.jl")
include("validation/poiseuille.jl")
include("validation/sphere_array.jl")

export LBMState,
    Q19, CS2, W19, CX19, CY19, CZ19,
    opposite, equilibrium, nonequilibrium,
    viscosity, viscosity_from_tau, tau_from_viscosity,
    macroscopic, macroscopic!, macroscopic_fields, total_kinetic_energy,
    init_equilibrium!, init_with_gradients!,
    collide!, stream!, step!, run!, fluid_velocity,
    Smagorinsky, total_relaxation_time, eddy_viscosity, nonequilibrium_flux_norm,
    strain_rate_magnitude,
    BounceBackLinks, build_links, solid_mask, refine_delta,
    bounce_back_values!, apply_bounce_back!, init_solid!,
    BaseballSeam, seam_point, seam_polyline, seam_length,
    BaseballGeometry, sdf, sphere_sdf, sdf_field, sdf_field!, solid_volume,
    TaylorGreen, PoiseuilleChannel, poiseuille_velocity, poiseuille_peak, channel_sdf,
    sphere_sdf_field, sphere_sdf_fn, exact_sphere_delta,
    hasimoto_factor, stokes_drag, superficial_velocity

end # module
