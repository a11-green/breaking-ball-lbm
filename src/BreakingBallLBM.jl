"""
    BreakingBallLBM

Lattice Boltzmann solver for the aerodynamics and trajectory of a spinning,
seamed baseball. See `docs/design/DESIGN.md` for the theory and the roadmap.

CPU reference implementation. D3Q19 and D3Q27 lattices; BGK for verification
against analytic solutions and a central-moment operator for the high-Reynolds
production runs; Smagorinsky subgrid viscosity; interpolated bounce-back on
curved, rotating walls with momentum-exchange forces; and the parametric seam
geometry of the ball.
"""
module BreakingBallLBM

using TOML

include("trajectory/quaternion.jl")
include("core/lattice.jl")
include("core/state.jl")
include("turbulence/smagorinsky.jl")
include("core/collision.jl")
include("core/central_moments.jl")
include("core/aa_pattern.jl")
include("core/streaming.jl")
include("boundary/links.jl")
include("boundary/bounceback.jl")
include("boundary/aa_walls.jl")
include("boundary/open.jl")
include("geometry/seam.jl")
include("geometry/sdf.jl")
include("geometry/rotating.jl")
include("validation/taylor_green.jl")
include("validation/poiseuille.jl")
include("trajectory/units.jl")
include("trajectory/body.jl")
include("trajectory/frame.jl")
include("trajectory/aero_model.jl")
include("trajectory/trajectory.jl")
include("trajectory/coupling.jl")
include("refine/two_grid.jl")
include("refine/multi_grid.jl")
include("postprocess/fields.jl")
include("postprocess/pitch_view.jl")
include("postprocess/vtk.jl")
include("config.jl")
include("refine/refined_flow.jl")
include("refine/chain_flow.jl")
include("validation/sphere_array.jl")

export LBMState, Lattice, D3Q19, D3Q27,
    CS2, nvelocities, cxs, cys, czs, weights,
    CX19, CY19, CZ19, W19, CX27, CY27, CZ27, W27,
    opposite, equilibrium, nonequilibrium,
    viscosity, viscosity_from_tau, tau_from_viscosity,
    macroscopic, macroscopic!, macroscopic_fields, total_kinetic_energy,
    init_equilibrium!, init_with_gradients!,
    collide!, collide_central_moments!, stream!, step!, run!, fluid_velocity,
    cube_velocity, cube_opposite, cube_weight, cube_equilibrium,
    to_cube_order!, from_cube_order!, aa_gather!, aa_scatter!, collide_buffer!,
    aa_step_node!, aa_run!, gpu_run!, gpu_run_walls!, gpu_wall, gpu_flow, gpu_rotating_flow,
    gpu_two_grid, gpu_refined_flow, gpu_gather_bandwidth,
    gpu_copy_bandwidth, gpu_backend_loaded,
    Smagorinsky, total_relaxation_time, eddy_viscosity, nonequilibrium_flux_norm,
    strain_rate_magnitude,
    BounceBackLinks, build_links, solid_mask, refine_delta,
    bounce_back_values!, apply_bounce_back!, init_solid!,
    WallField, build_wall_field, aa_scatter_walls!, aa_step_node_walls!, aa_run_walls!,
    BaseballSeam, seam_point, seam_polyline, seam_length, seam_distance,
    BaseballGeometry, BallShape, shape_sdf, sdf, sdf_exhaustive, sphere_sdf, sdf_field, sdf_field!, solid_volume,
    RotatingWall, recut!, refill_fresh!, body_sdf, shell_is_sufficient,
    maybe_recut!, surface_drift_per_step, max_substeps,
    recut_distance, recut_delta, recut_column!, refill_node!,
    TaylorGreen, PoiseuilleChannel, poiseuille_velocity, poiseuille_peak, channel_sdf,
    sphere_sdf_field, sphere_sdf_fn, exact_sphere_delta,
    hasimoto_factor, stokes_drag, superficial_velocity,
    TwoGrid, fine_tau, neq_rescale, fine_force, grid_sizes, check_patch,
    GridChain, levels, level_taus, base_grid, finest_grid, chain_sizes,
    chain_cycle_walls!, chain_force, chain_torque, init_chain!, level_origin,
    ChainFlow, init_chain_flow!, refresh_base_solid!, deepest_index,
    level_steps!, level_wall_steps!, gpu_chain, gpu_chain_flow,
    interface_fill!, restrict!, save_coarse!, refine_cycle!,
    init_refined!, level_macroscopic, node_macroscopic,
    TrajectorySamples, samples, break_references, seam_world, spin_axis_world,
    PITCH_TYPES, pitch_family, at_time, plate_box, coefficient_series,
    velocity_field, density_field, magnitude, velocity_gradient,
    write_vtk, write_snapshot, snapshot_box,
    load_settings!, settings_toml, coerce_setting,
    vorticity, q_criterion, field_limits, slice_field,
    interface_nodes, fill_interface_node!, restrict_node!, coarse_span,
    RefinedFlow, refine_cycle_walls!, coarse_force, coarse_torque,
    refresh_coarse_solid!, init_refined_flow!,
    Quat, quat_from_axis_angle, rotate, unrotate, quat_rate, rotation_matrix,
    LatticeUnits, mach_number, lattice_reynolds, resolution_report,
    boundary_layer_thickness, grid_budget,
    to_lattice_length, to_lattice_velocity, to_lattice_acceleration,
    to_lattice_rate, to_lattice_steps,
    to_physical_length, to_physical_velocity, to_physical_time,
    to_physical_force, to_physical_torque,
    AIR_DENSITY, AIR_VISCOSITY, GRAVITY,
    BallProperties, BaseballProperties, BallState, advance, ball_rates,
    spin_from_rpm, spin_rpm, speed, spin_parameter, transverse_spin,
    frame_acceleration, fluid_body_acceleration, freestream_velocity,
    lattice_body_force, lattice_freestream, lattice_spin,
    aerodynamic_coefficients,
    seam_orientation, orientation_drift,
    CoefficientAero, BallisticAero, no_magnus,
    PLATE_DISTANCE, PFX_SEGMENT, simulate_trajectory, state_at_distance,
    PitchMetrics, pitch_metrics, inches,
    mean_fluid_velocity, region_mean_velocity, fluid_node_count, to_lattice_force,
    flow_wall, flow_recuts, flow_coarse_solid, solid_mask_of,
    OpenChannel, apply_open!, open_node!, open_is_clear, inlet_source, outlet_source,
    flow_fluid_count, advance_flow!, flow_mean_velocity,
    PitchRun, PitchState, couple_step!, couple_residual, spin_up!, fly!

"""
    gpu_run!(g, nsteps, τ; force, operator, omega_bulk, omega_higher, omega_odd)

Advance an AA-pattern state held on the device. Defined by the CUDA extension,
so it needs `using CUDA, StaticArrays` before it resolves.
"""
function gpu_run! end

"""
    gpu_copy_bandwidth(T = Float32; n, repeats)

Bytes per second a plain device-to-device copy sustains. The datasheet figure is
not a safe stand-in — the same card ships with two memory types — so the
benchmark measures the ceiling it is about to compare against.
"""
function gpu_copy_bandwidth end

"""
    gpu_run_walls!(g, wall, contrib, nsteps, τ; kwargs...)

Advance a wall-bounded AA-pattern state on the device, returning the
`(force, torque)` of the final step. `wall` must hold device arrays (see
[`gpu_wall`](@ref)) and `contrib` is scratch of size `(6, length(wall))`.
"""
function gpu_run_walls! end

"""
    gpu_wall(wall)

Copy a [`WallField`](@ref) to the device.
"""
function gpu_wall end

"""
    gpu_flow(wall)

Device-side state for a coupled run: the wall geometry, the force-reduction
scratch and the fluid mask. Pass the result to [`couple_step!`](@ref) in place
of a host `WallField`.
"""
function gpu_flow end

"""
    gpu_rotating_flow(rw)

Move a [`RotatingWall`](@ref) to the device, with the scratch a re-cut needs.
Defined by the CUDA extension.
"""
function gpu_rotating_flow end

"""
    gpu_two_grid(rg; layers)
    gpu_refined_flow(rf; layers)

Move a [`TwoGrid`](@ref) or a [`RefinedFlow`](@ref) to the device. Defined by
the CUDA extension.
"""
function gpu_two_grid end
function gpu_refined_flow end

"""
    gpu_chain(ch; layers)
    gpu_chain_flow(cf; layers)

Move a [`GridChain`](@ref) or a [`ChainFlow`](@ref) to the device, sharing the
arrays between levels as the host chain does. Defined by the CUDA extension.
"""
function gpu_gather_bandwidth end
function gpu_chain end
function gpu_chain_flow end

"""Whether the CUDA extension has been loaded."""
gpu_backend_loaded() = !isnothing(Base.get_extension(@__MODULE__, :BreakingBallLBMCUDAExt))

end # module
