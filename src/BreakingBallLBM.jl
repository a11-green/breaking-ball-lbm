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
include("validation/taylor_green.jl")

export LBMState,
    Q19, CS2, W19, CX19, CY19, CZ19,
    opposite, equilibrium, nonequilibrium,
    viscosity, viscosity_from_tau, tau_from_viscosity,
    macroscopic, macroscopic!, macroscopic_fields, total_kinetic_energy,
    init_equilibrium!, init_with_gradients!,
    collide!, stream!, step!, run!,
    TaylorGreen

end # module
