# V&V-1: decaying Taylor-Green vortex against the analytic incompressible solution.
# Two properties are checked:
#   1. the realised viscosity matches ν = c_s²(τ - 1/2), via the energy decay rate;
#   2. the scheme is second-order accurate in space under diffusive scaling.
@testset "Taylor-Green vortex (V&V-1)" begin
    nz = 4   # the vortex is uniform along z; a few cells keep the 3D code paths live

    @testset "viscous decay rate matches ν = cs²(τ - 1/2)" begin
        n, τ, u0 = 32, 0.6, 0.01
        ν = viscosity_from_tau(τ)
        tg = TaylorGreen(u0 = u0, n = n, ν = ν)

        s = LBMState(n, n, nz, τ)
        BBL.init!(s, tg)
        e0 = total_kinetic_energy(s)

        nsteps = 200
        run!(s, nsteps)
        e1 = total_kinetic_energy(s)

        # Kinetic energy ~ u², so it decays twice as fast as the velocity.
        expected = exp(-4 * ν * tg.k^2 * nsteps)
        @test e1 / e0 ≈ expected rtol = 0.02

        # An effective viscosity recovered from the measured decay.
        ν_measured = -log(e1 / e0) / (4 * tg.k^2 * nsteps)
        @test ν_measured ≈ ν rtol = 0.02
    end

    @testset "velocity field follows the analytic solution" begin
        n, τ, u0 = 32, 0.6, 0.01
        ν = viscosity_from_tau(τ)
        tg = TaylorGreen(u0 = u0, n = n, ν = ν)

        s = LBMState(n, n, nz, τ)
        BBL.init!(s, tg)
        @test first(BBL.l2_velocity_error(s, tg, 0)) < 1e-12   # exact at t = 0

        nsteps = 200
        run!(s, nsteps)
        rel, _ = BBL.l2_velocity_error(s, tg, nsteps)
        @test rel < 0.02

        # The flow stays z-invariant and two-dimensional.
        for k in 2:nz, j in 1:n, i in 1:n
            ρ1, ux1, uy1, uz1 = macroscopic(s, i, j, 1)
            ρk, uxk, uyk, uzk = macroscopic(s, i, j, k)
            @test ρk ≈ ρ1 rtol = 1e-12
            @test uxk ≈ ux1 atol = 1e-13
            @test uyk ≈ uy1 atol = 1e-13
            @test uzk ≈ 0.0 atol = 1e-13
        end
    end

    @testset "second-order spatial convergence" begin
        # Diffusive scaling: τ (hence ν) fixed, u0 ∝ 1/n, steps ∝ n², so both the
        # Reynolds number and the non-dimensional end time stay put while Δx shrinks.
        #
        # τ = 0.8 keeps the measurement well conditioned. At τ near 1/2 the BGK
        # higher-order error terms are comparable to the second-order one and the
        # measured order wanders (0.55 gives 1.3–1.8 here, and drifts above 2 when
        # the Mach number is lowered — a sign of competing error terms, not of a
        # different convergence rate).
        τ = 0.8
        ν = viscosity_from_tau(τ)
        tstar = 0.1          # non-dimensional end time, t* = t·u0/n
        u0_base, n_base = 0.04, 16

        errors = Float64[]
        grids = (16, 32, 64)
        for n in grids
            u0 = u0_base * n_base / n
            nsteps = round(Int, tstar * n / u0)
            tg = TaylorGreen(u0 = u0, n = n, ν = ν)
            s = LBMState(n, n, nz, τ)
            BBL.init!(s, tg)
            run!(s, nsteps)
            rel, _ = BBL.l2_velocity_error(s, tg, nsteps)
            push!(errors, rel)
        end

        orders = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
        @info "Taylor-Green convergence" grids errors orders
        @test all(<(0.05), errors)
        @test all(>(1.8), orders)
        @test orders[end] < 2.3   # not better than second order either
    end
end
