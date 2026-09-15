@testset "Smagorinsky subgrid model" begin
    τ0 = 0.6
    ρ = 1.0

    @testset "closed form solves its own definition" begin
        # τ_tot is the root of ν_tot = ν₀ + (CsΔ)²|S| with |S| itself computed
        # from τ_tot, so check the two sides agree for a range of strains.
        for model in (Smagorinsky(cs = 0.1), Smagorinsky(cs = 0.16), Smagorinsky(cs = 0.2, Δ = 2.0)),
            Πnorm in (0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0)

            τtot = total_relaxation_time(model, τ0, ρ, Πnorm)
            νtot = viscosity_from_tau(τtot)
            S = strain_rate_magnitude(model, τ0, ρ, Πnorm)
            @test νtot ≈ viscosity_from_tau(τ0) + (model.cs * model.Δ)^2 * S rtol = 1e-12
            @test τtot >= τ0
        end
    end

    @testset "eddy viscosity is non-negative and grows with strain" begin
        model = Smagorinsky(cs = 0.16)
        @test eddy_viscosity(model, τ0, ρ, 0.0) ≈ 0.0 atol = 1e-15
        νs = [eddy_viscosity(model, τ0, ρ, Π) for Π in (1e-4, 1e-3, 1e-2, 1e-1)]
        @test all(νs .>= 0)
        @test issorted(νs)
        # cs = 0 must switch the model off entirely.
        @test total_relaxation_time(Smagorinsky(cs = 0.0), τ0, ρ, 0.5) ≈ τ0
    end

    @testset "flux norm vanishes at equilibrium" begin
        s = LBMState(4, 4, 4, τ0)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.02, -0.01, 0.005))
        ρ, ux, uy, uz = macroscopic(s, 2, 2, 2)
        @test nonequilibrium_flux_norm(s, 2, 2, 2, ρ, ux, uy, uz) < 1e-15
    end

    @testset "collision with cs = 0 matches plain BGK" begin
        off = Smagorinsky(cs = 0.0)
        s1 = LBMState(6, 6, 6, τ0)
        s2 = LBMState(6, 6, 6, τ0)
        field = (i, j, k) -> (1.0 + 0.01sin(i + j), 0.03cos(k), 0.02sin(i), -0.01cos(j))
        grad = (i, j, k) -> [0.001i 0.002j 0.0; 0.0 -0.001i 0.0; 0.0 0.0 0.0]
        init_with_gradients!(s1, field, grad)
        init_with_gradients!(s2, field, grad)

        collide!(s1)
        collide!(s2; les = off)
        @test s1.f ≈ s2.f rtol = 1e-14
    end

    @testset "subgrid viscosity damps a Taylor-Green vortex faster" begin
        n, τ, u0 = 32, 0.6, 0.05
        tg = TaylorGreen(u0 = u0, n = n, ν = viscosity_from_tau(τ))

        function decay(les)
            s = LBMState(n, n, 4, τ)
            BBL.init!(s, tg)
            e0 = total_kinetic_energy(s)
            for _ in 1:200
                collide!(s; les = les)
                stream!(s)
            end
            return total_kinetic_energy(s) / e0
        end

        plain = decay(nothing)
        modelled = decay(Smagorinsky(cs = 0.16))
        @test modelled < plain
        @test plain - modelled < 0.2      # a subgrid correction, not a different flow
    end
end
