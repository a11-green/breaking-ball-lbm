@testset "equilibrium and collision" begin
    c = (CX19, CY19, CZ19)
    δ(α, β) = α == β ? 1.0 : 0.0

    @testset "equilibrium moments" begin
        for (ρ, u) in ((1.0, (0.0, 0.0, 0.0)),
                       (1.0, (0.05, 0.0, 0.0)),
                       (0.97, (0.02, -0.03, 0.01)),
                       (1.1, (-0.04, 0.05, -0.02)))
            feq = [equilibrium(q, ρ, u...) for q in 1:Q19]
            @test sum(feq) ≈ ρ
            for α in 1:3
                @test sum(feq[q] * c[α][q] for q in 1:Q19) ≈ ρ * u[α] atol = 1e-14
            end
            # Momentum flux: ρ u_α u_β + ρ cs² δ_αβ (exact for the 2nd-order equilibrium).
            for α in 1:3, β in 1:3
                m = sum(feq[q] * c[α][q] * c[β][q] for q in 1:Q19)
                @test m ≈ ρ * u[α] * u[β] + ρ * CS2 * δ(α, β) atol = 1e-14
            end
        end
    end

    @testset "equilibrium is positive at pitch-relevant Mach numbers" begin
        for q in 1:Q19
            @test equilibrium(q, 1.0, 0.1, 0.0, 0.0) > 0
            @test equilibrium(q, 1.0, 0.05, 0.05, 0.05) > 0
        end
    end

    @testset "collision conserves mass and momentum" begin
        s = LBMState(6, 5, 4, 0.7)
        init_equilibrium!(s, (i, j, k) -> (1.0 + 0.01 * sin(i + 2j + 3k),
                                           0.02 * cos(i), 0.01 * sin(j), -0.015 * cos(k)))
        s.f .+= 1e-4 .* [sin(i + j + k + q) for i in 1:6, j in 1:5, k in 1:4, q in 1:Q19]

        before = [macroscopic(s, i, j, k) for i in 1:6, j in 1:5, k in 1:4]
        collide!(s)
        after = [macroscopic(s, i, j, k) for i in 1:6, j in 1:5, k in 1:4]

        for idx in eachindex(before)
            ρb, uxb, uyb, uzb = before[idx]
            ρa, uxa, uya, uza = after[idx]
            @test ρa ≈ ρb rtol = 1e-13
            @test uxa ≈ uxb atol = 1e-13
            @test uya ≈ uyb atol = 1e-13
            @test uza ≈ uzb atol = 1e-13
        end
    end

    @testset "equilibrium is a fixed point of collision" begin
        s = LBMState(4, 4, 4, 0.9)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.03, -0.02, 0.01))
        f0 = copy(s.f)
        collide!(s)
        @test s.f ≈ f0 rtol = 1e-14
    end

    @testset "τ controls the relaxation rate" begin
        # A perturbation that leaves ρ and ρu untouched (so the equilibrium does not
        # move) must decay by exactly (1 - 1/τ) per collision.
        for τ in (0.6, 1.0, 1.5)
            s = LBMState(2, 2, 2, τ)
            init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
            f0 = copy(s.f)
            ε = 1e-3
            s.f[1, 1, 1, 2] += ε        # +x
            s.f[1, 1, 1, 3] += ε        # -x, so the momentum change cancels
            s.f[1, 1, 1, 1] -= 2ε       # rest, so the mass change cancels
            ρ, ux, uy, uz = macroscopic(s, 1, 1, 1)
            @test ρ ≈ 1.0 atol = 1e-14
            @test ux ≈ 0.0 atol = 1e-14

            collide!(s)
            # atol, not rtol: the expected factor is exactly 0 at τ = 1.
            @test (s.f[1, 1, 1, 2] - f0[1, 1, 1, 2]) / ε ≈ 1 - 1 / τ atol = 1e-10
            @test (s.f[1, 1, 1, 1] - f0[1, 1, 1, 1]) / (-2ε) ≈ 1 - 1 / τ atol = 1e-10
        end
    end
end
