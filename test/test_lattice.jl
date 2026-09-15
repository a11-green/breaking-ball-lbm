@testset "D3Q19 lattice" begin
    c = (CX19, CY19, CZ19)
    δ(α, β) = α == β ? 1.0 : 0.0

    @test length(W19) == Q19
    @test sum(W19) ≈ 1.0

    @testset "opposite velocities" begin
        for q in 1:Q19
            p = opposite(q)
            @test CX19[p] == -CX19[q]
            @test CY19[p] == -CY19[q]
            @test CZ19[p] == -CZ19[q]
            @test opposite(p) == q
            @test W19[p] == W19[q]
        end
    end

    @testset "velocity set" begin
        # One rest velocity, six face neighbours, twelve edge neighbours.
        norms = [CX19[q]^2 + CY19[q]^2 + CZ19[q]^2 for q in 1:Q19]
        @test count(==(0), norms) == 1
        @test count(==(1), norms) == 6
        @test count(==(2), norms) == 12
        # No duplicated directions.
        @test length(unique(zip(CX19, CY19, CZ19))) == Q19
    end

    @testset "lattice moments" begin
        # First moment vanishes.
        for α in 1:3
            @test sum(W19[q] * c[α][q] for q in 1:Q19) ≈ 0.0 atol = 1e-14
        end
        # Second moment: cs² δ_αβ.
        for α in 1:3, β in 1:3
            m = sum(W19[q] * c[α][q] * c[β][q] for q in 1:Q19)
            @test m ≈ CS2 * δ(α, β) atol = 1e-14
        end
        # Third moment vanishes.
        for α in 1:3, β in 1:3, γ in 1:3
            m = sum(W19[q] * c[α][q] * c[β][q] * c[γ][q] for q in 1:Q19)
            @test m ≈ 0.0 atol = 1e-14
        end
        # Fourth moment: cs⁴ (δ_αβ δ_γδ + δ_αγ δ_βδ + δ_αδ δ_βγ).
        for α in 1:3, β in 1:3, γ in 1:3, d in 1:3
            m = sum(W19[q] * c[α][q] * c[β][q] * c[γ][q] * c[d][q] for q in 1:Q19)
            expected = CS2^2 * (δ(α, β) * δ(γ, d) + δ(α, γ) * δ(β, d) + δ(α, d) * δ(β, γ))
            @test m ≈ expected atol = 1e-14
        end
    end

    @testset "viscosity / relaxation time round trip" begin
        for τ in (0.51, 0.6, 0.8, 1.0, 2.0)
            @test tau_from_viscosity(viscosity_from_tau(τ)) ≈ τ
        end
        @test viscosity_from_tau(0.5) ≈ 0.0
        @test viscosity_from_tau(1.0) ≈ CS2 * 0.5
    end
end
