@testset "discrete velocity sets" begin
    δ(α, β) = α == β ? 1.0 : 0.0

    for lat in (D3Q19(), D3Q27())
        nq = nvelocities(lat)
        c = (cxs(lat), cys(lat), czs(lat))
        w = weights(lat)

        @testset "$(typeof(lat))" begin
            @test length(w) == nq
            @test sum(w) ≈ 1.0

            @testset "opposite velocities" begin
                for q in 1:nq
                    p = opposite(q)
                    @test (c[1][p], c[2][p], c[3][p]) == (-c[1][q], -c[2][q], -c[3][q])
                    @test opposite(p) == q
                    @test w[p] == w[q]
                end
            end

            @testset "velocity set" begin
                norms = [c[1][q]^2 + c[2][q]^2 + c[3][q]^2 for q in 1:nq]
                @test count(==(0), norms) == 1
                @test count(==(1), norms) == 6
                @test count(==(2), norms) == 12
                @test count(==(3), norms) == (nq == 27 ? 8 : 0)
                @test length(unique(zip(c...))) == nq
            end

            @testset "lattice moments" begin
                for α in 1:3
                    @test sum(w[q] * c[α][q] for q in 1:nq) ≈ 0.0 atol = 1e-14
                end
                for α in 1:3, β in 1:3
                    m = sum(w[q] * c[α][q] * c[β][q] for q in 1:nq)
                    @test m ≈ CS2 * δ(α, β) atol = 1e-14
                end
                for α in 1:3, β in 1:3, γ in 1:3
                    m = sum(w[q] * c[α][q] * c[β][q] * c[γ][q] for q in 1:nq)
                    @test m ≈ 0.0 atol = 1e-14
                end
                for α in 1:3, β in 1:3, γ in 1:3, d in 1:3
                    m = sum(w[q] * c[α][q] * c[β][q] * c[γ][q] * c[d][q] for q in 1:nq)
                    expected = CS2^2 * (δ(α, β) * δ(γ, d) + δ(α, γ) * δ(β, d) + δ(α, d) * δ(β, γ))
                    @test m ≈ expected atol = 1e-14
                end
            end
        end
    end

    @testset "only D3Q27 factorises in all three directions" begin
        # Σ w cx² cy² cz² is c_s⁶ for a genuine tensor-product quadrature. D3Q19
        # has no corner velocities at all, so it gives zero — which is why the
        # central-moment operator needs D3Q27.
        mixed(lat) = sum(weights(lat)[q] * cxs(lat)[q]^2 * cys(lat)[q]^2 * czs(lat)[q]^2
                         for q in 1:nvelocities(lat))
        @test mixed(D3Q19()) ≈ 0.0 atol = 1e-15
        @test mixed(D3Q27()) ≈ CS2^3 atol = 1e-15
    end

    @testset "viscosity / relaxation time round trip" begin
        for τ in (0.51, 0.6, 0.8, 1.0, 2.0)
            @test tau_from_viscosity(viscosity_from_tau(τ)) ≈ τ
        end
        @test viscosity_from_tau(0.5) ≈ 0.0
        @test viscosity_from_tau(1.0) ≈ CS2 * 0.5
    end
end
