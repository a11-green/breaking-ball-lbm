@testset "streaming" begin
    @testset "populations move one cell along their velocity" begin
        nx, ny, nz = 5, 4, 3
        for q in 1:Q19
            s = LBMState(nx, ny, nz, 0.8)
            fill!(s.f, 0.0)
            i0, j0, k0 = 2, 3, 2
            s.f[i0, j0, k0, q] = 1.0
            stream!(s)

            i1 = mod1(i0 + CX19[q], nx)
            j1 = mod1(j0 + CY19[q], ny)
            k1 = mod1(k0 + CZ19[q], nz)
            @test s.f[i1, j1, k1, q] == 1.0
            @test sum(s.f) == 1.0   # nothing created or lost
        end
    end

    @testset "periodic wrap-around" begin
        nx, ny, nz = 4, 4, 4
        s = LBMState(nx, ny, nz, 0.8)
        fill!(s.f, 0.0)
        s.f[nx, 1, 1, 2] = 1.0        # +x at the last cell wraps to the first
        stream!(s)
        @test s.f[1, 1, 1, 2] == 1.0

        fill!(s.f, 0.0)
        s.f[1, 1, 1, 3] = 1.0         # -x at the first cell wraps to the last
        stream!(s)
        @test s.f[nx, 1, 1, 3] == 1.0
    end

    @testset "streaming conserves total mass" begin
        s = LBMState(7, 6, 5, 0.75)
        init_equilibrium!(s, (i, j, k) -> (1.0 + 0.02 * sin(i) * cos(j),
                                           0.01 * cos(k), 0.02 * sin(i), 0.0))
        m0 = sum(s.f)
        for _ in 1:10
            stream!(s)
        end
        @test sum(s.f) ≈ m0 rtol = 1e-14
    end

    @testset "uniform flow is preserved exactly" begin
        s = LBMState(8, 6, 4, 0.65)
        ρ0, u0 = 1.0, (0.05, -0.03, 0.02)
        init_equilibrium!(s, (i, j, k) -> (ρ0, u0...))
        run!(s, 25)
        for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
            ρ, ux, uy, uz = macroscopic(s, i, j, k)
            @test ρ ≈ ρ0 rtol = 1e-12
            @test ux ≈ u0[1] atol = 1e-12
            @test uy ≈ u0[2] atol = 1e-12
            @test uz ≈ u0[3] atol = 1e-12
        end
    end
end
