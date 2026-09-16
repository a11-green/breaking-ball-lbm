@testset "AA-pattern streaming (D3Q27)" begin
    lat = D3Q27()

    @testset "cube slot arithmetic matches the lattice tables" begin
        for q in 1:27
            s = BBL.CUBE27[q]
            @test cube_velocity(s) == (CX27[q], CY27[q], CZ27[q])
            @test cube_weight(Float64, s) ≈ W27[q]
            @test cube_opposite(s) == BBL.CUBE27[opposite(q)]
        end
        # Every slot is used exactly once, and opposites pair up.
        @test sort(collect(BBL.CUBE27)) == collect(1:27)
        for s in 1:27
            @test cube_opposite(cube_opposite(s)) == s
            @test cube_velocity(cube_opposite(s)) == .-cube_velocity(s)
        end
    end

    @testset "cube equilibrium matches the lattice equilibrium" begin
        for (ρ, u) in ((1.0, (0.0, 0.0, 0.0)), (0.98, (0.05, -0.02, 0.03)))
            for q in 1:27
                @test cube_equilibrium(BBL.CUBE27[q], ρ, u...) ≈ equilibrium(lat, q, ρ, u...)
            end
        end
    end

    @testset "order conversion round-trips" begin
        f = rand(4, 3, 5, 27)
        g = similar(f)
        back = similar(f)
        to_cube_order!(g, f)
        from_cube_order!(back, g)
        @test back == f
    end

    for operator in (:bgk, :central_moment)
        @testset "AA-pattern reproduces the two-lattice scheme ($operator)" begin
            # The reference path (collide, then push-stream into a second array) is
            # the one the analytic benchmarks validated. AA-pattern must agree with
            # it step for step, which is what makes the device kernels trustworthy.
            nx, ny, nz, τ = 7, 5, 6, 0.7
            field = (i, j, k) -> (1.0 + 0.01sin(i + 2j),
                                  0.02cos(i), -0.015sin(j + k), 0.01cos(k))
            grad = (i, j, k) -> [0.001i 0.002j 0.0; 0.0 -0.001k 0.0; 0.0 0.0 0.0]

            ref = LBMState(nx, ny, nz, τ; lattice = lat)
            init_with_gradients!(ref, field, grad)
            g = to_cube_order!(similar(ref.f), ref.f)

            force = (2e-5, -1e-5, 3e-5)
            nsteps = 8
            run!(ref, nsteps; force = force, operator = operator)
            aa_run!(g, nsteps, τ; force = force, operator = operator)

            got = from_cube_order!(similar(g), g)
            @test got ≈ ref.f rtol = 1e-12
        end
    end

    @testset "AA-pattern conserves mass" begin
        nx, ny, nz = 6, 6, 6
        s = LBMState(nx, ny, nz, 0.8; lattice = lat)
        init_equilibrium!(s, (i, j, k) -> (1.0 + 0.02sin(i + j + k), 0.01, 0.0, -0.01))
        g = to_cube_order!(similar(s.f), s.f)
        m0 = sum(g)
        aa_run!(g, 20, 0.8)
        @test sum(g) ≈ m0 rtol = 1e-12
    end

    @testset "the collision accepts any AbstractVector buffer" begin
        # The GPU kernel hands collide_buffer! a thread-local MVector, not a
        # Vector. Nothing on this path may demand a Vector specifically, or the
        # device compile fails with a MethodError it cannot even throw. A view is
        # the stand-in available without StaticArrays.
        for operator in (:bgk, :central_moment)
            populations = [0.03 + 0.004 * sin(2.3s) for s in 1:27]
            force = (1e-5, -2e-5, 3e-6)

            plain = copy(populations)
            collide_buffer!(plain, 0.7, force, Val(operator), 1.0, 1.0)

            padded = vcat(zeros(5), copy(populations), zeros(5))
            strided = @view padded[6:32]
            @test strided isa AbstractVector
            @test !(strided isa Vector)
            collide_buffer!(strided, 0.7, force, Val(operator), 1.0, 1.0)

            @test collect(strided) ≈ plain rtol = 1e-14
            @test all(iszero, padded[1:5]) && all(iszero, padded[33:37])   # stayed in bounds
        end
    end

    @testset "single precision works end to end" begin
        # The GPU runs Float32, so the host path has to hold up there too.
        n, τ = 24, 0.6
        tg = TaylorGreen(u0 = 0.02, n = n, ν = viscosity_from_tau(τ))
        s = LBMState{Float32}(n, n, 4, τ; lattice = lat)
        BBL.init!(s, tg)
        g = to_cube_order!(similar(s.f), s.f)
        @test eltype(g) == Float32

        from_cube_order!(s.f, g)
        e0 = total_kinetic_energy(s)
        aa_run!(g, 200, τ)
        from_cube_order!(s.f, g)
        e1 = total_kinetic_energy(s)

        ν_measured = -log(e1 / e0) / (4 * tg.k^2 * 200)
        @test ν_measured ≈ viscosity_from_tau(τ) rtol = 0.02
    end

    @testset "odd step counts are rejected" begin
        g = zeros(3, 3, 3, 27)
        @test_throws ArgumentError aa_run!(g, 3, 0.8)
    end
end
