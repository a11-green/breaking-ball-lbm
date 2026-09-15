@testset "central-moment collision (D3Q27)" begin
    lat = D3Q27()

    """Central moments computed the slow, obvious way, straight from the definition."""
    function reference_moments(f::Vector{Float64}, u)
        κ = zeros(3, 3, 3)
        for q in 1:27, o in 0:2, n in 0:2, m in 0:2
            κ[m+1, n+1, o+1] += f[q] * (CX27[q] - u[1])^m * (CY27[q] - u[2])^n * (CZ27[q] - u[3])^o
        end
        return κ
    end

    """Pack populations into the 3×3×3 cube the transform works on."""
    cube(f) = [f[q] for q in 1:27][sortperm(collect(BBL.CUBE27))]

    @testset "transform matches the definition" begin
        f = [0.01 + 0.003 * sin(3.1q) for q in 1:27]
        for u in ((0.0, 0.0, 0.0), (0.05, -0.03, 0.02), (0.1, 0.1, -0.1))
            buf = cube(f)
            BBL.to_moments!(buf, u...)
            κ = reference_moments(f, u)
            for o in 0:2, n in 0:2, m in 0:2
                @test buf[1+m+3n+9o] ≈ κ[m+1, n+1, o+1] atol = 1e-14
            end
        end
    end

    @testset "transform round-trips" begin
        f = [0.02 + 0.007 * cos(1.7q) for q in 1:27]
        for u in ((0.0, 0.0, 0.0), (0.04, 0.02, -0.05), (-0.1, 0.08, 0.03))
            buf = cube(f)
            original = copy(buf)
            BBL.to_moments!(buf, u...)
            BBL.to_populations!(buf, u...)
            @test buf ≈ original rtol = 1e-13
        end
    end

    @testset "collision conserves mass and momentum" begin
        s = LBMState(5, 4, 3, 0.7; lattice = lat)
        init_equilibrium!(s, (i, j, k) -> (1.0 + 0.01sin(i + 2j), 0.03cos(i), -0.02sin(j), 0.01cos(k)))
        s.f .+= 1e-4 .* [sin(i + j + k + q) for i in 1:5, j in 1:4, k in 1:3, q in 1:27]

        before = [macroscopic(s, i, j, k) for i in 1:5, j in 1:4, k in 1:3]
        collide!(s; operator = :central_moment)
        after = [macroscopic(s, i, j, k) for i in 1:5, j in 1:4, k in 1:3]
        for idx in eachindex(before)
            for a in 1:4
                @test after[idx][a] ≈ before[idx][a] atol = 1e-13
            end
        end
    end

    @testset "uniform flow is preserved" begin
        s = LBMState(6, 6, 6, 0.65; lattice = lat)
        ρ0, u0 = 1.0, (0.05, -0.03, 0.02)
        init_equilibrium!(s, (i, j, k) -> (ρ0, u0...))
        run!(s, 30; operator = :central_moment)
        for k in 1:6, j in 1:6, i in 1:6
            ρ, ux, uy, uz = macroscopic(s, i, j, k)
            @test ρ ≈ ρ0 rtol = 1e-12
            @test (ux, uy, uz) .- u0 |> v -> maximum(abs.(v)) < 1e-12
        end
    end

    @testset "body force enters the momentum exactly" begin
        s = LBMState(5, 5, 5, 0.8; lattice = lat)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
        g = (3e-5, -1e-5, 2e-5)
        for step in 1:6
            collide!(s; force = g, operator = :central_moment)
            stream!(s)
            # Every node gains exactly g of raw momentum per step.
            for a in 1:3
                total = sum(s.f[i, j, k, q] * (cxs(lat), cys(lat), czs(lat))[a][q]
                            for i in 1:5, j in 1:5, k in 1:5, q in 1:27)
                @test total ≈ 125 * g[a] * step rtol = 1e-10
            end
        end
    end

    @testset "recovers the analytic viscosity" begin
        n, τ, u0 = 32, 0.6, 0.01
        ν = viscosity_from_tau(τ)
        tg = TaylorGreen(u0 = u0, n = n, ν = ν)
        s = LBMState(n, n, 4, τ; lattice = lat)
        BBL.init!(s, tg)
        e0 = total_kinetic_energy(s)
        nsteps = 200
        run!(s, nsteps; operator = :central_moment)
        e1 = total_kinetic_energy(s)

        ν_measured = -log(e1 / e0) / (4 * tg.k^2 * nsteps)
        @test ν_measured ≈ ν rtol = 0.02
        rel, _ = BBL.l2_velocity_error(s, tg, nsteps)
        @test rel < 0.02
    end

    @testset "more accurate than BGK, and close to second order" begin
        τ = 0.8
        ν = viscosity_from_tau(τ)
        tstar = 0.1
        u0_base, n_base = 0.04, 16
        grids = (16, 32, 64)

        function study(operator, lattice)
            errs = Float64[]
            for n in grids
                u0 = u0_base * n_base / n
                nsteps = round(Int, tstar * n / u0)
                tg = TaylorGreen(u0 = u0, n = n, ν = ν)
                s = LBMState(n, n, 4, τ; lattice = lattice)
                BBL.init!(s, tg)
                run!(s, nsteps; operator = operator)
                push!(errs, first(BBL.l2_velocity_error(s, tg, nsteps)))
            end
            return errs
        end

        cm = study(:central_moment, D3Q27())
        bgk = study(:bgk, D3Q19())
        orders = [log2(cm[i] / cm[i+1]) for i in 1:length(cm)-1]
        @info "central-moment vs BGK" grids cm bgk orders
        @test all(cm .< bgk)               # uniformly more accurate
        @test all(>(1.7), orders)
    end

    @testset "the cubic defect is the lattice's, not the operator's" begin
        # With c ∈ {-1,0,1} the lattice forces c³ = c, so the third central moment
        # cannot vanish as a Maxwellian's does: it is pinned at -ρu³. No collision
        # operator can remove it, which is why the drift sensitivity of this scheme
        # is of the same order as BGK's. Geier's cumulant method adds explicit
        # correction terms for it; those are not implemented here.
        for u in (0.05, 0.1, 0.2)
            s = LBMState(3, 3, 3, 0.8; lattice = lat)
            init_equilibrium!(s, (i, j, k) -> (1.0, u, 0.0, 0.0))
            ρ, ux, _, _ = macroscopic(s, 2, 2, 2)
            κ300 = sum(s.f[2, 2, 2, q] * (CX27[q] - ux)^3 for q in 1:27)
            @test κ300 ≈ -ρ * u^3 rtol = 1e-10
            @test abs(κ300) > 1e-6            # decidedly not the Maxwellian zero
        end
    end

    @testset "stays stable where BGK blows up" begin
        # Doubly periodic shear layer, the standard LBM stability benchmark, at a
        # relaxation time far closer to 1/2 than the production runs will need.
        τ = 0.5005
        n, u0, ε = 64, 0.1, 0.05
        δ = 1 / 28

        function survives(operator, lattice; steps = 1500)
            s = LBMState(n, n, 1, τ; lattice = lattice)
            init_equilibrium!(s, (i, j, k) -> begin
                x, y = (i - 1) / n, (j - 1) / n
                ux = y <= 0.5 ? u0 * tanh((y - 0.25) / δ) : u0 * tanh((0.75 - y) / δ)
                (1.0, ux, u0 * ε * sin(2π * (x + 0.25)), 0.0)
            end)
            e0 = total_kinetic_energy(s)
            for _ in 1:(steps ÷ 100)
                run!(s, 100; operator = operator)
                e = total_kinetic_energy(s)
                (isfinite(e) && e < 5e0 * e0) || return false
            end
            return true
        end

        @test !survives(:bgk, D3Q19())
        @test survives(:central_moment, D3Q27())
    end
end
