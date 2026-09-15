@testset "walls, bounce-back and wall force" begin

    """Raw momentum Σ_q f_q c_q summed over fluid nodes."""
    function fluid_momentum(s, solid)
        p = zeros(Float64, 3)
        for k in 1:s.nz, j in 1:s.ny, i in 1:s.nx
            solid[i, j, k] && continue
            for q in 1:Q19
                fq = s.f[i, j, k, q]
                p[1] += CX19[q] * fq
                p[2] += CY19[q] * fq
                p[3] += CZ19[q] * fq
            end
        end
        return (p[1], p[2], p[3])
    end

    @testset "link extraction" begin
        dims = (20, 20, 20)
        R = 6.0
        ϕ = sphere_sdf_field(dims, R)
        solid = solid_mask(ϕ)
        links = build_links(ϕ)

        @test length(links) > 0
        @test all(0 .< links.δ .<= 1)
        for n in eachindex(links.q)
            i, j, k, q = links.i[n], links.j[n], links.k[n], Int(links.q[n])
            @test !solid[i, j, k]                                    # links start in fluid
            ns = (mod1(i + CX19[q], 20), mod1(j + CY19[q], 20), mod1(k + CZ19[q], 20))
            @test solid[ns...]                                       # and point into solid
        end

        # Every fluid/solid interface is covered exactly once.
        expected = 0
        for k in 1:20, j in 1:20, i in 1:20
            solid[i, j, k] && continue
            for q in 2:Q19
                solid[mod1(i + CX19[q], 20), mod1(j + CY19[q], 20), mod1(k + CZ19[q], 20)] &&
                    (expected += 1)
            end
        end
        @test length(links) == expected

        # δ from the signed-distance field matches the exact sphere crossing on
        # axis-aligned links, where the wall distance is analytic.
        center = (dims .+ 1) ./ 2
        for n in eachindex(links.q)
            q = Int(links.q[n])
            (CX19[q]^2 + CY19[q]^2 + CZ19[q]^2) == 1 || continue
            xw = links.xw[n]
            @test sqrt(xw[1]^2 + xw[2]^2 + xw[3]^2) ≈ R rtol = 0.05
        end
    end

    @testset "momentum balance is exact" begin
        # Δ(fluid momentum) = (body force on every fluid node) - (force on the wall).
        # This holds step by step, to round-off, for either bounce-back rule.
        dims = (18, 18, 18)
        ϕ = sphere_sdf_field(dims, 5.0)
        solid = solid_mask(ϕ)
        links = build_links(ϕ)
        vals = zeros(Float64, length(links))
        nfluid = count(!, solid)
        g = (2e-5, 0.0, 0.0)

        for rule in (:interpolated, :halfway), spin in ((0.0, 0.0, 0.0), (0.0, 0.0, 1e-3))
            s = LBMState(dims..., 0.9)
            init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
            init_solid!(s, solid)

            for _ in 1:5
                p0 = fluid_momentum(s, solid)
                F, _ = step!(s, links, vals; force = g, solid = solid, rule = rule, spin = spin)
                p1 = fluid_momentum(s, solid)
                for a in 1:3
                    @test p1[a] - p0[a] ≈ nfluid * g[a] - F[a] atol = 1e-12
                end
            end
        end
    end

    @testset "fluid at rest stays at rest" begin
        dims = (16, 16, 16)
        ϕ = sphere_sdf_field(dims, 4.5)
        solid = solid_mask(ϕ)
        links = build_links(ϕ)
        vals = zeros(Float64, length(links))

        s = LBMState(dims..., 0.8)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
        init_solid!(s, solid)
        for _ in 1:20
            F, τq = step!(s, links, vals; solid = solid)
            @test all(abs.(F) .< 1e-14)
            @test all(abs.(τq) .< 1e-14)
        end
        for k in 1:16, j in 1:16, i in 1:16
            solid[i, j, k] && continue
            ρ, ux, uy, uz = macroscopic(s, i, j, k)
            @test ρ ≈ 1.0 rtol = 1e-12
            @test max(abs(ux), abs(uy), abs(uz)) < 1e-14
        end
    end

    @testset "Poiseuille flow with off-grid walls" begin
        # Walls at 3.3 and 20.7: neither sits midway between nodes, so plain
        # bounce-back must get the channel width wrong and the interpolated rule
        # must get it right.
        τ = 1.0
        ν = viscosity_from_tau(τ)
        ch = PoiseuilleChannel(y_lo = 3.3, y_hi = 20.7, force = 4.0e-6, ν = ν)
        dims = (4, 24, 4)
        ϕ = channel_sdf(ch, dims)
        solid = solid_mask(ϕ)
        links = build_links(ϕ)
        g = (ch.force, 0.0, 0.0)

        function run_channel(rule)
            s = LBMState(dims..., τ)
            init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
            init_solid!(s, solid)
            vals = zeros(Float64, length(links))
            local F
            for _ in 1:8000
                F, _ = step!(s, links, vals; force = g, solid = solid, rule = rule)
            end
            profile = [fluid_velocity(s, 2, j, 2, g)[2] for j in 1:dims[2]]
            return s, profile, F
        end

        s, profile, F = run_channel(:interpolated)
        fluid_rows = [j for j in 1:dims[2] if !solid[2, j, 2]]
        @test fluid_rows == collect(4:20)

        # The profile follows the analytic parabola for the true wall positions.
        for j in fluid_rows
            @test profile[j] ≈ poiseuille_velocity(ch, j) rtol = 0.01
        end

        # Recover the wall positions by fitting the parabola through the profile:
        # u = a(y - y_lo)(y_hi - y) has roots exactly at the walls.
        function fit_walls(profile, rows)
            # Least squares fit of u(y) = c0 + c1 y + c2 y².
            A = [ones(length(rows)) Float64.(rows) Float64.(rows) .^ 2]
            c = A \ [profile[j] for j in rows]
            disc = sqrt(c[2]^2 - 4 * c[3] * c[1])
            return ((-c[2] + disc) / (2c[3]), (-c[2] - disc) / (2c[3]))
        end

        lo, hi = fit_walls(profile, fluid_rows)
        @test lo ≈ ch.y_lo atol = 0.05
        @test hi ≈ ch.y_hi atol = 0.05

        # Total wall force balances the body force at steady state.
        nfluid = count(!, solid)
        @test F[1] ≈ nfluid * ch.force rtol = 1e-3

        # Plain bounce-back puts both walls midway instead, i.e. at 3.5 and 20.5.
        _, profile_h, _ = run_channel(:halfway)
        lo_h, hi_h = fit_walls(profile_h, fluid_rows)
        @test lo_h ≈ 3.5 atol = 0.05
        @test hi_h ≈ 20.5 atol = 0.05
        @test abs(lo_h - ch.y_lo) > 4 * abs(lo - ch.y_lo)   # interpolation really helps
    end

    @testset "torque on a rotating sphere" begin
        # Creeping flow around a sphere spinning at ω has torque 8πμR³ω.
        dims = (28, 28, 28)
        R = 5.5
        τ = 1.0
        ν = viscosity_from_tau(τ)
        μ = ν                                   # ρ₀ = 1 in lattice units
        ωz = 1.5e-3

        ϕ = sphere_sdf_field(dims, R)
        solid = solid_mask(ϕ)
        links = build_links(ϕ)
        vals = zeros(Float64, length(links))

        s = LBMState(dims..., τ)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
        init_solid!(s, solid)

        local τq
        for _ in 1:4000
            _, τq = step!(s, links, vals; solid = solid, spin = (0.0, 0.0, ωz))
        end

        analytic = -8π * μ * R^3 * ωz           # the fluid resists the rotation
        @test τq[3] ≈ analytic rtol = 0.1
        @test abs(τq[1]) < 0.02 * abs(analytic)
        @test abs(τq[2]) < 0.02 * abs(analytic)
    end
end
