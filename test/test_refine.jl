@testset "two-level grid refinement" begin
    T = Float64
    τc = 0.8
    dims = (24, 24, 24)
    lo, hi = (7, 7, 7), (18, 18, 18)

    @testset "the scaling relations are what acoustic scaling says" begin
        @test fine_tau(0.8, 2) ≈ 1.1 rtol = 1e-15
        @test fine_tau(0.5, 2) ≈ 0.5                # a singular τ stays singular
        @test viscosity_from_tau(fine_tau(τc, 2)) ≈ 2 * viscosity_from_tau(τc) rtol = 1e-14

        # α = (τ_f/τ_c)/m, equivalently τ_f(τ_c−1/2) / (τ_c(τ_f−1/2)).
        for τ in (0.51, 0.8, 2.0, 0.500031)
            τf = fine_tau(τ, 2)
            a = neq_rescale(τ, τf, 2)
            @test a ≈ τf * (τ - 0.5) / (τ * (τf - 0.5)) rtol = 1e-13
        end

        # α is not a constant: it runs from 1/m at τ → 1/2 to 1 as τ grows. A
        # formula calibrated at τ ≈ 1 is therefore wrong by nearly m at the
        # relaxation times a pitch Reynolds number forces — which is the regime
        # every production run sits in.
        @test neq_rescale(0.500031, fine_tau(0.500031, 2), 2) ≈ 0.5 rtol = 1e-4
        @test neq_rescale(5.0, fine_tau(5.0, 2), 2) ≈ 0.95 rtol = 1e-2
        # Writing τ−1/2 where τ belongs gives 1 everywhere: right nowhere, and
        # worst by a factor of m exactly where it matters.
        badform(τ) = ((fine_tau(τ, 2) - 0.5) / (τ - 0.5)) / 2
        @test badform(0.500031) ≈ 1.0 rtol = 1e-12
        @test badform(0.500031) / neq_rescale(0.500031, fine_tau(0.500031, 2), 2) ≈ 2 rtol = 1e-3

        @test collect(fine_force((4.0, -2.0, 0.0), 2)) ≈ [2.0, -1.0, 0.0]
    end

    @testset "the patch is placed where it is asked" begin
        rg = TwoGrid(T, dims, lo, hi, τc)
        @test size(rg.fine)[1:3] == (2 .* (hi .- lo) .+ 1)
        s = grid_sizes(rg)
        @test s.covered == prod(hi .- lo .+ 1)
        @test 0 < s.fraction < 1
        @test_throws ArgumentError TwoGrid(T, dims, lo, hi, τc; ratio = 4)
        @test_throws ArgumentError TwoGrid(T, dims, (1, 7, 7), hi, τc)   # touches the edge
        @test_throws ArgumentError TwoGrid(T, dims, lo, (24, 18, 18), τc)
    end

    @testset "uniform flow crosses the interface unchanged" begin
        uniform = (x, y, z) -> (1.0, 0.03, -0.02, 0.01)

        # The transfers alone, first: a constant equilibrium has no f^neq, so
        # both directions are pure equilibrium arithmetic and must be exact
        # whatever the collision does afterwards.
        rg = TwoGrid(T, dims, lo, hi, τc)
        init_refined!(rg, uniform)
        c0, f0 = copy(rg.coarse), copy(rg.fine)
        save_coarse!(rg)
        interface_fill!(rg, 1.0)
        restrict!(rg)
        @test maximum(abs.(rg.fine .- f0)) < 1e-16
        @test maximum(abs.(rg.coarse .- c0)) < 1e-15

        # Whole cycles under BGK, which leaves an equilibrium alone exactly.
        rg = TwoGrid(T, dims, lo, hi, τc)
        init_refined!(rg, uniform)
        for _ in 1:4
            refine_cycle!(rg; operator = :bgk)
        end
        @test maximum(abs.(rg.coarse .- c0)) < 1e-14
        @test maximum(abs.(rg.fine .- f0)) < 1e-14

        # The central-moment operator does *not* leave one alone: relaxing the
        # third cumulants to their Gaussian values fights the irreducible cubic
        # defect of a {−1,0,1} lattice, κ₃₀₀ = −ρu³ (§2.3). So a uniform flow is
        # not an exact fixed point of it, and the two-grid result inherits that.
        #
        # What has to be shown is that refinement inherits it and nothing more.
        # Two signatures say so: the drift scales as u³, which is the defect's
        # own scaling and no interpolation error's, and the ratio between the
        # two-grid and single-grid drifts is a constant independent of
        # amplitude — refinement rescales the operator's error, it does not add
        # a mechanism of its own.
        function drift(u)
            r = TwoGrid(T, dims, lo, hi, τc)
            field = (x, y, z) -> (1.0, u, -2u / 3, u / 3)
            init_refined!(r, field)
            base = copy(r.coarse)
            refine_cycle!(r; operator = :central_moment)
            two = maximum(abs.(r.coarse .- base))

            bare = copy(base)
            aa_run!(bare, 2, τc; operator = :central_moment)
            return two, maximum(abs.(bare .- base))
        end

        d1, s1 = drift(0.03)
        d2, s2 = drift(0.015)
        d3, s3 = drift(0.0075)

        @test 1e-7 < s1 < 1e-4                  # real, and O(u³) with u = 0.03
        @test d1 / d2 ≈ 8 rtol = 0.05           # cubic in the velocity
        @test d2 / d3 ≈ 8 rtol = 0.05
        @test d1 / s1 ≈ d2 / s2 rtol = 1e-6     # a fixed multiple, not a new error
        @test d2 / s2 ≈ d3 / s3 rtol = 1e-6
        @test 1 < d1 / s1 < 2
    end

    """A state at constant (ρ, u) carrying a constant shear ∂u_x/∂z = γ."""
    function shear!(g, ρ, u, γ, τ)
        nx, ny, nz = size(g, 1), size(g, 2), size(g, 3)
        for k in 1:nz, j in 1:ny, i in 1:nx, s in 1:27
            cx, cy, cz = BBL.cube_velocity(s)
            neq = -(cube_weight(T, s) * τ * ρ / CS2) * cx * cz * γ
            g[i, j, k, s] = cube_equilibrium(s, ρ, u[1], u[2], u[3]) + neq
        end
        return g
    end

    """Recover γ from a node's non-equilibrium part: Π^neq_xz = −τ ρ c_s² γ."""
    function shear_of(g, i, j, k, ρ, u, τ)
        Π = 0.0
        for s in 1:27
            cx, cy, cz = BBL.cube_velocity(s)
            Π += cx * cz * (g[i, j, k, s] - cube_equilibrium(s, ρ, u[1], u[2], u[3]))
        end
        return -Π / (τ * ρ * CS2)
    end

    @testset "the shear construction inverts itself" begin
        g = zeros(T, 4, 4, 4, 27)
        shear!(g, 1.0, (0.02, 0.0, 0.0), 3e-4, τc)
        @test shear_of(g, 2, 2, 2, 1.0, (0.02, 0.0, 0.0), τc) ≈ 3e-4 rtol = 1e-12
        ρ, ux, uy, uz = node_macroscopic(g, 2, 2, 2, T)
        @test ρ ≈ 1.0 rtol = 1e-13          # f^neq carries no mass
        @test ux ≈ 0.02 rtol = 1e-12        # and no momentum
        @test abs(uy) < 1e-15 && abs(uz) < 1e-15
    end

    @testset "the transfer preserves the physical strain rate" begin
        # This is the decisive test of α, and it needs no flow: build a known
        # constant shear on the coarse level, hand it across, and ask what shear
        # the fine level thinks it received. Acoustic scaling halves the strain
        # rate in lattice units — the *physical* rate is what must survive.
        rg = TwoGrid(T, dims, lo, hi, τc)
        ρ, u, γc = 1.0, (0.02, 0.0, 0.0), 4e-4
        shear!(rg.coarse, ρ, u, γc, rg.τc)
        shear!(rg.prev, ρ, u, γc, rg.τc)          # steady: same at both times

        interface_fill!(rg, 1.0)
        for (a, b, c) in ((1, 5, 5), (2, 6, 7), (size(rg.fine, 1), 9, 4))
            @test shear_of(rg.fine, a, b, c, ρ, u, rg.τf) ≈ γc / 2 rtol = 1e-12
            ρf, uxf, = node_macroscopic(rg.fine, a, b, c, T)
            @test ρf ≈ ρ rtol = 1e-13
            @test uxf ≈ u[1] rtol = 1e-12
        end

        # And the other way: the fine level's shear restricted onto the coarse
        # has to come back doubled, so that the physical rate is again the same.
        rg2 = TwoGrid(T, dims, lo, hi, τc)
        γf = 2e-4
        shear!(rg2.fine, ρ, u, γf, rg2.τf)
        restrict!(rg2)
        @test shear_of(rg2.coarse, lo[1] + 3, lo[2] + 3, lo[3] + 3, ρ, u, rg2.τc) ≈
              2 * γf rtol = 1e-11

        # The optional filter is a symmetric weighted mean, so a state with a
        # constant shear passes through it untouched. That it is nevertheless
        # off by default is a measurement, not a principle — see `restrict!`.
        rg3 = TwoGrid(T, dims, lo, hi, τc)
        shear!(rg3.fine, ρ, u, γf, rg3.τf)
        restrict!(rg3; filtered = true)
        @test maximum(abs.(rg3.coarse[lo[1]+1:hi[1]-1, lo[2]+1:hi[2]-1, lo[3]+1:hi[3]-1, :] .-
                           rg2.coarse[lo[1]+1:hi[1]-1, lo[2]+1:hi[2]-1, lo[3]+1:hi[3]-1, :])) < 1e-14
    end

    @testset "a vortex crosses the interface at second order" begin
        # The dynamic test, with the interface cut straight through the core of
        # the vortex — the worst placement, chosen so the interpolation error is
        # as exposed as it can be. Acoustic refinement of the whole problem (Δt
        # with Δx, τ set to hold the Reynolds number) separates the two error
        # sources: the spatial interpolation is second order in Δx, the
        # boundary's one-step staleness would be first.
        function vortex_error(n, cycles)
            τ = 0.5 + 0.3 * n / 32                    # ν_lattice ∝ n holds Re fixed
            tg = TaylorGreen(u0 = 0.02, n = n, ν = viscosity_from_tau(τ))
            nz = 8
            lo2 = (n ÷ 4 + 1, n ÷ 4 + 1, 2)
            hi2 = (3n ÷ 4, 3n ÷ 4, 7)
            field = (x, y, z) -> begin
                ux, uy, uz = BBL.velocity(tg, x - 1, y - 1, 0.0)
                (BBL.density(tg, x - 1, y - 1, 0.0), ux, uy, uz)
            end
            r = TwoGrid(T, (n, n, nz), lo2, hi2, τ)
            init_refined!(r, field)
            for _ in 1:cycles
                refine_cycle!(r; operator = :bgk)
            end

            t = 2 * cycles
            nf = size(r.fine, 1)
            num = 0.0; den = 0.0
            for c in 3:size(r.fine, 3)-2, b in 5:nf-4, a in 5:nf-4
                x = lo2[1] + (a - 1) / 2
                y = lo2[2] + (b - 1) / 2
                _, ux, uy, uz = node_macroscopic(r.fine, a, b, c, T)
                ex, ey, _ = BBL.velocity(tg, x - 1, y - 1, t)
                num += (ux - ex)^2 + (uy - ey)^2 + uz^2
                den += ex^2 + ey^2
            end
            return sqrt(num / den)
        end

        e16 = vortex_error(16, 12)
        e32 = vortex_error(32, 24)
        order = log2(e16 / e32)
        @info "refined Taylor-Green" e16 e32 order
        @test e32 < e16
        @test order > 1.7                 # second order, within the usual slack
        @test e32 < 0.02                  # and small, with the interface in the core
    end

    @testset "a wrong rescaling would be caught" begin
        # Not a hypothetical: the two candidate prefactors differ by four orders
        # of magnitude at production τ. Half the right α misreports the strain
        # rate by half, which this test sees immediately.
        rg = TwoGrid(T, dims, lo, hi, τc)
        ρ, u, γc = 1.0, (0.02, 0.0, 0.0), 4e-4
        shear!(rg.coarse, ρ, u, γc, rg.τc)
        shear!(rg.prev, ρ, u, γc, rg.τc)
        bad = TwoGrid{T}(rg.coarse, rg.fine, rg.τc, rg.τf, rg.lo, rg.hi, rg.ratio,
                         rg.α / 2, rg.prev)
        interface_fill!(bad, 1.0)
        @test shear_of(bad.fine, 2, 6, 7, ρ, u, bad.τf) ≈ γc / 4 rtol = 1e-12
        @test !isapprox(shear_of(bad.fine, 2, 6, 7, ρ, u, bad.τf), γc / 2; rtol = 0.1)
    end
end
