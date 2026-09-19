@testset "a chain of levels" begin
    T = Float64
    τc = 0.7
    cdims = (32, 32, 32)
    geom = BaseballGeometry()

    @testset "one pair is the two-level cycle, exactly" begin
        # The chain is supposed to *be* the existing mechanism repeated, so at
        # length one it has to reproduce it bit for bit. Anything less and the
        # deeper chain is a second implementation of the same physics, which is
        # the thing this design was chosen to avoid.
        lo, hi = (9, 9, 9), (24, 24, 24)
        stream(x, y, z) = (1.0, -0.05, 0.0, 0.0)
        force = (2e-6, 0.0, 0.0)
        spin = (0.0, 0.0, 1.5e-3)

        rg = TwoGrid(T, cdims, lo, hi, τc)
        init_refined!(rg, stream)
        w1 = RotatingWall(geom, size(rg.fine)[1:3], geom.radius / 6)
        F1, M1 = refine_cycle_walls!(rg, w1; force = force, spin = spin)   # fine-level spin

        # The chain takes the *base* level's spin and scales it down per level;
        # `refine_cycle_walls!` takes its fine level's, already scaled (that is
        # what `advance_flow!` does for it). So the same physical spin is passed
        # differently to the two, which is the point of the convention.
        ch = GridChain(T, cdims, [(lo, hi)], τc)
        init_chain!(ch, stream)
        w2 = RotatingWall(geom, size(finest_grid(ch))[1:3], geom.radius / 6)
        F2, M2 = chain_cycle_walls!(ch, w2; force = force, spin = spin .* 2)

        @test base_grid(ch) == rg.coarse
        @test finest_grid(ch) == rg.fine
        @test collect(F2) == collect(F1)
        @test collect(M2) == collect(M1)
    end

    @testset "the levels share their arrays" begin
        # Level k's fine array *is* level k+1's coarse one: that is what makes
        # the transfers work unchanged. A copy would leave the pairs solving
        # different flows that agree only by accident.
        ch = GridChain(T, cdims, [((9, 9, 9), (24, 24, 24)), ((8, 8, 8), (24, 24, 24))], τc)
        @test length(ch) == 3
        @test ch.levels[2].coarse === ch.levels[1].fine
        @test levels(ch)[2] === ch.levels[1].fine
        @test base_grid(ch) === ch.levels[1].coarse
        @test finest_grid(ch) === ch.levels[2].fine
    end

    @testset "each level halves the step and doubles tau minus a half" begin
        # Acoustic scaling (§6.5.1): nu is fixed in physical terms, so
        # tau - 1/2 doubles at every level, and a level runs twice its parent's
        # steps for the same physical time.
        ch = GridChain(T, cdims, [((9, 9, 9), (24, 24, 24)), ((8, 8, 8), (24, 24, 24))], τc)
        τs = level_taus(ch)
        @test τs[1] == τc
        for k in 2:length(τs)
            @test τs[k] - 0.5 ≈ 2 * (τs[k - 1] - 0.5) rtol = 1e-14
        end
        s = chain_sizes(ch)
        @test s.steps == [2, 4, 8]
        @test length(s.nodes) == 3
    end

    @testset "a uniform stream survives three levels" begin
        # The interface and the restriction both have to leave a flow they
        # cannot improve on alone. With no body and no forcing, a uniform stream
        # is exactly that, at every level.
        # With no body in it: a ball would disturb the levels within a few
        # cycles — sound crosses the deepest level in far less time than it
        # crosses the base one — so there would be nowhere left to compare.
        u = -0.05
        ch = GridChain(T, cdims, [((9, 9, 9), (24, 24, 24)), ((8, 8, 8), (24, 24, 24))], τc)
        init_chain!(ch, (x, y, z) -> (1.0, u, 0.0, 0.0))
        before = [copy(a) for a in levels(ch)]
        for _ in 1:4
            chain_cycle_walls!(ch, nothing)
        end
        for (n, a) in enumerate(levels(ch))
            @test maximum(abs.(a .- before[n])) < 1e-12
        end
    end

    @testset "force climbs the levels by the right power" begin
        # Force carries ρL⁴/T² and torque ρL⁵/T², so each level climbed is m²
        # and m³. Getting the torque's power wrong would misreport spin decay by
        # exactly the size of the signal it is trying to measure.
        F = (1.0, 2.0, 3.0)
        M = (4.0, 5.0, 6.0)
        @test collect(chain_force(F, 2, 1)) == collect(coarse_force(F, 2))
        @test collect(chain_torque(M, 2, 1)) == collect(coarse_torque(M, 2))
        @test collect(chain_force(F, 2, 2)) == collect(F) ./ 16
        @test collect(chain_torque(M, 2, 2)) == collect(M) ./ 64
    end

    @testset "a vortex crosses two interfaces at second order" begin
        # The two-level version of this (`test_refine.jl`) is what says the
        # interface and the rescaling are right. Repeating it through a chain is
        # what says *repeating* them is right: each level is initialised from
        # the analytic solution, run, and judged against it on the deepest
        # level, so an error in how the levels are placed or in how time is
        # nested shows up as a loss of order rather than as a small offset.
        function vortex_error(n, cycles)
            τ = 0.5 + 0.3 * n / 32                   # nu_lattice ∝ n holds Re fixed
            tg = TaylorGreen(u0 = 0.02, n = n, ν = viscosity_from_tau(τ))
            nz = 16
            p1 = ((n ÷ 4 + 1, n ÷ 4 + 1, 3), (3n ÷ 4, 3n ÷ 4, 14))
            # In level 2's indices: the middle half of the patch, which puts the
            # second interface inside the first one's refined region and the
            # vortex core inside both.
            # Not `f2`: `3f2` is a Float32 literal, so `3f2[1]` indexes 300.0f0.
            mid = size(GridChain(T, (n, n, nz), [p1], τ).levels[1].fine)[1:3]
            p2 = ((mid[1] ÷ 4, mid[2] ÷ 4, 3),
                  (3 * mid[1] ÷ 4, 3 * mid[2] ÷ 4, mid[3] - 2))
            field = (x, y, z) -> begin
                ux, uy, uz = BBL.velocity(tg, x - 1, y - 1, 0.0)
                (BBL.density(tg, x - 1, y - 1, 0.0), ux, uy, uz)
            end

            ch = GridChain(T, (n, n, nz), [p1, p2], τ)
            init_chain!(ch, field)
            for _ in 1:cycles
                chain_cycle_walls!(ch, nothing; operator = :bgk)
            end

            t = 2 * cycles                            # base steps
            deep = finest_grid(ch)
            origin, h = level_origin(ch, length(ch))
            nfx, nfy, nfz = size(deep, 1), size(deep, 2), size(deep, 3)
            num = 0.0
            den = 0.0
            for c in 3:(nfz - 2), b in 5:(nfy - 4), a in 5:(nfx - 4)
                x = origin[1] + (a - 1) * h
                y = origin[2] + (b - 1) * h
                _, ux, uy, uz = node_macroscopic(deep, a, b, c, T)
                ex, ey, _ = BBL.velocity(tg, x - 1, y - 1, t)
                num += (ux - ex)^2 + (uy - ey)^2 + uz^2
                den += ex^2 + ey^2
            end
            return sqrt(num / den)
        end

        e16 = vortex_error(16, 12)
        e32 = vortex_error(32, 24)
        order = log2(e16 / e32)
        @info "three-level Taylor-Green" e16 e32 order
        @test e32 < e16
        @test order > 1.7
        @test e32 < 0.02
    end

    @testset "it refuses a patch that does not fit" begin
        @test_throws ArgumentError GridChain(T, cdims, [], τc)
        @test_throws ArgumentError GridChain(T, cdims, [((1, 1, 1), (24, 24, 24))], τc)
        # The second patch is measured in the first patch's index space, so a
        # box that would fit the base grid can still overflow the patch.
        @test_throws ArgumentError GridChain(T, cdims,
                                             [((9, 9, 9), (24, 24, 24)),
                                              ((2, 2, 2), (40, 40, 40))], τc)
    end
end
