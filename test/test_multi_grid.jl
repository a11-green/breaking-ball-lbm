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

    @testset "a ball on the deepest level" begin
        geom2 = BaseballGeometry()
        # Two patches, each centred, with the ball six nodes in radius on the
        # deepest level — the same shape the two-level tests use, one deeper.
        p1 = ((9, 9, 9), (24, 24, 24))
        probe = GridChain(T, cdims, [p1], τc)
        f1 = size(probe.levels[1].fine)[1:3]
        p2 = ((f1[1] ÷ 4, f1[2] ÷ 4, f1[3] ÷ 4),
              (3 * f1[1] ÷ 4, 3 * f1[2] ÷ 4, 3 * f1[3] ÷ 4))
        ch = GridChain(T, cdims, [p1, p2], τc)
        w = RotatingWall(geom2, size(finest_grid(ch))[1:3], geom2.radius / 6)
        cf = ChainFlow(ch, w)
        init_chain_flow!(cf, (x, y, z) -> (1.0, -0.05, 0.0, 0.0))

        @testset "the footprint lands on the base grid" begin
            # The ball is cut on the deepest level; the base level has no
            # geometry of its own and must be told where the hole is, through
            # two ratios. A mapping error puts the hole somewhere the ball is
            # not, and the box mean then averages over the ball's interior.
            @test count(cf.base_solid) > 0
            @test flow_fluid_count(cf) == prod(cdims) - count(cf.base_solid)
            idx = findall(cf.base_solid)
            lo_s = ntuple(d -> minimum(x -> x[d], idx), 3)
            hi_s = ntuple(d -> maximum(x -> x[d], idx), 3)
            # A ball of six deepest nodes is 1.5 base nodes in radius, centred.
            mid = (cdims .+ 1) ./ 2
            @test all(abs.((lo_s .+ hi_s) ./ 2 .- mid) .<= 1)
            @test all(hi_s .- lo_s .<= 4)

            # Every solid base node maps to a solid deepest node, which is the
            # property the mapping is for.
            for c in idx
                d = deepest_index(ch, c[1], c[2], c[3])
                @test d !== nothing
                @test w.wall.kind[d...] == BBL.SOLID_NODE
            end
        end

        @testset "the loop's units come back in base terms" begin
            # Force gains m² per level climbed and torque m³, so two levels are
            # 16 and 64. Reported in the wrong power, the torque would be out by
            # the size of a real spin-decay signal.
            spin = (0.0, 0.0, 1.0e-3)
            F, M = advance_flow!(ch, cf, 2, τc; spin = spin)
            Fd, Md = chain_cycle_walls!(ch, w; spin = spin)   # deepest-level units
            @test all(isfinite, F) && all(isfinite, M)
            @test chain_force(Fd, 2, 2)[1] / Fd[1] ≈ 1 / 16 rtol = 1e-12
            @test chain_torque(Md, 2, 2)[1] / Md[1] ≈ 1 / 64 rtol = 1e-12
        end

        @testset "the sub-cycle shortens with depth" begin
            # The deepest level takes m^n steps per base step, so the seam
            # travels that much further per base step and the re-cut is due that
            # much sooner. At two levels it is a quarter of the one-level bound.
            # **The same physical ball**, which is the comparison that means
            # anything: the deepest level is twice as fine, so the same sphere
            # is six of its nodes and three of the pair's fine ones. Holding
            # `radius_nodes` fixed instead would compare two different balls,
            # and the two effects — half the angle per step, twice the radius in
            # nodes — would cancel and say the sub-cycle does not shorten.
            spin = (0.0, 0.0, 2.0e-3)
            deep = max_substeps(cf, spin, 0.25)

            rg = TwoGrid(T, cdims, p1[1], p1[2], τc)
            rf = RefinedFlow(rg, RotatingWall(geom2, size(rg.fine)[1:3], geom2.radius / 3))
            shallow = max_substeps(rf, spin, 0.25)
            @test deep <= shallow ÷ 2 + 2         # even numbers, so allow the rounding
            @test deep >= 2
        end

        @testset "a re-cut moves the footprint with it" begin
            before = copy(cf.base_solid)
            q = BBL.Quat{T}(cos(0.4), 0.0, 0.0, sin(0.4))
            fresh = maybe_recut!(ch, cf, q, (0.0, 0.0, 1.0e-3), 0.0)
            @test fresh >= 0
            @test flow_fluid_count(cf) == prod(cdims) - count(cf.base_solid)
            # The sphere is rotationally symmetric, so the footprint of a
            # *seamed* ball may move by a node or two but cannot move far.
            @test count(xor.(before, cf.base_solid)) <= 8
        end

        @testset "the coupled loop flies it" begin
            # The chain answers the same three questions as every other backend,
            # so `couple_step!` should not know how deep it is. If it does, the
            # depth has leaked into the loop and the next level would leak again.
            units = LatticeUnits(T; nodes_per_diameter = 6, speed = 39.0,
                                 lattice_speed = 0.05, ν = 39.0 * 0.0748 / 40)
            props = BaseballProperties(T)
            ball = BallState(T; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                             spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
            run = PitchRun(units, props; substeps = 2, control_time = 100)
            st = PitchState(ball)

            ch2 = GridChain(T, cdims, [p1, p2], τc)
            w2 = RotatingWall(geom2, size(finest_grid(ch2))[1:3], geom2.radius / 6)
            cf2 = ChainFlow(ch2, w2)
            init_chain_flow!(cf2, (x, y, z) -> (1.0, -0.05, 0.0, 0.0))

            for _ in 1:4
                couple_step!(ch2, run, st, cf2)
            end
            @test all(isfinite, st.force)
            @test all(isfinite, st.torque)
            @test st.force[1] < 0                      # drag opposes the flight
            @test st.ball.v[1] < 39.0                  # and slows it
            @test st.ball.t > 0
            @test isfinite(couple_residual(run, st, cf2))
            @test flow_fluid_count(cf2) < prod(cdims)
        end

        @testset "it refuses a ball the patch cannot hold" begin
            big = RotatingWall(geom2, size(finest_grid(ch))[1:3], geom2.radius / 20)
            @test_throws ArgumentError ChainFlow(ch, big)
            wrong = RotatingWall(geom2, (9, 9, 9), geom2.radius / 6)
            @test_throws ArgumentError ChainFlow(ch, wrong)
        end
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
