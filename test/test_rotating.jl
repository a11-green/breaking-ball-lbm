@testset "geometry that turns with the ball" begin
    T = Float64
    geom = BaseballGeometry()
    N = 16                                  # nodes per diameter — small, for speed
    dims = (40, 40, 40)
    dx = geom.radius * 2 / N

    @testset "the windowed seam search matches the exhaustive one" begin
        # The window is an argument about the parameterisation (φ = u, so the
        # azimuth seeds the parameter); this is the check of it. Agreement is
        # only claimed near the seam, which is the only place the answer is used
        # for anything but a sign.
        seam = geom.seam
        poly = seam_polyline(seam, 8192)
        h = geom.seam_height
        worst_near = 0.0
        worst_below = 0.0
        near = 0
        rng = 12345
        for m in 1:4000
            rng = (1103515245 * rng + 12345) % 2147483648       # reproducible, no deps
            a = 2π * (rng / 2147483648)
            rng = (1103515245 * rng + 12345) % 2147483648
            c = 2 * (rng / 2147483648) - 1
            rng = (1103515245 * rng + 12345) % 2147483648
            r = geom.radius * (1 + 0.05 * (rng / 2147483648 - 0.3))
            s = sqrt(max(1 - c^2, 0.0))
            p = (r * s * cos(a), r * s * sin(a), r * c)

            exact = BBL.distance_to_seam(p, poly)
            fast = seam_distance(seam, p)
            # A restricted minimum can never be smaller — except by the chord
            # error of the polyline itself, which is what this bounds.
            worst_below = min(worst_below, fast - exact)
            if exact < 6h
                near += 1
                worst_near = max(worst_near, abs(fast - exact))
            end
        end
        @test near > 200                       # the sample actually covers the seam
        @test worst_near < 1e-4 * h            # exact, for every purpose here
        # The bound is the polyline's own chord sagitta: a restricted minimum
        # cannot undercut the true one, but it can undercut a chord approximation
        # to it, and by about R(Δu)²/8 at 8192 samples.
        @test worst_below > -1e-6 * geom.radius

        # And it is the same function, called the same way.
        p = (0.03, 0.01, 0.02)
        @test sdf(geom, p) ≈ sdf_exhaustive(geom, p) rtol = 1e-6
    end

    rw = RotatingWall(geom, dims, dx)

    @testset "the shell contains everything that can change" begin
        @test length(rw.shell) == count(>(Int32(0)), rw.slot)
        @test rw.nfluid + rw.fixed_solid + count(rw.solid) == prod(dims)
        @test rw.radius_nodes ≈ N / 2 rtol = 1e-12
        @test 0 < geom.seam_height / dx < 1     # sub-cell, as §6.5 says it always is

        # Nodes outside the shell must never acquire a solid neighbour, at any
        # orientation. The margin is a claim; this checks it.
        @test shell_is_sufficient(rw; samples = 8)
        @test_throws ArgumentError RotatingWall(geom, dims, dx; margin = 1.5)
    end

    @testset "the seam's own symmetry comes back exactly" begin
        # z/R = A sin(2u) with φ = u is invariant under u → u + π, which is a
        # half turn about z: the solid set maps to itself, so re-cutting there
        # has to reproduce the cut bit for bit. Nothing about the search, the
        # shell numbering or the bisection is allowed to break that.
        recut!(rw, one(Quat{T}))
        kind0, deltas0 = copy(rw.wall.kind), copy(rw.wall.deltas)

        recut!(rw, quat_from_axis_angle((0.0, 0.0, 1.0), π))
        @test rw.wall.kind == kind0
        @test rw.wall.deltas == deltas0          # exactly, not approximately

        # A quarter turn is a different ball: sin(2u + π) = −sin(2u) flips the
        # seam in z, so the geometry genuinely moves.
        recut!(rw, quat_from_axis_angle((0.0, 0.0, 1.0), π / 2))
        moved = count(!=(0.0), rw.wall.deltas .- deltas0)
        @test moved > 100
        @test maximum(abs.(rw.wall.deltas .- deltas0)) > 0.2
    end

    @testset "the identity cut is the cut build_wall_field makes" begin
        recut!(rw, one(Quat{T}); iterations = 40)
        ϕ = sdf_field(geom, dims, dx; center = (dims .+ 1) ./ 2) ./ dx
        fn = body_sdf(rw, one(Quat{T}))
        ref = build_wall_field(ϕ; sdf_fn = fn)

        @test count(==(BBL.SOLID_NODE), rw.wall.kind) == count(==(BBL.SOLID_NODE), ref.kind)
        worst = 0.0
        boundary = 0
        agree = true
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            a, b = rw.wall.kind[i, j, k], ref.kind[i, j, k]
            agree &= (a > 0) == (b > 0)          # the same nodes are boundary nodes
            a > 0 && b > 0 || continue
            boundary += 1
            for s in 1:27
                worst = max(worst, abs(rw.wall.deltas[s, a] - ref.deltas[s, b]))
            end
        end
        @test agree
        @test boundary > 100
        @test worst < 1e-9                       # the columns are numbered differently,
    end                                          # the wall fractions are not

    @testset "wall fractions carry the bisection's resolution and no more" begin
        # Each δ is a bisection result, so it is only defined to 2^-iterations.
        # This is worth pinning down because it sets what any *other*
        # implementation of the same cut — the CUDA kernel, say — can be asked
        # to match: last-bit differences in sin and cos flip the final
        # comparison, and the two answers then differ by exactly one bin. That
        # is agreement to the algorithm's full precision, not a discrepancy.
        q = quat_from_axis_angle((0.0, 0.0, 1.0), 0.37)
        recut!(rw, q; iterations = 20)
        coarse = copy(rw.wall.deltas)
        recut!(rw, q; iterations = 34)
        fine = copy(rw.wall.deltas)

        worst = maximum(abs.(coarse .- fine))
        @test worst <= 2.0^-20
        @test worst > 2.0^-34        # the bin is real, not a formality
        @test count(!=(0.0), coarse .- fine) > 10
    end

    @testset "single precision moves the wall by a ten-thousandth of a cell" begin
        # Same statement as above, across precisions rather than across depths.
        # In Float32 a bin at twenty halvings is eight ulps wide, so the two
        # bisections wander a few bins apart on most links — which is why the
        # bar for any second implementation is the wall's *position*, in cells,
        # rather than a count of bins.
        g32 = BaseballGeometry(; diameter = Float32(0.0748), seam_height = Float32(0.00079),
                               seam_amplitude = Float32(0.7))
        w32 = RotatingWall(g32, dims, Float32(dx))
        q64 = quat_from_axis_angle((0.0, 0.0, 1.0), 0.37)
        q32 = quat_from_axis_angle((0.0f0, 0.0f0, 1.0f0), 0.37f0)
        recut!(rw, q64)
        recut!(w32, q32)

        @test w32.wall.kind == rw.wall.kind
        worst = maximum(abs.(Float64.(w32.wall.deltas) .- rw.wall.deltas))
        @test worst < 1e-4                   # of a lattice spacing
        @test worst > 2.0^-20                # and more than one bin, as expected
    end

    @testset "a smooth sphere cannot notice being turned" begin
        smooth = BaseballGeometry(; seam_height = 0.0)
        sw = RotatingWall(smooth, dims, dx)
        recut!(sw, one(Quat{T}))
        kind0, deltas0 = copy(sw.wall.kind), copy(sw.wall.deltas)
        recut!(sw, quat_from_axis_angle((0.3, -1.0, 0.5), 0.9))
        @test sw.wall.kind == kind0
        @test sw.wall.deltas == deltas0
    end

    """Force on the ball after a short run at a given seam orientation."""
    function force_at(wall, angle; steps = 40, τ = 0.55)
        recut!(wall, quat_from_axis_angle((0.0, 0.0, 1.0), angle))
        s = LBMState{T}(dims..., τ; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
        g = to_cube_order!(similar(s.f), s.f)
        F, _ = aa_run_walls!(g, wall.wall, steps, τ; spin = (0.0, 0.0, 2e-3),
                             reduction = :mean)
        return F
    end

    @testset "turning the seam changes the force; turning a sphere does not" begin
        # The solver is deterministic, so two runs from the same initial state
        # differ only through the geometry. That makes this decisive rather than
        # statistical: any difference at all is the seam doing something.
        a = force_at(rw, 0.0)
        b = force_at(rw, 0.4)
        @test a[1] < 0 && b[1] < 0                       # still drag
        @test abs(b[2] - a[2]) > 0.05 * abs(a[2])        # and the side force moves

        smooth = RotatingWall(BaseballGeometry(; seam_height = 0.0), dims, dx)
        c = force_at(smooth, 0.0)
        d = force_at(smooth, 0.4)
        @test collect(c) == collect(d)                   # bit for bit
    end

    @testset "nodes the ridge uncovers are refilled" begin
        recut!(rw, one(Quat{T}))
        s = LBMState{T}(dims..., 0.55; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
        g = to_cube_order!(similar(s.f), s.f)

        fresh = 0
        for m in 1:12
            fresh += recut!(rw, quat_from_axis_angle((0.0, 0.0, 1.0), 0.08m), g,
                            (0.0, 0.0, 2e-3))
        end
        @test fresh > 0                          # the ridge really does sweep nodes
        @test all(isfinite, g)

        # Every fluid node has a sensible density: a refill that left scratch
        # behind would show up here long before it showed up as a diverged run.
        worst = 0.0
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            rw.wall.kind[i, j, k] == BBL.SOLID_NODE && continue
            ρ = sum(g[i, j, k, s] for s in 1:27)
            worst = max(worst, abs(ρ - 1))
        end
        @test worst < 0.05
    end

    @testset "re-cutting waits for the surface to have moved" begin
        recut!(rw, one(Quat{T}))
        before = rw.recuts
        g = zeros(T, dims..., 27)
        tiny = quat_from_axis_angle((0.0, 0.0, 1.0), 1e-6)
        @test maybe_recut!(g, rw, tiny, (0.0, 0.0, 0.0), 0.25) == 0
        @test rw.recuts == before

        big = quat_from_axis_angle((0.0, 0.0, 1.0), 0.5)
        maybe_recut!(g, rw, big, (0.0, 0.0, 0.0), 0.25)
        @test rw.recuts == before + 1
        @test rw.orientation.w ≈ big.w rtol = 1e-14

        # Static geometry ignores all of this.
        plain = build_wall_field(sphere_sdf_field(T, dims, 5.0))
        @test maybe_recut!(g, plain, big, (0.0, 0.0, 0.0), 0.25) == 0
    end

    @testset "how often the geometry has to be re-cut" begin
        # A 2,708 rpm ball at production resolution: the number that forces the
        # coupled loop's sub-cycle to be short (§4.4).
        units = LatticeUnits(; nodes_per_diameter = 40, speed = 39.0)
        ball = BallState(; velocity = (39.0, 0.0, 0.0),
                         spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
        prod_wall = RotatingWall(geom, (12, 12, 12), geom.radius * 2 / 40; margin = 2.0)
        drift = surface_drift_per_step(prod_wall, lattice_spin(units, ball))
        @test 0.005 < drift < 0.05
        @test 10 < 0.25 / drift < 40         # a quarter spacing every ~20 steps
    end
end
