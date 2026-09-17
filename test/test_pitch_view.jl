@testset "what a pitch viewer draws" begin
    T = Float64
    props = BaseballProperties()
    aero = CoefficientAero(props)
    release = (2.0, 0.0, 1.75)
    sweeper = BallState(; position = release, velocity = (38.99, 0.0, 0.0),
                        spin = spin_from_rpm((0.15, 0.0, 1.0), 2708))

    @testset "columns are the states, transposed" begin
        traj = simulate_trajectory(sweeper, props, aero)
        s = samples(traj)
        @test length(s) == length(traj)
        @test s.t[1] == traj[1].t && s.t[end] == traj[end].t
        @test s.x[end] ≈ traj[end].x[1] && s.z[end] ≈ traj[end].x[3]
        @test s.speed[1] ≈ speed(traj[1]) rtol = 1e-14
        @test abs(s.q[end].w) <= 1 + 1e-12          # still a unit quaternion
        @test issorted(s.t)
    end

    @testset "the gaps in the picture are the numbers in the table" begin
        # The figure exists to show that break depends on which trajectory is
        # subtracted. If the drawn gap and the reported coefficient could
        # disagree, the figure would be worse than no figure.
        r = break_references(sweeper, props, aero)
        m = pitch_metrics(sweeper, props, aero)

        @test r.actual.y[end] - r.pfx.y[end] ≈ m.pfx_horizontal rtol = 1e-6
        @test r.actual.z[end] - r.pfx.z[end] ≈ m.pfx_vertical atol = 1e-4
        @test r.actual.y[end] - r.induced.y[end] ≈ m.induced_horizontal rtol = 1e-6
        @test r.actual.y[end] - r.ballistic.y[end] ≈ m.total_horizontal rtol = 1e-6
        @test r.actual.z[end] - r.ballistic.z[end] ≈ m.total_vertical rtol = 1e-4

        # And the pfx reference really does branch forty feet out, not at release.
        x40 = PLATE_DISTANCE - PFX_SEGMENT
        @test r.pfx.x[1] ≈ x40 rtol = 1e-6
        @test r.induced.x[1] ≈ release[1] rtol = 1e-12
        # Which is why it is the smaller of the two.
        @test abs(m.pfx_horizontal) < abs(m.induced_horizontal)

        # The ballistic line is a line, sampled with the pitch so the two scrub
        # together.
        @test length(r.ballistic) == length(r.actual)
        @test r.ballistic.z[end] ≈ release[3] rtol = 1e-12       # no gravity in it
        mid = length(r.ballistic) ÷ 2
        @test r.ballistic.x[mid] ≈ release[1] + r.ballistic.t[mid] * 38.99 rtol = 1e-12
    end

    @testset "the seam is what makes a spinning ball look like one" begin
        geom = BaseballGeometry()
        centre = (1.0, 2.0, 3.0)
        line = seam_world(geom, one(Quat{T}), centre; samples = 64)
        @test length(line) == 65                     # closed: first point repeated
        @test line[1] == line[end]
        for p in line
            @test sqrt(sum(abs2, p .- centre)) ≈ geom.radius rtol = 1e-12
        end

        # Turning the ball moves the seam, and a half turn about z maps the seam
        # to itself — the same symmetry the re-cut is held to.
        half = quat_from_axis_angle((0.0, 0.0, 1.0), π)
        turned = seam_world(geom, half, centre; samples = 64)
        quarter = seam_world(geom, quat_from_axis_angle((0.0, 0.0, 1.0), π / 2),
                             centre; samples = 64)
        @test maximum(maximum(abs.(a .- b)) for (a, b) in zip(line, quarter)) > 0.01
        # z/R = A sin 2u is invariant under u → u + π, so a half turn about z
        # maps the seam to itself: every turned point still lies on the curve.
        @test all(minimum(maximum(abs.(t .- l)) for l in line) < 1e-9 for t in turned)

        scaled = seam_world(geom, one(Quat{T}), centre; samples = 16, scale = 10)
        @test sqrt(sum(abs2, scaled[1] .- centre)) ≈ 10 * geom.radius rtol = 1e-12
    end

    @testset "the spin axis is drawn through the centre" begin
        a, b = spin_axis_world(sweeper, 0.5)
        @test collect((a .+ b) ./ 2) ≈ collect(sweeper.x) rtol = 1e-12
        @test sqrt(sum(abs2, b .- a)) ≈ 0.5 rtol = 1e-12
        # A ball with no spin has no axis to draw.
        still = BallState(; position = release, velocity = (39.0, 0.0, 0.0))
        c, d = spin_axis_world(still, 0.5)
        @test c == d == still.x
    end

    @testset "the named pitches break the way their names say" begin
        fam = pitch_family()
        by = Dict(e.name => e for e in fam)
        @test length(fam) == length(PITCH_TYPES)

        @test by["4-seam fastball"].metrics.pfx_vertical > 0.2          # it rises
        @test by["12-6 curve"].metrics.pfx_vertical < -0.3              # it drops
        @test by["sweeper"].metrics.pfx_horizontal > 0.3                # toward +y
        @test by["2-seam / sinker"].metrics.pfx_horizontal < 0          # the other way
        # A gyroball spins about the flight direction, so it has almost nothing.
        g = by["gyroball"].metrics
        @test hypot(g.pfx_horizontal, g.pfx_vertical) <
              0.15 * abs(by["4-seam fastball"].metrics.pfx_vertical)

        for e in fam
            @test length(e.samples) > 100
            @test e.samples.x[end] >= PLATE_DISTANCE
            @test e.samples.speed[end] < e.spec.speed                   # drag
        end
    end

    @testset "scrubbing and the strike zone" begin
        s = samples(simulate_trajectory(sweeper, props, aero))
        @test at_time(s, s.t[1]) == 1
        @test at_time(s, s.t[end]) == length(s)
        @test at_time(s, -1.0) == 1
        @test at_time(s, 1e6) == length(s)
        mid = s.t[1] + (s.t[end] - s.t[1]) / 2
        @test abs(s.t[at_time(s, mid)] - mid) <= (s.t[end] - s.t[1]) / length(s)

        box = plate_box()
        @test box[1] == box[end]                                        # closed
        @test all(p -> p[1] ≈ PLATE_DISTANCE, box)
        @test maximum(p[2] for p in box) - minimum(p[2] for p in box) ≈ 2 * 0.2159
    end
end

@testset "coefficients along a trajectory" begin
    props = BaseballProperties()
    aero = CoefficientAero(props; CD = 0.35, CL_slope = 1.0)
    s0 = BallState(; position = (2.0, 0.0, 1.75), velocity = (38.99, 0.0, 0.0),
                   spin = spin_from_rpm((0.15, 0.0, 1.0), 2708))
    traj = simulate_trajectory(s0, props, aero)
    c = coefficient_series(traj, props, aero)

    @test length(c.CD) == length(traj)
    @test all(x -> x ≈ 0.35, c.CD)                     # the model's own input, returned
    # C_L is built from the *transverse* spin parameter — the component of ω
    # along the flight path makes no Magnus force — so it is below the full one
    # by exactly the gyro fraction of this axis.
    b0 = traj[1]
    S⊥ = sqrt(sum(abs2, transverse_spin(b0))) * props.radius / speed(b0)
    @test c.CL[1] ≈ S⊥ rtol = 1e-12
    @test c.CL[1] < spin_parameter(b0, props)
    # A coefficient model has no force out of the Magnus plane. The CFD will.
    @test maximum(abs, c.Cside) < 1e-12
    @test c.CL[end] > c.CL[1]                          # S grows as the ball slows
end
