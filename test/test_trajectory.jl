@testset "ball-following frame and 6DOF" begin
    props = BaseballProperties()
    g0 = (0.0, 0.0, 0.0)

    @testset "quaternion algebra" begin
        q = quat_from_axis_angle((1.0, 2.0, -0.5), 0.7)
        @test abs(q) ≈ 1 rtol = 1e-15

        v = (0.3, -1.2, 0.8)
        @test sqrt(sum(abs2, rotate(q, v))) ≈ sqrt(sum(abs2, v)) rtol = 1e-14
        @test collect(unrotate(q, rotate(q, v))) ≈ collect(v) rtol = 1e-14
        @test collect(rotate(one(Quat{Float64}), v)) ≈ collect(v)

        # A full turn is the identity, and the axis itself is fixed.
        full = quat_from_axis_angle((0.0, 0.0, 1.0), 2π)
        @test collect(rotate(full, v)) ≈ collect(v) atol = 1e-14
        @test collect(rotate(q, (1.0, 2.0, -0.5))) ≈ [1.0, 2.0, -0.5] rtol = 1e-14

        # Composition is matrix multiplication, in the same order.
        r = quat_from_axis_angle((0.0, 1.0, 0.3), -1.1)
        @test rotation_matrix(q * r) ≈ rotation_matrix(q) * rotation_matrix(r) rtol = 1e-13
    end

    @testset "spin integrates to the rotation it describes" begin
        # dq/dt = ½ ω ⊗ q with constant ω must give a rotation of |ω|t about ω̂.
        axis = (0.2, -1.0, 0.4)
        ω = spin_from_rpm(axis, 2708)
        @test spin_rpm(BallState(; spin = ω)) ≈ 2708 rtol = 1e-13

        s = BallState(; velocity = (1.0, 0.0, 0.0), spin = ω)
        dt = 1e-5
        for _ in 1:2000
            s = advance(s, props, BallisticAero(), dt; gravity = g0)
        end
        expected = quat_from_axis_angle(axis, sqrt(sum(abs2, ω)) * s.t)
        v = (0.7, 0.1, -0.3)
        @test collect(rotate(s.q, v)) ≈ collect(rotate(expected, v)) rtol = 1e-10
        @test abs(s.q) ≈ 1 rtol = 1e-14
    end

    @testset "free flight is a straight line, gravity a parabola" begin
        s0 = BallState(; position = (0.0, 0.1, 1.9), velocity = (39.0, -1.0, 2.0))
        s = s0
        for _ in 1:1000
            s = advance(s, props, BallisticAero(), 1e-4; gravity = g0)
        end
        @test collect(s.x) ≈ collect(s0.x .+ s.t .* s0.v) rtol = 1e-13
        @test collect(s.v) ≈ collect(s0.v) rtol = 1e-14

        # RK4 is exact for a constant acceleration, so this should hold to round-off.
        s = s0
        for _ in 1:1000
            s = advance(s, props, BallisticAero(), 1e-4)
        end
        drop = 0.5 * GRAVITY[3] * s.t^2
        @test s.x[3] ≈ s0.x[3] + s0.v[3] * s.t + drop rtol = 1e-12
    end

    @testset "pure drag matches its closed form" begin
        # dv/dt = -k v² has v = v0/(1 + k v0 t) and x = ln(1 + k v0 t)/k.
        aero = CoefficientAero(props; CD = 0.35, CL_slope = 0.0)
        v0 = 39.0
        k = 0.5 * AIR_DENSITY * props.area * 0.35 / props.mass
        s = BallState(; position = (0.0, 0.0, 0.0), velocity = (v0, 0.0, 0.0))
        for _ in 1:5000
            s = advance(s, props, aero, 1e-4; gravity = g0)
        end
        @test s.v[1] ≈ v0 / (1 + k * v0 * s.t) rtol = 1e-10
        @test s.x[1] ≈ log1p(k * v0 * s.t) / k rtol = 1e-10
    end

    @testset "spin decays exponentially when given a decay time" begin
        aero = CoefficientAero(props; CD = 0.0, CL_slope = 0.0, spin_decay = 12.0)
        ω0 = spin_from_rpm((0.0, 0.0, 1.0), 2708)
        s = BallState(; velocity = (39.0, 0.0, 0.0), spin = ω0)
        for _ in 1:5000
            s = advance(s, props, aero, 1e-4; gravity = g0)
        end
        @test s.ω[3] ≈ ω0[3] * exp(-s.t / 12.0) rtol = 1e-10
        @test spin_rpm(s) / 2708 > 0.95      # "a few percent over a pitch" (§1.5)
    end

    @testset "lattice units are self-consistent" begin
        u = LatticeUnits(; diameter = 0.0748, nodes_per_diameter = 40, speed = 39.0,
                         lattice_speed = 0.05)
        r = resolution_report(u)

        @test to_lattice_length(u, 0.0748) ≈ 40 rtol = 1e-13
        @test to_lattice_velocity(u, 39.0) ≈ 0.05 rtol = 1e-13
        @test r.reynolds ≈ 39.0 * 0.0748 / AIR_VISCOSITY rtol = 1e-12
        @test r.mach ≈ 0.05 / sqrt(CS2) rtol = 1e-14
        # Only 1e-9, and that is the point: τ - 1/2 is 3e-5 here, so recovering
        # ν from τ cancels away five significant digits. Nothing downstream may
        # reconstruct the viscosity that way — ν_lattice is stored for this reason.
        @test viscosity_from_tau(u.τ) ≈ u.ν_lattice rtol = 1e-9
        @test !isapprox(viscosity_from_tau(u.τ), u.ν_lattice; rtol = 1e-14)

        # Round trips, including the ρL⁴/T² and ρL⁵/T² that force and torque carry.
        for (to, from, x) in ((to_lattice_length, to_physical_length, 0.37),
                              (to_lattice_velocity, to_physical_velocity, 12.5),
                              (to_lattice_steps, to_physical_time, 0.43))
            @test from(u, to(u, x)) ≈ x rtol = 1e-12
        end
        F_lat = 0.6
        @test to_physical_force(u, F_lat) ≈ F_lat * AIR_DENSITY * u.dx^4 / u.dt^2 rtol = 1e-13
        @test to_physical_torque(u, F_lat) ≈ to_physical_force(u, F_lat) * u.dx rtol = 1e-13

        # A drag force of the right order must come back as a lattice force of
        # order one: if this drifts by orders of magnitude the conversion is wrong.
        drag = 0.5 * AIR_DENSITY * 39.0^2 * props.area * 0.35
        @test 0.1 < drag / to_physical_force(u, 1.0) < 10

        # τ sits just above 1/2 at pitch Reynolds numbers — the regime §2.3 exists for.
        @test 0 < r.tau_margin < 1e-4
        @test LatticeUnits(; nodes_per_diameter = 400).τ > u.τ   # finer grid, more margin
    end

    @testset "the frame's body force is minus the ball's acceleration" begin
        F = (-1.4, 0.2, 0.7)
        a = frame_acceleration(F, props)
        @test collect(a) ≈ collect(F ./ props.mass .+ GRAVITY) rtol = 1e-14
        @test collect(fluid_body_acceleration(F, props)) ≈ -collect(a) rtol = 1e-14

        # In free fall the air appears to accelerate upward at g, and nothing else.
        @test collect(fluid_body_acceleration((0.0, 0.0, 0.0), props)) ≈
              [0.0, 0.0, -GRAVITY[3]] rtol = 1e-14

        # The identity the scheme rests on: the same uniform body force that
        # stands for the frame's acceleration carries the far field to exactly
        # the velocity the next step's inflow condition will ask for.
        s = BallState(; velocity = (39.0, -0.4, 1.1),
                      spin = spin_from_rpm((0.0, -1.0, 0.0), 2400))
        aero = CoefficientAero(props)
        Fnow, _ = aero(s)
        dt = 1e-6
        s1 = advance(s, props, aero, dt)
        du = (freestream_velocity(s1) .- freestream_velocity(s)) ./ dt
        @test collect(du) ≈ collect(fluid_body_acceleration(Fnow, props)) rtol = 1e-4
    end

    @testset "frame quantities convert to lattice units" begin
        u = LatticeUnits(; speed = 39.0, lattice_speed = 0.05)
        s = BallState(; velocity = (39.0, 0.0, 0.0),
                      spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
        @test collect(lattice_freestream(u, s)) ≈ [-0.05, 0.0, 0.0] rtol = 1e-13

        F = (-1.4, 0.0, 0.0)
        @test collect(lattice_body_force(u, F, props)) ≈
              collect(to_lattice_acceleration(u, fluid_body_acceleration(F, props))) rtol = 1e-14
        # A 2,708 rpm ball turns well under a milliradian per lattice step.
        @test 0 < abs(lattice_spin(u, s)[3]) < 1e-3

        # Surface speed from spin, in lattice units, must stay well subsonic.
        @test abs(lattice_spin(u, s)[3]) * u.nodes_per_diameter / 2 < 0.1
    end

    @testset "orientation drift measures surface travel" begin
        q0 = quat_from_axis_angle((0.0, 0.0, 1.0), 0.3)
        q1 = quat_from_axis_angle((0.0, 0.0, 1.0), 0.3 + 0.02)
        @test orientation_drift(q0, q1, 20.0) ≈ 0.02 * 20 rtol = 1e-12
        @test orientation_drift(q0, q0, 20.0) ≈ 0 atol = 1e-12
        # The shorter way round: 2π - ε reads as ε, not as 2π.
        q2 = quat_from_axis_angle((0.0, 0.0, 1.0), 0.3 - 0.02)
        @test orientation_drift(q0, q2, 20.0) ≈ 0.02 * 20 rtol = 1e-12
    end
end

@testset "pitch movement" begin
    props = BaseballProperties()

    @testset "no spin means no break" begin
        s0 = BallState(; position = (0.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0))
        m = pitch_metrics(s0, props, CoefficientAero(props))
        @test abs(m.pfx_horizontal) < 1e-12
        @test abs(m.pfx_vertical) < 1e-12
        @test abs(m.induced_vertical) < 1e-12

        # With drag and gravity only, the drop away from a straight line is
        # the full gravitational drop.
        @test m.total_vertical < 0
        @test m.flight_time > (PLATE_DISTANCE - s0.x[1]) / 39.0    # drag makes it later
    end

    @testset "a ballistic pitch drops exactly ½gt²" begin
        aero = CoefficientAero(props; CD = 0.0, CL_slope = 0.0)
        s0 = BallState(; position = (0.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0))
        m = pitch_metrics(s0, props, aero)
        t = PLATE_DISTANCE / 39.0
        @test m.flight_time ≈ t rtol = 1e-12
        @test m.plate_speed ≈ sqrt(39.0^2 + (GRAVITY[3] * t)^2) rtol = 1e-12
        @test m.total_vertical ≈ 0.5 * GRAVITY[3] * t^2 rtol = 1e-10
        @test abs(m.induced_vertical) < 1e-12
    end

    @testset "Magnus points where the right-hand rule says" begin
        aero = CoefficientAero(props)
        base = (0.0, 0.0, 1.8)

        # Backspin (ω along -y) lifts; topspin drops.
        back = pitch_metrics(BallState(; position = base, velocity = (39.0, 0.0, 0.0),
                                       spin = spin_from_rpm((0.0, -1.0, 0.0), 2400)), props, aero)
        top = pitch_metrics(BallState(; position = base, velocity = (39.0, 0.0, 0.0),
                                      spin = spin_from_rpm((0.0, 1.0, 0.0), 2400)), props, aero)
        @test back.induced_vertical > 0
        @test top.induced_vertical ≈ -back.induced_vertical rtol = 0.02
        @test abs(back.induced_horizontal) < 1e-12

        # Spin about +z pushes toward +y, by ω × V̂.
        side = pitch_metrics(BallState(; position = base, velocity = (39.0, 0.0, 0.0),
                                       spin = spin_from_rpm((0.0, 0.0, 1.0), 2400)), props, aero)
        @test side.induced_horizontal > 0
        # Not exactly zero vertically: gravity tilts the velocity downward during
        # the flight, so ω × V̂ stops being purely horizontal. Small, and real.
        @test 0 < abs(side.induced_vertical) < 0.05 * side.induced_horizontal

        # Gyro spin — along the flight direction — makes no Magnus force at the
        # moment of release, which is the statement that can be made exactly.
        gyro0 = BallState(; position = base, velocity = (39.0, 0.0, 0.0),
                          spin = spin_from_rpm((1.0, 0.0, 0.0), 2400))
        F, M = aero(gyro0)
        @test F[1] < 0 && abs(F[2]) < 1e-15 && abs(F[3]) < 1e-15    # pure drag
        @test all(M .== 0)

        # Over the flight it picks up a little, for the same reason the sidespin
        # case does, but stays an order of magnitude below a transverse-spin pitch.
        gyro = pitch_metrics(gyro0, props, aero)
        @test hypot(gyro.induced_horizontal, gyro.induced_vertical) <
              0.1 * back.induced_vertical
        @test gyro.spin_parameter ≈ back.spin_parameter rtol = 1e-13   # but S is unchanged
    end

    @testset "break grows with spin and the definitions rank as expected" begin
        aero = CoefficientAero(props)
        pitch(rpm) = pitch_metrics(
            BallState(; position = (0.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                      spin = spin_from_rpm((0.0, -1.0, 0.0), rpm)), props, aero)

        slow, fast = pitch(1200), pitch(2400)
        @test fast.induced_vertical > slow.induced_vertical
        # C_L is linear in S, so doubling the spin roughly doubles the break.
        @test 1.9 < fast.induced_vertical / slow.induced_vertical < 2.1

        # The last 40 ft is about half the flight, and break grows like the
        # square of the remaining time, so pfx lands near a third of the
        # whole-flight figure. What matters for §8 is that the two are far
        # enough apart that using the wrong one would fail a 20% test.
        @test 0.3 < fast.pfx_vertical / fast.induced_vertical < 0.6

        # Total movement is the induced break less the gravitational drop.
        drop = fast.total_vertical - fast.induced_vertical
        @test drop < -0.8       # metres; gravity over roughly half a second
    end

    @testset "a sweeper sweeps" begin
        # WBC 2023 final, the last pitch to Trout (§8, V&V-5): 87.2 mph,
        # 2,708 rpm. The spin axis is not yet established, so this only checks
        # that a sweeper-like axis produces sweeper-like motion — horizontal
        # break dominating, with the ball dropping more than a fastball would.
        aero = CoefficientAero(props)
        axis = (0.15, 0.0, 1.0)      # mostly sidespin, slight gyro component
        s0 = BallState(; position = (0.0, 0.0, 1.75), velocity = (38.99, 0.0, 0.0),
                       spin = spin_from_rpm(axis, 2708))
        m = pitch_metrics(s0, props, aero)
        @test m.induced_horizontal > abs(m.induced_vertical)
        @test m.total_vertical < -0.5
        @test 0.2 < m.spin_parameter < 0.5          # the range §1.2 quotes
        @test 0.35 < m.flight_time < 0.55
    end
end

@testset "grid budget" begin
    b = grid_budget(; nodes_per_diameter = 40, domain_diameters = 8)
    @test b.edge == 320
    @test b.nodes == 320^3
    @test b.gib ≈ 320^3 * 112 / 2^30 rtol = 1e-12
    @test b.seam_cells ≈ 0.00079 / (0.0748 / 40) rtol = 1e-12

    # The design document claimed 40 points per diameter would put two or three
    # cells across the seam ridge. It puts less than half of one.
    @test b.seam_cells < 0.5

    # Seam and boundary layer resolve together: the ridge is one boundary layer
    # tall, which is what makes it a trip rather than a bump (§1.3).
    @test 0.9 < b.seam_cells / b.boundary_layer_cells < 1.15
    @test boundary_layer_thickness() < 0.00079      # and just under it

    # Domain width costs seam resolution at the third power.
    wide = grid_budget(; nodes_per_diameter = 40, domain_diameters = 8)
    tight = grid_budget(; nodes_per_diameter = 80, domain_diameters = 4)
    @test wide.nodes == tight.nodes
    @test tight.seam_cells ≈ 2 * wide.seam_cells rtol = 1e-12
    @test tight.blockage ≈ 2 * wide.blockage rtol = 1e-12

    # Refining raises the step count as well as the node count, so cost goes as
    # the fourth power of the resolution.
    fine = grid_budget(; nodes_per_diameter = 80, domain_diameters = 8)
    @test fine.hours / wide.hours ≈ 16 rtol = 1e-10
end
