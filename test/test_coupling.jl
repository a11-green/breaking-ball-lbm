@testset "tight coupling" begin
    T = Float64
    dims = (20, 20, 20)
    R = 3.5
    # Re = 40: steady and attached enough that a short run settles, so the
    # momentum balance below is testing the coupling rather than a transient.
    units = LatticeUnits(; nodes_per_diameter = 7, speed = 39.0, lattice_speed = 0.05,
                         ν = 39.0 * 0.0748 / 40)
    props = BaseballProperties()
    ϕ = sphere_sdf_field(T, dims, R)
    wall = build_wall_field(ϕ; sdf_fn = sphere_sdf_fn(dims, R))

    """A uniform stream at -0.05 in x, in cube-slot order."""
    function uniform(ux = -0.05, uy = 0.0, uz = 0.0, ρ = 1.0)
        s = LBMState{T}(dims..., units.τ; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (ρ, ux, uy, uz))
        return to_cube_order!(similar(s.f), s.f)
    end

    @testset "the box mean reads back what was put in" begin
        g = uniform(-0.05, 0.012, -0.007, 1.03)
        ρ̄, ū = mean_fluid_velocity(g, wall)
        @test ρ̄ ≈ 1.03 rtol = 1e-13
        @test collect(ū) ≈ [-0.05, 0.012, -0.007] rtol = 1e-12

        # Guo defines momentum as Σcf + F/2, so the body force belongs in the mean.
        F = (2e-3, -1e-3, 5e-4)
        _, ūF = mean_fluid_velocity(g, wall, F)
        @test collect(ūF .- ū) ≈ collect(F ./ 2 ./ 1.03) rtol = 1e-10

        # Solid nodes are excluded, not counted as quiescent fluid.
        @test fluid_node_count(wall) == count(!=(BBL.SOLID_NODE), wall.kind)
        @test fluid_node_count(wall) < prod(dims)
    end

    @testset "a weighted region measures what the box mean cannot" begin
        # The box mean is fixed by the mass flux through any plane, so it says
        # nothing about the profile. A region mean is what reads the profile —
        # and the two have to agree when the region is the whole fluid.
        g = uniform(-0.05, 0.012, -0.007, 1.03)
        fluid = T[wall.kind[i, j, k] == BBL.SOLID_NODE ? 0 : 1
                  for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        ρ̄, ū = mean_fluid_velocity(g, wall)
        ρ̄r, ūr = region_mean_velocity(g, fluid)
        @test ρ̄r ≈ ρ̄ rtol = 1e-13
        @test collect(ūr) ≈ collect(ū) rtol = 1e-12

        # The force enters as Guo's F/2, the same as in the box mean.
        F = (2e-3, -1e-3, 5e-4)
        _, ūF = region_mean_velocity(g, fluid, F)
        @test collect(ūF .- ūr) ≈ collect(F ./ 2 ./ 1.03) rtol = 1e-10

        # A profile the box mean is blind to: one slab slow, one fast, with the
        # mass flux — and so the box mean — unchanged.
        slow, fast = -0.03, -0.07
        s2 = LBMState{T}(dims..., units.τ; lattice = D3Q27())
        init_equilibrium!(s2, (i, j, k) -> (1.0, i <= dims[1] ÷ 2 ? slow : fast, 0.0, 0.0))
        g2 = to_cube_order!(similar(s2.f), s2.f)
        ones3 = ones(T, dims)
        _, ūall = region_mean_velocity(g2, ones3)
        @test ūall[1] ≈ (slow + fast) / 2 rtol = 1e-12

        half = zeros(T, dims)
        half[1:dims[1]÷2, :, :] .= 1
        _, ūslow = region_mean_velocity(g2, half)
        @test ūslow[1] ≈ slow rtol = 1e-12
        @test ūslow[1] / ūall[1] ≈ 0.6 rtol = 1e-12

        # A disc on the axis, which is the shape the sweep actually samples.
        c = (dims .+ 1) ./ 2
        disc = T[(i == 7 && (j - c[2])^2 + (k - c[3])^2 <= 16) ? 1 : 0
                 for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        @test sum(disc) > 0
        _, ūdisc = region_mean_velocity(g2, disc)
        @test ūdisc[1] ≈ slow rtol = 1e-12

        @test_throws DimensionMismatch region_mean_velocity(g, zeros(T, 3, 3, 3))
        # An empty region is a zero, not a division by zero.
        @test region_mean_velocity(g, zeros(T, dims))[1] == 0
    end

    @testset "the mean survives single precision" begin
        # The controller acts on target − mean, which at a production grid is a
        # fraction of a percent of the mean. Accumulating the sum in Float32
        # would round away that much on its own, so the reduction is in double
        # whatever the solver's precision — and a Float32 grid has to return the
        # mean to far better than Float32 summation would manage.
        u32 = LatticeUnits(Float32; nodes_per_diameter = 7, speed = 39.0,
                           lattice_speed = 0.05, ν = 39.0 * 0.0748 / 40)
        wall32 = build_wall_field(sphere_sdf_field(Float32, dims, R);
                                  sdf_fn = sphere_sdf_fn(dims, R))
        s32 = LBMState{Float32}(dims..., u32.τ; lattice = D3Q27())
        init_equilibrium!(s32, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
        g32 = to_cube_order!(similar(s32.f), s32.f)
        _, ū32 = mean_fluid_velocity(g32, wall32)
        @test ū32[1] ≈ -0.05f0 rtol = 1e-6
        @test abs(ū32[2]) < 1e-8 && abs(ū32[3]) < 1e-8
    end

    @testset "the coupled loop with the faces open" begin
        # The ball is off-centre so there is run-up ahead of it and wake behind,
        # which is the layout an inlet and an outlet are for.
        long = (40, 20, 20)
        centre = (28.0, 10.5, 10.5)
        wall_o = build_wall_field(sphere_sdf_field(T, long, R; center = centre);
                                 sdf_fn = sphere_sdf_fn(long, R; center = centre))
        ch = OpenChannel{T}()
        @test open_is_clear(wall_o, ch)

        function loop(chan, cycles)
            ls = LBMState{T}(long..., units.τ; lattice = D3Q27())
            init_equilibrium!(ls, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
            gg = to_cube_order!(similar(ls.f), ls.f)
            stt = PitchState(BallState(T; position = (2.0, 0.0, 1.8),
                                       velocity = (39.0, 0.0, 0.0),
                                       spin = spin_from_rpm((0.0, 0.0, 1.0), 2708)))
            rr = PitchRun(units, props; substeps = 10, control_time = 200,
                          channel = chan)
            for _ in 1:cycles
                couple_step!(gg, rr, stt, wall_o)
            end
            return gg, stt, rr
        end

        _, open_st, open_run = loop(ch, 8)
        _, closed_st, closed_run = loop(nothing, 8)

        # The controller is switched off, not merely quiet: with an inlet the
        # stream is stated at the face, so there is nothing to integrate.
        @test all(open_st.control .== 0)
        @test open_st.integral == (0.0, 0.0, 0.0)
        @test any(closed_st.control .!= 0)

        # The ball is still being flown by the same force path.
        @test open_st.force[1] != 0
        @test sign(open_st.force[1]) == sign(closed_st.force[1])
        @test open_st.ball.t ≈ closed_st.ball.t

        # And the residual reports the quantity that exists in each case. With
        # open faces the controller's balance is not one of them, so what comes
        # back is the free stream's drift from what the inlet states — finite,
        # small, and not a momentum balance (`couple_residual`).
        drift = couple_residual(open_run, open_st, wall_o)
        @test isfinite(drift)
        @test 0 <= drift < 0.5
        target = BBL.lattice_freestream(open_run.units, open_st.ball)
        @test drift ≈ sqrt(sum(abs2, open_st.mean_velocity .- target)) /
                      sqrt(sum(abs2, target))
        @test isfinite(couple_residual(closed_run, closed_st, wall_o))

        # (The refined path's refusal is checked where its fixtures live, in
        # test_refined_flow.jl.)
    end

    @testset "the coupled loop barely notices single precision" begin
        # Not a tolerance chosen to pass: if the loop were precision-sensitive
        # the control signal would be the first thing to go, since it is a small
        # difference of large sums.
        function loop(::Type{S}, cycles) where {S}
            un = LatticeUnits(S; nodes_per_diameter = 7, speed = 39.0,
                              lattice_speed = 0.05, ν = 39.0 * 0.0748 / 40)
            pr = BaseballProperties(S)
            w = build_wall_field(sphere_sdf_field(S, dims, R);
                                 sdf_fn = sphere_sdf_fn(dims, R))
            ls = LBMState{S}(dims..., un.τ; lattice = D3Q27())
            init_equilibrium!(ls, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
            gg = to_cube_order!(similar(ls.f), ls.f)
            stt = PitchState(BallState(S; position = (2.0, 0.0, 1.8),
                                       velocity = (39.0, 0.0, 0.0),
                                       spin = spin_from_rpm((0.0, 0.0, 1.0), 2708)))
            rr = PitchRun(un, pr; substeps = 10, control_time = 200)
            for _ in 1:cycles
                couple_step!(gg, rr, stt, w)
            end
            return stt
        end

        a, b = loop(Float64, 6), loop(Float32, 6)
        @test b.mean_velocity[1] ≈ a.mean_velocity[1] rtol = 1e-6
        @test b.control[1] ≈ a.control[1] rtol = 1e-5
        @test b.force[1] ≈ a.force[1] rtol = 1e-6
        @test b.ball.v[1] ≈ a.ball.v[1] rtol = 1e-6
        @test b.ball.v[2] ≈ a.ball.v[2] rtol = 1e-5
    end

    @testset "force converts to lattice units and back" begin
        for F in (1.0, -4.78, 1e-3)
            @test to_physical_force(units, to_lattice_force(units, F)) ≈ F rtol = 1e-12
        end
        @test collect(to_lattice_force(units, (1.0, -2.0, 0.5))) ≈
              [to_lattice_force(units, x) for x in (1.0, -2.0, 0.5)]
    end

    @testset "a run rejects settings it cannot honour" begin
        @test_throws ArgumentError PitchRun(units, props; substeps = 21)
        @test_throws ArgumentError PitchRun(units, props; substeps = 20, control_time = 20)
    end

    @testset "the mean reduction is the mean, exactly" begin
        # Averaging over twenty steps has to equal averaging ten successive
        # two-step averages — the definition, checked rather than assumed. The
        # ratio to the final step's force is no test at all: over the acoustic
        # transient of a startup the force changes sign, so the two disagree in
        # sign as well as size, which is precisely why the coupled loop averages.
        F = (1e-6, 0.0, 0.0)
        g1 = uniform(); g2 = copy(g1); g3 = copy(g1)
        f_mean, t_mean = aa_run_walls!(g1, wall, 20, units.τ; force = F, reduction = :mean)

        acc_f = (0.0, 0.0, 0.0); acc_t = (0.0, 0.0, 0.0)
        local f_last, t_last
        for _ in 1:10
            f, t = aa_run_walls!(g2, wall, 2, units.τ; force = F, reduction = :mean)
            acc_f = acc_f .+ f; acc_t = acc_t .+ t
            f_last, t_last = aa_run_walls!(g3, wall, 2, units.τ; force = F, reduction = :last)
        end
        @test collect(acc_f ./ 10) ≈ collect(f_mean) rtol = 1e-12
        @test collect(acc_t ./ 10) ≈ collect(t_mean) rtol = 1e-12
        @test g1 ≈ g2 rtol = 1e-14          # the reduction does not touch the flow
        @test g1 ≈ g3 rtol = 1e-14

        @test_throws ArgumentError aa_run_walls!(g1, wall, 2, units.τ; reduction = :both)
    end

    run = PitchRun(units, props; substeps = 20, control_time = 1000)

    @testset "spin-up holds the free stream and leaves the trajectory alone" begin
        g = uniform(-0.04)          # deliberately 20% below the target
        st = PitchState(BallState(; velocity = (39.0, 0.0, 0.0),
                                  spin = spin_from_rpm((0.0, 0.0, 1.0), 2708)))
        x0, v0, q0 = st.ball.x, st.ball.v, st.ball.q
        target = lattice_freestream(units, st.ball)

        couple_step!(g, run, st, wall; frozen = true)
        first_gap = abs(st.mean_velocity[1] - target[1])
        res = spin_up!(g, run, st, wall; cycles = 40)

        @test collect(st.ball.x) == collect(x0)         # frozen means frozen
        @test collect(st.ball.v) == collect(v0)
        @test st.ball.t > 0                             # but time passes
        @test abs(st.ball.q.w - q0.w) > 1e-6            # and the seam turns
        @test st.steps == 41 * run.substeps

        @test abs(st.mean_velocity[1] - target[1]) < first_gap    # the controller works
        @test abs(st.mean_velocity[1] - target[1]) < 0.15 * abs(target[1])
        @test isfinite(res)
    end

    @testset "the controller supplies the momentum the sphere removes" begin
        # Two quantities by different routes: the uniform body force the
        # controller has settled on, times the fluid volume, against the
        # momentum-exchange sum over the surface.
        g = uniform()
        st = PitchState(BallState(; velocity = (39.0, 0.0, 0.0)))
        spin_up!(g, run, st, wall; cycles = 120)
        supplied = st.control[1] * fluid_node_count(wall)
        removed = to_lattice_force(units, st.force[1])
        @test sign(supplied) == sign(removed)
        @test 0.5 < supplied / removed < 1.6
        @test couple_residual(run, st, wall) < 0.6
    end

    @testset "symmetry and the Magnus direction" begin
        function settle(spin; cycles = 50)
            g = uniform()
            st = PitchState(BallState(; velocity = (39.0, 0.0, 0.0), spin = spin))
            spin_up!(g, run, st, wall; cycles = cycles)
            return st
        end

        plain = settle((0.0, 0.0, 0.0))
        @test plain.force[1] < 0                        # drag opposes the flow
        @test abs(plain.force[2]) < 1e-12               # and nothing else survives
        @test abs(plain.force[3]) < 1e-12               #   the symmetry
        @test all(abs.(plain.torque) .< 1e-12)

        # ω × V̂ with ω along +z and flight along +x points along +y.
        up = settle(spin_from_rpm((0.0, 0.0, 1.0), 2708))
        down = settle(spin_from_rpm((0.0, 0.0, -1.0), 2708))
        @test up.force[2] > 0
        @test down.force[2] < 0
        @test up.force[2] ≈ -down.force[2] rtol = 1e-6          # antisymmetric in ω
        @test up.force[1] ≈ down.force[1] rtol = 1e-6           # drag is not
        @test abs(up.force[2]) > 0.02 * abs(up.force[1])        # and it is not noise

        # Spin drags on the fluid, so the fluid drags back: the torque opposes ω.
        @test up.torque[3] < 0
        @test down.torque[3] > 0
    end

    @testset "flying: the frame identity holds through the whole loop" begin
        g = uniform()
        st = PitchState(BallState(; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                                  spin = spin_from_rpm((0.0, 0.0, 1.0), 2708)))
        spin_up!(g, run, st, wall; cycles = 30)

        # One live cycle, checked against the identity the design rests on:
        # the uniform body force must equal d(u∞)/dt exactly, not approximately.
        before = lattice_freestream(units, st.ball)
        sample = st.force
        body = couple_step!(g, run, st, wall)
        after = lattice_freestream(units, st.ball)
        a_frame = lattice_body_force(units, sample, props)
        @test collect(after .- before) ≈ collect(run.substeps .* a_frame) rtol = 1e-11
        @test collect(body .- st.control) ≈ collect(a_frame) rtol = 1e-12
    end

    @testset "flying: drag slows the ball and gravity drops it" begin
        g = uniform()
        st = PitchState(BallState(; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                                  spin = spin_from_rpm((0.0, 0.0, 1.0), 2708)))
        spin_up!(g, run, st, wall; cycles = 30)
        v0 = st.ball.v
        fly!(g, run, st, wall; cycles = 40)

        @test st.ball.v[1] < v0[1]                  # drag
        @test st.ball.v[3] < 0                      # gravity
        @test st.ball.v[2] > 0                      # Magnus, in the ω × V̂ direction
        @test st.ball.x[1] > 2.0                    # and it is going somewhere
        @test st.ball.x[3] < 1.8

        # The callback can stop the loop, which is how a caller ends at the plate.
        n = 0
        fly!(g, run, st, wall; cycles = 100, callback = _ -> (n += 1) < 3)
        @test n == 3
    end

    @testset "flying with a seam that turns" begin
        # End to end: a seamed ball whose geometry is re-cut as it spins, driven
        # by the same loop. What this checks is that the pieces fit — the re-cut
        # fires on the right schedule, the refill leaves a usable flow, and the
        # trajectory still behaves.
        geom = BaseballGeometry()
        N = 10
        rdims = (24, 24, 24)
        rdx = geom.radius * 2 / N
        runits = LatticeUnits(; nodes_per_diameter = N, speed = 39.0,
                              lattice_speed = 0.05, ν = 39.0 * 0.0748 / 60)
        rw = RotatingWall(geom, rdims, rdx)

        ball = BallState(; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                         spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
        spin_lat = lattice_spin(runits, ball)

        # Size the sub-cycle from the geometry, not by guessing.
        nsub = max_substeps(rw, spin_lat, 0.25)
        @test iseven(nsub) && nsub >= 2
        @test surface_drift_per_step(rw, spin_lat) * nsub <= 0.25

        rrun = PitchRun(runits, BaseballProperties(); substeps = nsub,
                        control_time = 40 * nsub, recut_drift = 0.25)
        rs = LBMState{T}(rdims..., runits.τ; lattice = D3Q27())
        init_equilibrium!(rs, (i, j, k) -> (1.0, -0.05, 0.0, 0.0))
        rg = to_cube_order!(similar(rs.f), rs.f)
        st = PitchState(ball)

        cuts0 = rw.recuts
        for _ in 1:14
            couple_step!(rg, rrun, st, rw)
        end

        @test rw.recuts > cuts0                     # the seam did move
        @test all(isfinite, rg)
        @test st.ball.v[1] < 39.0                   # drag
        @test st.ball.v[3] < 0                      # gravity
        @test abs(st.mean_velocity[1] + 0.05) < 0.02
        @test isfinite(couple_residual(rrun, st, rw))

        # The fluid node count has to track the re-cuts, or the controller is
        # spreading its momentum over the wrong volume.
        @test flow_fluid_count(rw) == count(!=(BBL.SOLID_NODE), rw.wall.kind)

        # Densities stay sane: a refill that left AA scratch behind shows here.
        worst = 0.0
        for k in 1:rdims[3], j in 1:rdims[2], i in 1:rdims[1]
            rw.wall.kind[i, j, k] == BBL.SOLID_NODE && continue
            worst = max(worst, abs(sum(rg[i, j, k, q] for q in 1:27) - 1))
        end
        @test worst < 0.05
    end
end
