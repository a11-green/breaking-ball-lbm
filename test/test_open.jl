@testset "inflow and outflow" begin
    T = Float64
    τ = 0.6
    dims = (24, 8, 8)

    """A uniform stream along -x, in cube-slot order."""
    function uniform(ux; ρ = 1.0, d = dims)
        s = LBMState{T}(d..., τ; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (ρ, ux, 0.0, 0.0))
        return to_cube_order!(similar(s.f), s.f)
    end

    @testset "the stream it states is a fixed point" begin
        # The one property a boundary condition must have: given exactly the
        # flow it is imposing, it changes nothing. Anything else is a source
        # sitting in the face, and every other test here would be measuring it.
        u = -0.05
        ch = OpenChannel{T}()
        g = uniform(u)
        before = copy(g)
        aa_run!(g, 40, τ; channel = ch, inlet = (u, 0.0, 0.0))
        @test maximum(abs.(g .- before)) < 1e-14
    end

    @testset "the wrap is cut, and only the wrap" begin
        # A disturbance at the outlet reaches the inlet end two ways: the short
        # way, through the wrap, and the long way, as sound across the box. The
        # condition can only cut the first, and a test that cannot tell them
        # apart measures nothing — so the box is long enough that in the time
        # allowed, sound covers the wrap's two planes many times over and the
        # direct crossing not at all.
        u = -0.05
        long = (48, 8, 8)
        steps = 20                                  # c_s * 20 ≈ 11 planes ≪ 44
        g = uniform(u; d = long)
        g[2, :, :, :] .*= 1.5                       # a blob in the outlet buffer
        clean = uniform(u; d = long)
        far = (long[1] - 4):long[1]

        cut = copy(g)
        aa_run!(cut, steps, τ; channel = OpenChannel{T}(), inlet = (u, 0.0, 0.0))
        @test maximum(abs.(cut[far, :, :, :] .- clean[far, :, :, :])) < 1e-12

        wrapped = copy(g)
        aa_run!(wrapped, steps, τ)
        @test maximum(abs.(wrapped[far, :, :, :] .- clean[far, :, :, :])) > 1e-6
    end

    @testset "the outlet anchors the density the inlet does not" begin
        # An inlet that set the density too would over-determine the problem;
        # with only the outlet pinning it, a box started off-reference is pulled
        # back rather than drifting.
        u = -0.05
        ch = OpenChannel{T}(density = 1.0)
        g = uniform(u; ρ = 1.02)
        mass0 = sum(g)
        aa_run!(g, 200, τ; channel = ch, inlet = (u, 0.0, 0.0))
        mass1 = sum(g)
        target = prod(dims) * 1.0
        @test abs(mass1 - target) < abs(mass0 - target)
        @test mass1 < mass0
    end

    @testset "a new free stream arrives from the inlet" begin
        # The inlet states the stream, so changing it has to change the flow.
        # It arrives acoustically, not advectively: one sound crossing of a box
        # twenty-four wide is about forty steps, while convecting the same
        # distance at 0.04 would take six hundred — so after forty steps the far
        # end has already responded, which is the LBM pressure wave doing the
        # work and not the fluid being carried there.
        u0, u1 = -0.04, -0.06
        ch = OpenChannel{T}()
        g = uniform(u0)
        aa_run!(g, 40, τ; channel = ch, inlet = (u1, 0.0, 0.0))
        ux(i) = BBL.node_macroscopic(g, i, 4, 4, T)[2]
        progress(i) = (ux(i) - u0) / (u1 - u0)
        @test progress(dims[1] - 3) > 0.8                # the inlet end has it
        @test progress(4) > 0.8                          # and so has the far end
        @test minimum(progress, 3:(dims[1] - 2)) > 0.8

        # And eventually everywhere — but slowly, because nothing here is
        # pushing the fluid except the pressure gradient the mismatch itself
        # sets up: a few thousand steps for a box twenty-four wide. That is a
        # property of this test, not of the production loop, where the frame's
        # body force accelerates the interior and the identity d(u∞)/dt =
        # a_fluid (§4.4) means the inlet and the interior change together
        # instead of one dragging the other.
        aa_run!(g, 12000, τ; channel = ch, inlet = (u1, 0.0, 0.0))
        for i in 3:(dims[1] - 2)
            @test BBL.node_macroscopic(g, i, 4, 4, T)[2] ≈ u1 rtol = 0.01
        end
    end

    @testset "a wake leaves instead of coming back" begin
        # The whole point. A momentum deficit on the axis is convected out, and
        # what arrives at the inlet afterwards is the stated free stream rather
        # than the deficit.
        u = -0.05
        ch = OpenChannel{T}()
        long = (48, 8, 8)
        s = LBMState{T}(long..., τ; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (1.0, i in 20:28 && j in 3:6 && k in 3:6 ?
                                                0.5u : u, 0.0, 0.0))
        g = to_cube_order!(similar(s.f), s.f)
        open_run = copy(g)
        wrap_run = copy(g)
        steps = 2 * (long[1] ÷ abs(u) ÷ 2)            # about one flow-through
        aa_run!(open_run, Int(steps), τ; channel = ch, inlet = (u, 0.0, 0.0))
        aa_run!(wrap_run, Int(steps), τ)

        defect(h, i) = 1 - BBL.node_macroscopic(h, i, 4, 4, T)[2] / u
        # Measured at the plane the deficit would have wrapped round to.
        @test abs(defect(open_run, long[1] - 4)) < 0.02
        @test abs(defect(open_run, long[1] - 4)) < abs(defect(wrap_run, long[1] - 4))
    end

    @testset "it refuses what it cannot mean" begin
        @test_throws ArgumentError OpenChannel{T}(depth = 1)
        @test_throws ArgumentError OpenChannel{T}(density = 0)
        ch = OpenChannel{T}()
        @test_throws ArgumentError apply_open!(uniform(-0.05), ch, (0.05, 0.0, 0.0))
        @test_throws ArgumentError apply_open!(uniform(-0.05; d = (5, 8, 8)), ch,
                                               (-0.05, 0.0, 0.0))

        # A body in the buffer is a body inside a boundary condition.
        R = 3.0
        box = (20, 12, 12)
        near = sphere_sdf_field(T, box, R; center = (3.5, 6.5, 6.5))
        @test !open_is_clear(build_wall_field(near;
                                              sdf_fn = sphere_sdf_fn(box, R;
                                                  center = (3.5, 6.5, 6.5))), ch)
        middle = sphere_sdf_field(T, box, R)
        @test open_is_clear(build_wall_field(middle; sdf_fn = sphere_sdf_fn(box, R)), ch)
    end
end
