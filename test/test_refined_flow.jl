@testset "a ball inside the refined patch" begin
    T = Float64
    τc = 0.7
    cdims = (32, 32, 32)
    lo, hi = (9, 9, 9), (24, 24, 24)
    geom = BaseballGeometry()

    """A refined flow with the ball six fine nodes in radius, at rest in a stream."""
    function build(; stream = -0.05)
        rg = TwoGrid(T, cdims, lo, hi, τc)
        w = RotatingWall(geom, size(rg.fine)[1:3], geom.radius / 6)
        rf = RefinedFlow(rg, w)
        init_refined_flow!(rf, (x, y, z) -> (1.0, stream, 0.0, 0.0))
        return rg, rf
    end

    @testset "the patch has to contain the ball with room to spare" begin
        rg, rf = build()
        @test count(rf.coarse_solid) > 50               # the footprint is found
        @test flow_fluid_count(rf) == prod(cdims) - count(rf.coarse_solid)

        # The coarse level is solved straight through the ball and overwritten
        # every cycle, so the footprint has to stay clear of the restriction's
        # edge — a ball that reaches it would leak meaningless values outward.
        tight = TwoGrid(T, cdims, (13, 13, 13), (20, 20, 20), τc)
        big = RotatingWall(geom, size(tight.fine)[1:3], geom.radius / 6)
        @test_throws ArgumentError RefinedFlow(tight, big)

        @test_throws ArgumentError RefinedFlow(TwoGrid(T, cdims, lo, hi, τc),
                                               RotatingWall(geom, (9, 9, 9), geom.radius / 6))
    end

    @testset "open faces are refused rather than ignored" begin
        # The buffer would have to be imposed on the coarse grid inside the
        # cycle, which is not written. Silently dropping the keyword would run a
        # periodic box while the caller believed the faces were open — the one
        # failure mode that looks like a result.
        rg, rf = build()
        @test_throws ArgumentError advance_flow!(rg, rf, 2, τc;
                                                 channel = OpenChannel{T}())
        @test advance_flow!(rg, rf, 2, τc) isa Tuple    # without one, unchanged
    end

    @testset "force and torque convert between the levels" begin
        # The same physical force, written in each level's lattice units, has to
        # agree. Force carries ρL⁴/T² and torque ρL⁵/T², so the two differ by m²
        # and m³ — using one where the other belongs would misreport the torque
        # by exactly the size of a real spin-decay signal.
        uc = LatticeUnits(; nodes_per_diameter = 20, speed = 39.0, lattice_speed = 0.05)
        uf = LatticeUnits(; nodes_per_diameter = 40, speed = 39.0, lattice_speed = 0.05)
        @test uf.dx ≈ uc.dx / 2 rtol = 1e-14
        @test uf.dt ≈ uc.dt / 2 rtol = 1e-14

        Ff = (0.8, -0.3, 0.05)
        @test to_physical_force(uc, coarse_force(Ff, 2)[1]) ≈
              to_physical_force(uf, Ff[1]) rtol = 1e-13
        Mf = (0.02, 0.4, -0.1)
        @test to_physical_torque(uc, coarse_torque(Mf, 2)[2]) ≈
              to_physical_torque(uf, Mf[2]) rtol = 1e-13

        # And they are genuinely different powers: a torque scaled as a force
        # would be out by exactly m.
        @test coarse_force(Mf, 2)[2] / coarse_torque(Mf, 2)[2] ≈ 2 rtol = 1e-14
    end

    """Advance a freshly built flow and return the settled force and torque."""
    function settle(spin; cycles = 30, operator = :bgk)
        rg, rf = build()
        local F, M
        for _ in 1:cycles
            F, M = advance_flow!(rg, rf, 2, τc; spin = spin, operator = operator)
        end
        return F, M, rg, rf
    end

    @testset "symmetry survives the interface" begin
        # The strongest check available: a sphere in a stream along x has no
        # transverse force, and nothing in the refinement — interpolation,
        # restriction, the wall on the fine level — is allowed to break that.
        F, M, rg, rf = settle((0.0, 0.0, 0.0))
        @test F[1] < 0                                  # drag opposes the stream
        @test abs(F[2]) < 1e-14 * abs(F[1])
        @test abs(F[3]) < 1e-13 * abs(F[1])

        # The seam is not axisymmetric, so a torque about the flow direction is
        # physical even without spin — small, and not zero.
        @test 0 < abs(M[1]) < 0.01 * abs(F[1])
        @test abs(M[2]) < 1e-13 * abs(F[1])

        ρ̄, ū = flow_mean_velocity(rg, rf, (0.0, 0.0, 0.0))
        @test ρ̄ ≈ 1.0 rtol = 1e-4
        @test -0.05 < ū[1] < -0.045                     # drag has slowed it a little
        @test abs(ū[2]) < 1e-9 && abs(ū[3]) < 1e-9
    end

    @testset "Magnus points the same way it does on a uniform grid" begin
        up, mup, = settle((0.0, 0.0, 4e-3))
        down, mdown, = settle((0.0, 0.0, -4e-3))

        @test up[2] > 0                                 # ω × V̂ with ω along +z
        @test down[2] < 0
        @test up[2] ≈ -down[2] rtol = 1e-9              # antisymmetric in ω
        @test up[1] ≈ down[1] rtol = 1e-6               # drag is not
        @test abs(up[2]) > 0.01 * abs(up[1])            # and it is not noise

        @test mup[3] < 0                                # the fluid drags the spin back
        @test mdown[3] > 0
        @test mup[3] ≈ -mdown[3] rtol = 1e-9
    end

    @testset "the sub-cycle has to be shorter on a refined grid" begin
        rg, rf = build()
        spin = (0.0, 0.0, 4e-3)
        n_ref = max_substeps(rf, spin, 0.25)
        n_uni = max_substeps(rf.wall, spin ./ rg.ratio, 0.25)
        @test iseven(n_ref) && n_ref >= 2
        @test n_ref ≈ n_uni / 2 rtol = 0.5              # a cycle moves the fine level twice
    end

    @testset "a refined grid reproduces the uniform one it replaces" begin
        # The validation that matters: same outer domain, same ball, same number
        # of fine steps, refined against uniform-at-the-fine-resolution. Anything
        # the interface gets wrong shows up as a force that does not match.
        #
        # It also settles how big the patch has to be. A tight patch puts the
        # interface within half a diameter of the surface, where the coarse level
        # cannot represent the wake it is being asked to impose, and the force
        # comes out several percent wrong. Room to spare fixes it.
        geo = BaseballGeometry()
        dxf = geo.radius / 4                     # ball radius: 4 fine nodes, 2 coarse
        τf = fine_tau(τc, 2)
        stream = -0.05
        spin = (0.0, 0.0, 6e-3)
        steps = 640                              # fine steps, both sides

        nu = 40                                  # 5 ball diameters at the fine spacing
        wu = RotatingWall(geo, (nu, nu, nu), dxf)
        lbm = LBMState{T}(nu, nu, nu, τf; lattice = D3Q27())
        init_equilibrium!(lbm, (i, j, k) -> (1.0, stream, 0.0, 0.0))
        gu = to_cube_order!(similar(lbm.f), lbm.f)
        Fu = (0.0, 0.0, 0.0)
        for _ in 1:(steps ÷ 2)
            Fu = first(aa_run_walls!(gu, wu.wall, 2, τf; spin = spin ./ 2,
                                     operator = :bgk, reduction = :mean))
        end
        @test Fu[1] < 0 && Fu[2] > 0             # drag and Magnus, as ever

        """Refined run over the same domain, patch `P` diameters across."""
        function refined(P)
            nc, mid = 20, 10
            halfw = round(Int, P * 4 / 2)
            g2 = TwoGrid(T, (nc, nc, nc), ntuple(_ -> mid - halfw, 3),
                         ntuple(_ -> mid + halfw, 3), τc)
            w2 = RotatingWall(geo, size(g2.fine)[1:3], dxf)
            r2 = RefinedFlow(g2, w2)
            init_refined_flow!(r2, (x, y, z) -> (1.0, stream, 0.0, 0.0))
            F = (0.0, 0.0, 0.0)
            for _ in 1:(steps ÷ 4)
                F = first(advance_flow!(g2, r2, 2, τc; spin = spin, operator = :bgk))
            end
            return F .* g2.ratio^2               # back into fine units, to compare
        end

        tight = refined(2.5)
        roomy = refined(3.5)
        @info "refined against uniform" drag_uniform = Fu[1] drag_tight = tight[1] drag_roomy = roomy[1] lift_uniform = Fu[2] lift_tight = tight[2] lift_roomy = roomy[2]

        @test roomy[1] ≈ Fu[1] rtol = 0.03       # room to spare: within a few percent
        @test roomy[2] ≈ Fu[2] rtol = 0.05
        # And the tight patch is measurably worse, which is the whole point of
        # knowing how big it has to be.
        @test abs(tight[1] / Fu[1] - 1) > abs(roomy[1] / Fu[1] - 1)
    end

    @testset "the coupled loop flies a refined ball" begin
        rg, rf = build()
        # Six coarse nodes per diameter, so twelve fine — the same ball as
        # `build`, which is what the patch has room for.
        units = LatticeUnits(; nodes_per_diameter = 6, speed = 39.0,
                             lattice_speed = 0.05, ν = 39.0 * 0.0748 / 60)
        @test units.dx / 2 ≈ geom.radius / 6 rtol = 1e-14
        rgt = TwoGrid(T, cdims, lo, hi, units.τ)
        w = RotatingWall(geom, size(rgt.fine)[1:3], units.dx / 2)
        rff = RefinedFlow(rgt, w)
        init_refined_flow!(rff, (x, y, z) -> (1.0, -0.05, 0.0, 0.0))

        ball = BallState(; position = (2.0, 0.0, 1.8), velocity = (39.0, 0.0, 0.0),
                         spin = spin_from_rpm((0.0, 0.0, 1.0), 2708))
        nsub = min(max_substeps(rff, lattice_spin(units, ball), 0.25), 10)
        run = PitchRun(units, BaseballProperties(); substeps = nsub,
                       control_time = 60 * nsub, operator = :bgk)
        st = PitchState(ball)

        spin_up!(rgt, run, st, rff; cycles = 10)
        v0 = st.ball.v
        fly!(rgt, run, st, rff; cycles = 20)

        @test all(isfinite, rgt.coarse) && all(isfinite, rgt.fine)
        @test st.ball.v[1] < v0[1]                      # drag
        @test st.ball.v[3] < 0                          # gravity
        @test st.ball.v[2] > 0                          # Magnus, the ω × V̂ way
        @test st.ball.x[1] > 2.0
        @test isfinite(couple_residual(run, st, rff))
        @test abs(st.mean_velocity[1] + 0.05) < 0.02
        @test rff.cycles == (10 + 20) * (nsub ÷ 2)
    end
end
