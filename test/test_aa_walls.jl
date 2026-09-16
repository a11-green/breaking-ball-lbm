@testset "walls in the AA-pattern" begin
    lat = D3Q27()
    dims = (18, 18, 18)
    R = 5.0
    τ = 0.8
    ϕ = sphere_sdf_field(dims, R)
    sdf_fn = sphere_sdf_fn(dims, R)
    solid = solid_mask(ϕ)

    @testset "wall field and link list describe the same geometry" begin
        wall = build_wall_field(ϕ; sdf_fn = sdf_fn)
        links = build_links(ϕ; sdf_fn = sdf_fn, lattice = lat)

        @test count(==(BBL.SOLID_NODE), wall.kind) == count(solid)
        @test length(wall) == count(>(Int32(0)), wall.kind)

        # Every link in the list appears in the field with the same δ, and nothing else does.
        counted = 0
        for n in eachindex(links.q)
            i, j, k = Int(links.i[n]), Int(links.j[n]), Int(links.k[n])
            b = wall.kind[i, j, k]
            @test b > 0
            s = BBL.CUBE27[Int(links.q[n])]
            @test wall.deltas[s, b] ≈ links.δ[n] rtol = 1e-12
            counted += 1
        end
        @test counted == count(>(0), wall.deltas)
    end

    """Populations of the fluid nodes only: solid nodes are scratch space for AA."""
    fluid_populations(f) = [f[i, j, k, q] for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3],
                            q in 1:27 if !solid[i, j, k]]

    for rule in (:halfway, :interpolated_local), operator in (:bgk, :central_moment)
        @testset "matches the two-lattice reference ($rule, $operator)" begin
            # The two-lattice path is the one validated against Poiseuille flow, the
            # momentum balance and the rotating-sphere torque. AA-pattern applies the
            # same rule on the scatter instead, and has to agree step for step.
            field = (i, j, k) -> (1.0 + 0.002 * sin(i + 2j), 0.01cos(i), -0.008sin(j), 0.006cos(k))
            force = (2.0e-5, -1.0e-5, 0.0)
            spin = (0.0, 0.0, 1.5e-3)
            nsteps = 6

            ref = LBMState(dims..., τ; lattice = lat)
            init_equilibrium!(ref, field)
            init_solid!(ref, solid)
            g = to_cube_order!(similar(ref.f), ref.f)

            links = build_links(ϕ; sdf_fn = sdf_fn, lattice = lat)
            vals = zeros(Float64, length(links))
            local Fref, τref
            for _ in 1:nsteps
                Fref, τref = step!(ref, links, vals; force = force, solid = solid,
                                   rule = rule, operator = operator, spin = spin)
            end

            wall = build_wall_field(ϕ; sdf_fn = sdf_fn)
            Faa, τaa = aa_run_walls!(g, wall, nsteps, τ; force = force, spin = spin,
                                     operator = operator, rule = rule)

            got = from_cube_order!(similar(g), g)
            @test fluid_populations(got) ≈ fluid_populations(ref.f) rtol = 1e-12
            @test collect(Faa) ≈ collect(Fref) rtol = 1e-10
            @test collect(τaa) ≈ collect(τref) rtol = 1e-10
        end
    end

    @testset "fluid at rest stays at rest" begin
        wall = build_wall_field(ϕ; sdf_fn = sdf_fn)
        s = LBMState(dims..., τ; lattice = lat)
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.0, 0.0, 0.0))
        g = to_cube_order!(similar(s.f), s.f)

        F, τq = aa_run_walls!(g, wall, 20, τ)
        @test all(abs.(F) .< 1e-14)
        @test all(abs.(τq) .< 1e-14)

        from_cube_order!(s.f, g)
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            solid[i, j, k] && continue
            ρ, ux, uy, uz = macroscopic(s, i, j, k)
            @test ρ ≈ 1.0 rtol = 1e-12
            @test max(abs(ux), abs(uy), abs(uz)) < 1e-14
        end
    end

    @testset "halfway conserves mass, interpolation leaks a little" begin
        # Plain bounce-back returns exactly what arrives, so the fluid keeps its
        # mass to round-off. Bouzidi's rule returns a combination instead, equal to
        # the incoming population only when δ = 1/2, so it leaks — a known property
        # of interpolated bounce-back and the price of placing the wall correctly.
        wall = build_wall_field(ϕ; sdf_fn = sdf_fn)
        s = LBMState(dims..., τ; lattice = lat)

        function drift(rule)
            init_equilibrium!(s, (i, j, k) -> (1.0 + 0.01sin(i + j), 0.01, 0.0, -0.005))
            g = to_cube_order!(similar(s.f), s.f)
            mass(h) = begin
                from_cube_order!(s.f, h)
                sum(s.f[i, j, k, q] for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3],
                    q in 1:27 if !solid[i, j, k])
            end
            m0 = mass(copy(g))
            aa_run_walls!(g, wall, 20, τ; rule = rule)
            return abs(mass(g) - m0) / m0
        end

        halfway_drift = drift(:halfway)
        interpolated_drift = drift(:interpolated_local)
        @info "mass drift over 20 steps" halfway_drift interpolated_drift
        @test halfway_drift < 1e-11          # summation round-off, nothing more
        @test interpolated_drift > 1e-9      # the leak is real, and
        @test interpolated_drift < 1e-4      # small enough not to matter here
    end

    @testset "odd step counts are rejected" begin
        wall = build_wall_field(ϕ)
        g = zeros(dims..., 27)
        @test_throws ArgumentError aa_run_walls!(g, wall, 3, τ)
    end
end
