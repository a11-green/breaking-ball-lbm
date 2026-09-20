@testset "flow-field snapshots" begin
    T = Float64
    dims = (16, 12, 10)
    τ = 0.6

    """Read a legacy VTK back: the header as text, the payload as big-endian floats."""
    function read_vtk(path)
        data = read(path)
        # The header ends at the line after POINT_DATA; everything from there is
        # a mix of short text lines and binary blocks, so walk it.
        pos = 1
        line() = begin
            e = findnext(==(UInt8('\n')), data, pos)
            s = String(data[pos:(e - 1)])
            pos = e + 1
            s
        end
        @test line() == "# vtk DataFile Version 3.0"
        title = line()
        @test line() == "BINARY"
        @test line() == "DATASET STRUCTURED_POINTS"
        d = parse.(Int, split(line())[2:4])
        origin = parse.(Float64, split(line())[2:4])
        spacing = parse.(Float64, split(line())[2:4])
        npts = parse(Int, split(line())[2])
        fields = Dict{String,Array{Float32}}()
        n = prod(d)
        while pos <= length(data)
            head = split(line())
            isempty(head) && continue
            if head[1] == "SCALARS"
                line()                                   # LOOKUP_TABLE
                raw = reinterpret(Float32, data[pos:(pos + 4n - 1)])
                pos += 4n
                fields[head[2]] = reshape(ntoh.(raw), d...)
            elseif head[1] == "VECTORS"
                raw = reinterpret(Float32, data[pos:(pos + 12n - 1)])
                pos += 12n
                fields[head[2]] = reshape(ntoh.(raw), 3, d...)
            end
        end
        return (; title, dims = Tuple(d), origin = Tuple(origin),
                spacing = Tuple(spacing), npts, fields)
    end

    """A uniform stream, with a sphere cut out of it."""
    function state(ux = -0.05; d = dims)
        s = LBMState{T}(d..., τ; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (1.0, ux, 0.0, 0.0))
        return to_cube_order!(similar(s.f), s.f)
    end

    @testset "what is written is what comes back" begin
        # The point of a file format is that someone else can read it, so the
        # test reads it rather than checking that bytes were produced.
        a = T[i + 10j + 100k for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
        v = T[i - j for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3], _ in 1:3]
        path = tempname() * ".vtk"
        write_vtk(path, ["a" => a], ["v" => v]; spacing = 0.25, origin = (1.0, 2.0, 3.0))

        got = read_vtk(path)
        @test got.dims == dims
        @test got.npts == prod(dims)
        @test got.spacing == (0.25, 0.25, 0.25)
        @test got.origin == (1.0, 2.0, 3.0)
        # The index order is the one ParaView assumes: x fastest.
        @test got.fields["a"] ≈ Float32.(a)
        @test got.fields["v"][1, 3, 4, 5] ≈ Float32(3 - 4)
        rm(path)
    end

    @testset "a snapshot says what the flow is doing" begin
        R = 3.0
        ϕ = sphere_sdf_field(T, dims, R)
        solid = solid_mask(ϕ)
        g = state(-0.05)
        path = tempname() * ".vtk"
        write_snapshot(path, g, solid; spacing = 0.002)

        got = read_vtk(path)
        @test got.dims == dims
        @test Set(keys(got.fields)) == Set(["density", "speed", "vorticity_magnitude",
                                            "q_criterion", "solid", "velocity",
                                            "vorticity"])
        # The ball is a hole, not a region of nonsense: masked nodes are NaN in
        # the fields and 1 in the mask.
        centre = (dims .+ 1) .÷ 2
        @test got.fields["solid"][centre...] == 1
        @test isnan(got.fields["speed"][centre...])
        # Away from it, a uniform stream at 0.05 with no vorticity.
        @test got.fields["speed"][1, 1, 1] ≈ 0.05f0 rtol = 1e-5
        @test got.fields["velocity"][1, 1, 1, 1] ≈ -0.05f0 rtol = 1e-5
        @test abs(got.fields["vorticity_magnitude"][1, 1, 1]) < 1e-6
        @test got.fields["density"][1, 1, 1] ≈ 1.0f0 rtol = 1e-6
        rm(path)
    end

    @testset "cropping and striding keep the grid honest" begin
        # A crop has to land in the right place in space, or a viewer draws the
        # near wake where the far wake was.
        big = (24, 24, 24)
        g = state(-0.05; d = big)
        solid = falses(big)
        path = tempname() * ".vtk"
        write_snapshot(path, g, solid; crop = 4, spacing = 0.5)
        got = read_vtk(path)
        @test all(got.dims .<= 11)               # 2*4 + 1, plus rounding
        @test all(got.dims .>= 9)
        # centre 12.5, half-width 4 → floor(8.5) = 8 is the first node inside the
        # crop (7 is the halo the gradients use and lose), so the file starts at
        # 0.5*(8-1) = 3.5. The overlap check below confirms it independently.
        @test got.origin == (3.5, 3.5, 3.5)
        @test got.spacing == (0.5, 0.5, 0.5)
        rm(path)

        write_snapshot(path, g, solid; crop = 4, stride = 2, spacing = 0.5)
        thin = read_vtk(path)
        @test all(thin.dims .< got.dims)
        @test thin.spacing == (1.0, 1.0, 1.0)    # spacing follows the stride
        @test thin.origin == got.origin
        rm(path)

        # The halo is taken for the gradients and then dropped, so a cropped
        # snapshot and a full one agree where they overlap.
        full = tempname() * ".vtk"
        write_snapshot(full, g, solid; spacing = 0.5)
        whole = read_vtk(full)
        i0 = round(Int, got.origin[1] / 0.5) + 1
        @test whole.fields["speed"][i0, i0, i0] ≈ got.fields["speed"][1, 1, 1] rtol = 1e-6
        rm(full)
    end

    @testset "seam and axis as polylines" begin
        # The geometry the animation needs: a closed seam curve and the spin
        # axis, in the same world metres the volume snapshot uses.
        geom = BaseballGeometry()
        q = Quat(one(T), zero(T), zero(T), zero(T))
        centre = (0.35, -0.10, 1.80)
        seam = seam_world(geom, q, centre; samples = 96)
        axis = [(centre[1], centre[2], centre[3] - 0.05),
                (centre[1], centre[2], centre[3] + 0.05)]

        path = tempname() * ".vtk"
        write_polylines(path, [seam, axis];
                        names = ["seam", "spin axis"], title = "t = 0.0100 s")

        data = read(path)
        pos = 1
        line() = begin
            e = findnext(==(UInt8('\n')), data, pos)
            str = String(data[pos:(e - 1)])
            pos = e + 1
            str
        end
        @test line() == "# vtk DataFile Version 3.0"
        # The names ride in the title, which is where a reader looks for them.
        @test line() == "t = 0.0100 s [seam, spin axis]"
        @test line() == "BINARY"
        @test line() == "DATASET POLYDATA"
        head = split(line())
        @test head[1] == "POINTS" && head[3] == "float"
        npoints = parse(Int, head[2])
        @test npoints == length(seam) + 2

        pts = reshape(ntoh.(reinterpret(Float32, data[pos:(pos + 12npoints - 1)])),
                      3, npoints)
        pos += 12npoints
        # Every point came back where it was put, to single precision.
        for (k, p3) in enumerate(seam), d in 1:3
            @test pts[d, k] ≈ Float32(p3[d]) rtol = 1e-6
        end
        # The seam closes: it is drawn as a loop, not an arc with a gap.
        @test pts[:, 1] == pts[:, length(seam)]
        # And it is a seam, not a great circle: every point is one radius out,
        # and the curve leaves the equator.
        radii = [sqrt(sum(abs2, pts[:, k] .- Float32.(centre))) for k in 1:length(seam)]
        @test all(r -> isapprox(r, geom.seam.radius; rtol = 1e-5), radii)
        @test maximum(pts[3, 1:length(seam)]) - Float32(centre[3]) >
              0.5 * geom.seam.radius

        @test line() == ""
        head = split(line())
        @test head[1] == "LINES" && parse(Int, head[2]) == 2
        @test parse(Int, head[3]) == npoints + 2
        conn = ntoh.(reinterpret(Int32, data[pos:(pos + 4 * (npoints + 2) - 1)]))
        # First record: the seam's length, then its own indices from zero.
        @test conn[1] == length(seam)
        @test conn[2:(1 + length(seam))] == Int32.(0:(length(seam) - 1))
        # Second: the axis, indexed on from where the seam left off.
        @test conn[2 + length(seam)] == 2
        @test conn[(3 + length(seam)):(4 + length(seam))] ==
              Int32.(length(seam):(length(seam) + 1))

        @test_throws ArgumentError write_polylines(tempname() * ".vtk",
                                                   Vector{NTuple{3,Float64}}[])
        @test_throws ArgumentError write_polylines(tempname() * ".vtk",
                                                   [NTuple{3,Float64}[]])
        @test_throws DimensionMismatch write_polylines(tempname() * ".vtk",
                                                       [seam, axis]; names = ["only one"])
    end

    @testset "read_polylines is the inverse of write_polylines" begin
        geom = BaseballGeometry()
        q = quat_from_axis_angle((0.3, 1.0, -0.2), 0.7)
        centre = (0.35, -0.10, 1.80)
        seam = seam_world(geom, q, centre; samples = 96)
        axis = [(centre[1], centre[2], centre[3] - 0.05),
                (centre[1], centre[2], centre[3] + 0.05)]

        path = tempname() * ".vtk"
        write_polylines(path, [seam, axis]; names = ["seam", "spin axis"],
                        title = "t = 0.0100 s")
        lines, ids, title = read_polylines(path)

        @test title == "t = 0.0100 s [seam, spin axis]"
        @test ids == [1, 2]
        @test length(lines) == 2
        @test length(lines[1]) == length(seam)
        @test length(lines[2]) == 2
        for (a, b) in zip(lines[1], seam), (x, y) in zip(a, b)
            @test x ≈ y atol = 1e-5      # single-precision round trip
        end
        for (a, b) in zip(lines[2], axis), (x, y) in zip(a, b)
            @test x ≈ y atol = 1e-5
        end

        # Files this reader was not asked to read say so, not "0 lines".
        bad = tempname() * ".vtk"
        write(bad, "not a vtk file at all\n")
        @test_throws ArgumentError read_polylines(bad)

        snap_path = tempname() * ".vtk"
        write_snapshot(snap_path, state(), falses(dims))
        @test_throws ArgumentError read_polylines(snap_path)   # STRUCTURED_POINTS, not POLYDATA
    end

    @testset "ball_from_seam recovers the ball a seam was drawn on" begin
        geom = BaseballGeometry()
        q = quat_from_axis_angle((0.1, -0.4, 0.9), 1.3)
        centre = (1.5, -0.02, 1.75)
        seam = seam_world(geom, q, centre; samples = 200)   # scale = 1: real metres
        axis = [(centre[1], centre[2], centre[3] - 0.05),
                (centre[1], centre[2], centre[3] + 0.05)]

        path = tempname() * ".vtk"
        write_polylines(path, [seam, axis]; names = ["seam", "spin axis"])
        lines, ids, _ = read_polylines(path)
        c, r = ball_from_seam(lines, ids)

        for (x, y) in zip(c, centre)
            @test x ≈ y atol = 1e-5
        end
        @test r ≈ geom.radius atol = 1e-5

        # The order `run_pitch.jl` writes in (seam, then axis) is what makes
        # `line_id == 1` mean "the seam" — a file that never says so is refused
        # rather than silently measured against the spin axis's two points.
        @test_throws ArgumentError ball_from_seam(lines, [2, 3])
    end

    @testset "sphere_mesh is a closed sphere of the right size" begin
        centre = (1.0, -2.0, 0.5)
        R = 0.0374
        pts, tris = sphere_mesh(centre, R; slices = 32, stacks = 24)

        @test length(pts) == 32 * 25          # (stacks + 1) rings of `slices`
        @test length(tris) == 2 * 32 * 23     # 2·slices·(stacks - 1)
        for p in pts
            d = sqrt((p[1] - centre[1])^2 + (p[2] - centre[2])^2 + (p[3] - centre[3])^2)
            @test d ≈ R atol = 1e-12          # every point is exactly on the sphere
        end
        # The two polar rings each collapse to one physical point.
        north = pts[1:32]
        @test all(p -> all(isapprox.(p, north[1]; atol = 1e-12)), north)
        south = pts[(end - 31):end]
        @test all(p -> all(isapprox.(p, south[1]; atol = 1e-12)), south)

        # Every triangle index is in bounds and non-degenerate.
        n = length(pts)
        for (a, b, c) in tris
            @test 1 <= a <= n && 1 <= b <= n && 1 <= c <= n
            @test length(Set((a, b, c))) == 3
        end

        # Total area converges to 4πR² as the mesh refines (a bound that would
        # not hold if triangles were overlapping or missing).
        function area(pts, tris)
            s = 0.0
            for (a, b, c) in tris
                pa, pb, pc = pts[a], pts[b], pts[c]
                u = pb .- pa
                v = pc .- pa
                cx = u[2] * v[3] - u[3] * v[2]
                cy = u[3] * v[1] - u[1] * v[3]
                cz = u[1] * v[2] - u[2] * v[1]
                s += 0.5 * sqrt(cx^2 + cy^2 + cz^2)
            end
            return s
        end
        exact = 4 * pi * R^2
        coarse = area(sphere_mesh(centre, R; slices = 8, stacks = 6)...)
        fine = area(pts, tris)
        @test abs(fine - exact) < abs(coarse - exact)
        @test fine ≈ exact rtol = 1e-2

        @test_throws ArgumentError sphere_mesh(centre, R; slices = 2)
        @test_throws ArgumentError sphere_mesh(centre, R; stacks = 1)
        @test_throws ArgumentError sphere_mesh(centre, -1.0)
    end

    @testset "write_ball / write_triangle_mesh produce a readable POLYDATA mesh" begin
        centre = (0.1, 0.2, 0.3)
        R = 0.5
        path = tempname() * ".vtk"
        write_ball(path, centre, R; slices = 12, stacks = 8, title = "a ball")

        data = read(path)
        pos = 1
        line() = begin
            e = findnext(==(UInt8('\n')), data, pos)
            s = String(data[pos:(e - 1)])
            pos = e + 1
            s
        end
        @test line() == "# vtk DataFile Version 3.0"
        @test line() == "a ball"
        @test line() == "BINARY"
        @test line() == "DATASET POLYDATA"
        head = split(line())
        npoints = parse(Int, head[2])
        @test npoints == 12 * 9
        coords = ntoh.(reinterpret(Float32, data[pos:(pos + 12npoints - 1)]))
        pos += 12npoints
        for k in 1:npoints
            p = (coords[3k-2], coords[3k-1], coords[3k])
            d = sqrt(sum(abs2, p .- Float32.(centre)))
            @test d ≈ Float32(R) atol = 1e-4
        end
        @test line() == ""
        head = split(line())
        @test head[1] == "POLYGONS"
        ntris = parse(Int, head[2])
        @test parse(Int, head[3]) == 4ntris
        conn = ntoh.(reinterpret(Int32, data[pos:(pos + 4 * 4ntris - 1)]))
        for t in 1:ntris
            @test conn[4t-3] == 3                       # every cell is a triangle
            @test all(0 .<= conn[(4t-2):4t] .< npoints)  # zero-based, in range
        end

        @test_throws ArgumentError write_triangle_mesh(tempname() * ".vtk",
                                                        NTuple{3,Float64}[], [(1, 2, 3)])
        @test_throws ArgumentError write_triangle_mesh(tempname() * ".vtk",
                                                        [(0.0, 0.0, 0.0)],
                                                        NTuple{3,Int}[])
        @test_throws ArgumentError write_triangle_mesh(tempname() * ".vtk",
                                                        [(0.0, 0.0, 0.0)], [(1, 2, 3)])
    end

    @testset "read_vtk_series round-trips the file run_pitch.jl writes" begin
        path = tempname() * ".vtk.series"
        write(path, """
              {
                "file-series-version" : "1.0",
                "files" : [
                  { "name" : "seam-00000.vtk", "time" : 0.007626 },
                  { "name" : "seam-00003.vtk", "time" : 0.008057 },
                  { "name" : "seam-00200.vtk", "time" : -1.5e-2 }
                ]
              }
              """)
        frames = read_vtk_series(path)
        @test length(frames) == 3
        @test frames[1] == (name = "seam-00000.vtk", time = 0.007626)
        @test frames[2].name == "seam-00003.vtk"
        @test frames[3].time ≈ -0.015

        # A file with no name/time pairs at all is an empty series, not an
        # error — but one whose counts disagree is not a series file at all.
        empty_path = tempname() * ".vtk.series"
        write(empty_path, "{}\n")
        @test read_vtk_series(empty_path) == []

        mismatched = tempname() * ".vtk.series"
        write(mismatched, "\"name\" : \"a.vtk\", \"name\" : \"b.vtk\", \"time\" : 0.1\n")
        @test_throws ArgumentError read_vtk_series(mismatched)
    end

    @testset "it refuses what it cannot write" begin
        a = zeros(T, dims)
        @test_throws ArgumentError write_vtk(tempname() * ".vtk",
                                             Pair{String,Array{T,3}}[])
        @test_throws DimensionMismatch write_vtk(tempname() * ".vtk",
                                                 ["a" => a, "b" => zeros(T, 3, 3, 3)])
        @test_throws DimensionMismatch write_snapshot(tempname() * ".vtk", state(),
                                                      falses(3, 3, 3))
        @test_throws ArgumentError write_snapshot(tempname() * ".vtk", state(),
                                                  falses(dims); stride = 0)
    end
end
