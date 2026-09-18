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
