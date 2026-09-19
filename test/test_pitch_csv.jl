@testset "reading a trajectory CSV back in" begin
    @testset "round-trips a run_pitch.jl-shaped file" begin
        path = tempname() * ".csv"
        write(path, """
              t,x,y,z,speed,rpm,recuts
              0.01,2.0,0.0,1.75,38.99,2708.0,12
              0.02,2.4,0.01,1.749,38.9,2707.5,18
              0.03,2.8,0.02,1.748,38.8,2707.0,24
              """)
        data = read_pitch_csv(path)

        @test Set(keys(data)) == Set([:t, :x, :y, :z, :speed, :rpm, :recuts])
        @test data[:t] == [0.01, 0.02, 0.03]
        @test data[:x] == [2.0, 2.4, 2.8]
        @test data[:recuts] == [12.0, 18.0, 24.0]     # integers parse as Float64 too
        @test all(length(v) == 3 for v in values(data))
    end

    @testset "scientific notation and negative values" begin
        path = tempname() * ".csv"
        write(path, "u_in_y,residual\n-1.4445184e-8,9.228374e-3\n1.2e2,-3.0\n")
        data = read_pitch_csv(path)
        @test data[:u_in_y] ≈ [-1.4445184e-8, 120.0]
        @test data[:residual] ≈ [9.228374e-3, -3.0]
    end

    @testset "a trailing blank line is not a phantom row" begin
        path = tempname() * ".csv"
        write(path, "a,b\n1,2\n3,4\n")   # write() leaves a final newline
        data = read_pitch_csv(path)
        @test length(data[:a]) == 2
    end

    @testset "an empty file has no header to read" begin
        path = tempname() * ".csv"
        write(path, "")
        @test_throws ArgumentError read_pitch_csv(path)
    end

    @testset "a short row names itself, not just its shape" begin
        path = tempname() * ".csv"
        write(path, "a,b,c\n1,2,3\n4,5\n")     # row 2 is missing a field
        err = try
            read_pitch_csv(path)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("row 3", err.msg)       # line 3 of the file, 1-based
        @test occursin("2", err.msg)           # what it had
        @test occursin("3", err.msg)           # what was expected
    end

    @testset "require_columns" begin
        data = Dict(:t => [1.0], :x => [2.0])
        @test require_columns(data, :t, :x) === nothing     # present: no error

        err = try
            require_columns(data, :t, :u_in_x, :rpm)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("u_in_x", err.msg)
        @test occursin("rpm", err.msg)
        # The columns that ARE present are listed too, so the message is
        # actionable without opening the file.
        @test occursin("t", err.msg) && occursin("x", err.msg)
    end

    @testset "moving_average" begin
        # A constant series is unchanged by any window.
        @test moving_average(fill(3.0, 10), 5) == fill(3.0, 10)

        # An even window is bumped up to the next odd one, so it stays centred
        # and lag-free rather than favouring one side.
        v = collect(1.0:10.0)
        @test moving_average(v, 4) == moving_average(v, 5)

        # Interior points of a straight line are exactly the line — a centred
        # average of a linear ramp is invariant away from the edges.
        avg = moving_average(v, 5)
        for i in 3:8
            @test avg[i] ≈ v[i]
        end
        # At the edges the window truncates rather than wrapping or padding
        # with zeros, so the edge value pulls toward the interior, not to 0.
        @test avg[1] > v[1]
        @test avg[end] < v[end]

        # window = 1 is the identity, and a window request of 0 or negative is
        # clamped to that rather than throwing or averaging an empty range.
        @test moving_average(v, 1) == v
        @test moving_average(v, 0) == v
        @test moving_average(v, -5) == v

        # A window wider than the data does not go out of bounds.
        @test length(moving_average([1.0, 2.0, 3.0], 101)) == 3
    end

    @testset "default_smoothing_window" begin
        @test default_smoothing_window(40) == 5           # clamps at the low end
        @test default_smoothing_window(400) == 10
        @test default_smoothing_window(4000) == 100
        @test default_smoothing_window(100_000) == 201     # clamps at the high end
    end

    @testset "reads back an actual run_pitch.jl file" begin
        # The real header, in the real column order, with two real-shaped rows
        # — a regression test against the schema drifting out from under the
        # reader without anyone noticing (the reader itself makes no
        # assumption about column order or count, so this checks the schema's
        # promise, not the reader's).
        cols = Symbol.(split(
            "t,x,y,z,vx,vy,vz,speed,u_in_x,u_in_y,u_in_z,u_lat_x,u_lat_y,u_lat_z," *
            "CD,CL,Cside,Fx,Fy,Fz,Tx,Ty,Tz,wx,wy,wz,qw,qx,qy,qz," *
            "ax_lat,ay_lat,az_lat,um_lat_x,um_lat_y,um_lat_z," *
            "rpm,steps,fresh,recuts,residual", ','))
        header = join(cols, ',')
        # Column values are the column's position, so a row is easy to check
        # by index and a transposition between columns would show up.
        row(k) = join((string(Float64(i) + 100k) for i in 1:length(cols)), ',')

        path = tempname() * ".csv"
        write(path, header * "\n" * row(1) * "\n" * row(2) * "\n")
        data = read_pitch_csv(path)

        for (i, name) in enumerate(cols)
            @test haskey(data, name)
            @test data[name] == [i + 100.0, i + 200.0]
        end
        require_columns(data, :u_in_x, :u_in_y, :u_in_z, :qw, :qx, :qy, :qz,
                        :ax_lat, :ay_lat, :az_lat, :um_lat_x, :recuts, :residual)
    end
end
