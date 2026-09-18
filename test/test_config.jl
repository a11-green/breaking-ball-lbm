@testset "settings files" begin
    Base.@kwdef mutable struct Settings
        resolution::Int = 40
        domain::Float64 = 8.0
        precision::Symbol = :f32
        axis::NTuple{3,Float64} = (0.0, 0.0, 1.0)
        out::String = "pitch.csv"
        open_faces::Bool = false
    end

    @testset "writing then reading is the identity" begin
        # The round trip is the whole point: a run prints its own inputs and
        # must be repeatable from what it printed.
        a = Settings(resolution = 33, domain = 6.5, precision = :f64,
                     axis = (0.15, -1.0, 0.25), out = "run/two.csv",
                     open_faces = true)
        path = tempname() * ".toml"
        write(path, settings_toml(a))
        b = load_settings!(Settings(), path)
        for k in fieldnames(Settings)
            @test getfield(b, k) == getfield(a, k)
        end
        rm(path)
    end

    @testset "types come from the field, not the file" begin
        # TOML has one integer and one float type; the struct decides which is
        # wanted, so `resolution = 40.0` is an Int and `domain = 6` a Float64.
        s = load_settings!(Settings(), Dict("resolution" => 40.0, "domain" => 6))
        @test s.resolution === 40
        @test s.domain === 6.0
        @test load_settings!(Settings(), Dict("precision" => "f64")).precision === :f64
        @test load_settings!(Settings(), Dict("open_faces" => true)).open_faces === true
        @test load_settings!(Settings(), Dict("axis" => [1, 2, 3])).axis === (1.0, 2.0, 3.0)

        # A number that is not the integer it claims to be is refused rather
        # than rounded: 40.5 nodes per diameter is a mistake, not a request.
        @test_throws ArgumentError load_settings!(Settings(),
                                                  Dict("resolution" => 40.5))
    end

    @testset "a misspelled key is an error" begin
        # Ignored, it would run something other than what the file says while
        # the log claimed otherwise — the one failure that looks like a result.
        err = try
            load_settings!(Settings(), Dict("reslution" => 20))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("reslution", err.msg)
        @test occursin("resolution", err.msg)      # the message names the near miss
        @test_throws ArgumentError load_settings!(Settings(), tempname() * ".toml")
        @test_throws ArgumentError load_settings!(Settings(), Dict("axis" => [1.0, 2.0]))
    end

    @testset "strings survive Windows paths" begin
        # Backslashes and quotes have to come back as themselves, or a run
        # written on one machine cannot be repeated on it.
        a = Settings(out = "C:\\Users\\bk\\runs\\a \"first\" try.csv")
        b = load_settings!(Settings(), Dict{String,Any}())
        path = tempname() * ".toml"
        write(path, settings_toml(a))
        load_settings!(b, path)
        @test b.out == a.out
        rm(path)
    end

    @testset "the file is a table, and order does not matter" begin
        text = settings_toml(Settings(resolution = 21, open_faces = true))
        shuffled = join(reverse(split(rstrip(text), '\n')), '\n')
        path = tempname() * ".toml"
        write(path, shuffled)
        s = load_settings!(Settings(), path)
        @test s.resolution == 21
        @test s.open_faces
        rm(path)
    end
end
