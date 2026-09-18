@testset "definition order" begin
    # Julia resolves the types in a method *signature* when the method is
    # defined, so naming a type that appears later in the load order is a
    # precompilation error — not a runtime one, and not one any test can reach
    # by calling something. For `ext/`, which needs a GPU to load at all, that
    # means development without one cannot catch it by running: the first
    # symptom is the extension failing to precompile on the machine that has
    # the card. So it is checked by reading instead.
    #
    # It has bitten twice: `flow_wall(::DeviceRotatingFlow)` placed a hundred
    # lines above the struct, and `aa_run_walls!` annotating a keyword with
    # `OpenChannel` from a file included after it.

    root = dirname(@__DIR__)

    """Struct definitions in a file, by name and line."""
    function struct_defs(path)
        defs = Dict{String,Int}()
        for (n, l) in enumerate(eachline(path))
            m = match(r"^\s*(?:mutable\s+)?struct\s+([A-Za-z_][A-Za-z0-9_!]*)", l)
            m === nothing || (defs[m.captures[1]] = n)
        end
        return defs
    end

    """
    Type names appearing in method signatures, by name and line.

    A signature is a `function` header up to where its parentheses balance, or
    a one-line `f(args) = body` up to the `=`. Uses inside a function *body*
    are resolved when the function runs, so they are not the error being looked
    for and are skipped.
    """
    function signature_uses(path)
        uses = Tuple{String,Int}[]
        insig = false
        depth = 0
        for (n, l) in enumerate(eachline(path))
            code = replace(l, r"#.*$" => "")
            if !insig
                if occursin(r"^\s*function\s", code)
                    insig = true
                    depth = 0
                elseif occursin(r"^\s*[A-Za-z_][\w.!]*\(.*\)\s*(where\s.*)?=[^=]", code)
                    head = first(split(code, "="))
                    for m in eachmatch(r"::\s*([A-Za-z_][A-Za-z0-9_!]*)", head)
                        push!(uses, (m.captures[1], n))
                    end
                    continue
                else
                    continue
                end
            end
            for m in eachmatch(r"::\s*([A-Za-z_][A-Za-z0-9_!]*)", code)
                push!(uses, (m.captures[1], n))
            end
            depth += count(==('('), code) - count(==(')'), code)
            insig = depth > 0
        end
        return uses
    end

    @testset "the detector detects" begin
        # A lint nobody has seen fail is a lint nobody should trust.
        bad = tempname() * ".jl"
        write(bad, """
              f(x::Later) = x
              struct Later
                  a::Int
              end
              """)
        defs = struct_defs(bad)
        uses = signature_uses(bad)
        @test defs["Later"] == 2
        @test ("Later", 1) in uses
        @test any(u -> haskey(defs, u[1]) && u[2] < defs[u[1]], uses)

        good = tempname() * ".jl"
        write(good, """
              struct Early
                  a::Int
              end
              f(x::Early) = x
              function g(y::Early,
                         z::Int)
                  return y
              end
              """)
        dg, ug = struct_defs(good), signature_uses(good)
        @test ("Early", 4) in ug
        @test ("Early", 5) in ug            # a signature spanning two lines
        @test !any(u -> haskey(dg, u[1]) && u[2] < dg[u[1]], ug)
        rm(bad); rm(good)
    end

    @testset "the extension defines before it dispatches" begin
        for path in filter(f -> endswith(f, ".jl"), readdir(joinpath(root, "ext"); join = true))
            defs = struct_defs(path)
            for (name, line) in signature_uses(path)
                haskey(defs, name) || continue
                @test line > defs[name] ||
                      error("$(basename(path)):$line dispatches on `$name`, " *
                            "which is defined at line $(defs[name])")
            end
        end
    end

    @testset "the package defines before it dispatches" begin
        # Across files, the order is the include list rather than the file
        # system's.
        main = joinpath(root, "src", "BreakingBallLBM.jl")
        order = String[]
        for l in eachline(main)
            m = match(r"^\s*include\(\"([^\"]+)\"\)", l)
            m === nothing || push!(order, joinpath(root, "src", m.captures[1]))
        end
        @test length(order) > 10

        defs = Dict{String,Tuple{Int,Int}}()
        for (idx, path) in enumerate(order), (name, line) in struct_defs(path)
            haskey(defs, name) || (defs[name] = (idx, line))
        end
        for (idx, path) in enumerate(order), (name, line) in signature_uses(path)
            haskey(defs, name) || continue
            di, dl = defs[name]
            @test (idx, line) > (di, dl) ||
                  error("$(basename(path)):$line dispatches on `$name`, defined " *
                        "in $(basename(order[di])):$dl, which is included later")
        end
    end
end
