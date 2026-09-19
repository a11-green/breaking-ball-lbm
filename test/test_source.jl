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

    @testset "the device backend answers every question the driver asks" begin
        # A coupled run holds one of three device handles and the driver asks it
        # all the same questions. A handle missing one of them is a MethodError
        # or a FieldError *at the first report* — which, after a spin-up, is
        # several minutes into a run that is then thrown away. It happened twice
        # (`flow_wall` and `flow_recuts` on `DeviceRefinedFlow`), so it is
        # checked here, where a GPU is not needed to notice.
        path = joinpath(root, "ext", "BreakingBallLBMCUDAExt.jl")
        src = read(path, String)

        """The type names appearing in any signature of `fname`, across lines."""
        function signature_types(fname)
            types = Set{String}()
            pat = Regex("(?:function\\s+)?BreakingBallLBM\\." * replace(fname, "!" => "!") * "\\(")
            for m in eachmatch(pat, src)
                i = m.offset + length(m.match) - 1   # the opening paren
                depth = 0
                j = i
                while j <= lastindex(src)
                    c = src[j]
                    c == '(' && (depth += 1)
                    if c == ')'
                        depth -= 1
                        depth == 0 && break
                    end
                    j = nextind(src, j)
                end
                for t in eachmatch(r"::\s*([A-Za-z_][A-Za-z0-9_]*)", src[i:j])
                    push!(types, t.captures[1])
                end
            end
            return types
        end

        asked = ["flow_fluid_count", "flow_mean_velocity", "advance_flow!",
                 "flow_wall", "flow_recuts"]
        # A handle that carries a patch also answers where the ball's footprint
        # is on the level the box mean is taken over, and how long a sub-cycle
        # may be — the driver calls both.
        patched = [asked; "maybe_recut!"; "max_substeps"; "flow_coarse_solid"]
        for (handle, fns) in (("DeviceFlow", asked),
                              ("DeviceRotatingFlow", [asked; "maybe_recut!"]),
                              ("DeviceRefinedFlow", patched),
                              ("DeviceChainFlow", patched))
            for f in fns
                @test handle in signature_types(f) ||
                      error("`$f` has no method naming `$handle` in $(basename(path))")
            end
        end

        # And the check has to be able to fail, or it is decoration.
        @test !("DeviceFlow" in signature_types("no_such_function"))
        @test "DeviceTwoGrid" in signature_types("flow_mean_velocity")   # spans lines
    end

    @testset "every docstring documents something" begin
        # A docstring has to be followed by a definition. Insert one between an
        # existing docstring and the thing it described and the first now
        # documents the second *string*, which Julia refuses at load time —
        # again only where the file is loaded, which for `ext/` means the
        # machine with the GPU. It has happened twice.
        function orphaned(path)
            ex = Meta.parse("begin\n" * read(path, String) * "\nend")
            found = String[]
            walk(e) = begin
                e isa Expr || return
                if e.head === :macrocall && length(e.args) >= 4 &&
                   e.args[1] === GlobalRef(Core, Symbol("@doc"))
                    e.args[4] isa AbstractString &&
                        push!(found, first(split(strip(String(e.args[3])), '\n')))
                end
                foreach(walk, e.args)
            end
            walk(ex)
            return found
        end

        for dir in ("src", "ext"), path in readdir(joinpath(root, dir); join = true)
            endswith(path, ".jl") || continue
            for d in orphaned(path)
                @test false ||
                      error("$(basename(path)): a docstring starting \"$d\" " *
                            "documents a string, not a definition")
            end
            @test isempty(orphaned(path))
        end
        for sub in ("core", "boundary", "geometry", "refine", "trajectory",
                    "postprocess", "turbulence", "validation")
            for path in readdir(joinpath(root, "src", sub); join = true)
                endswith(path, ".jl") && @test isempty(orphaned(path))
            end
        end
    end

    @testset "the extension loads only what it may" begin
        # An extension can `using` its parent package, the parent's [deps] and
        # the [weakdeps] that trigger it — nothing else, not even a standard
        # library. Loading anything else is a precompilation error on the
        # machine with the GPU and nowhere else, which is the third time that
        # has happened here (`Random`, for a benchmark's shuffle).
        proj = read(joinpath(root, "Project.toml"), String)
        section(name) = begin
            m = match(Regex("\\[" * name * "\\]\\n((?:[^\\[]*\\n)*)"), proj)
            m === nothing ? String[] :
            [strip(first(split(l, "="))) for l in split(m.captures[1], '\n') if occursin("=", l)]
        end
        allowed = Set(vcat(section("deps"), section("weakdeps"), ["BreakingBallLBM"]))

        for path in filter(f -> endswith(f, ".jl"),
                           readdir(joinpath(root, "ext"); join = true))
            for (n, l) in enumerate(eachline(path))
                m = match(r"^\s*(?:using|import)\s+([A-Za-z_][A-Za-z0-9_]*)", l)
                m === nothing && continue
                mod = m.captures[1]
                @test mod in allowed ||
                      error("$(basename(path)):$n loads `$mod`, which is not in " *
                            "Project.toml's [deps] or [weakdeps] — an extension " *
                            "cannot load it. Known: $(join(sort(collect(allowed)), ", "))")
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
