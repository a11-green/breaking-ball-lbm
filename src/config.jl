"""
Settings files, for any mutable struct whose fields are scalars or triples.

**Why this is in the library and not in the script that uses it.** It decides
what physics a run performs — a setting read as the wrong number, or a
misspelled key quietly ignored, changes the answer and leaves a log that says
something else happened. That is the kind of code the project tests, and code in
a script is code the test suite cannot reach.

**TOML rather than JSON.** Julia ships a TOML parser in its standard library and
no JSON one. JSON would mean either a package for whoever runs this, or a
hand-rolled parser — and a hand-rolled parser for a format with escapes and a
number grammar is exactly the code that is wrong in a way nobody notices until
it reads the wrong number. TOML also takes comments, which a settings file
wants.

**The round trip is the point.** `settings_toml` emits what `load_settings!`
accepts, so a run can print its own resolved inputs into its log, write them
beside its results, and be repeated from either.
"""

"""
    coerce_setting(name, current, v)

A value from a file, converted to the type the field already holds. The existing
value is the schema: there is no separate declaration to keep in agreement with
the struct.
"""
function coerce_setting(name::Symbol, current, v)
    try
        current isa Bool && return Bool(v)
        current isa Integer && return typeof(current)(v)
        current isa AbstractFloat && return typeof(current)(v)
        current isa Symbol && return Symbol(v)
        current isa AbstractString && return String(v)
        if current isa Vector{<:Real}
            # A single number where a list is expected is a list of one, which
            # keeps `refine = 3.0` meaning what it did before `refine` could
            # name several levels.
            E = eltype(current)
            return v isa AbstractVector ? E[E(x) for x in v] : E[E(v)]
        end
        if current isa NTuple{3,<:Real}
            length(v) == 3 ||
                throw(ArgumentError("`$name` needs three numbers, got $(length(v))"))
            E = eltype(current)
            return (E(v[1]), E(v[2]), E(v[3]))
        end
    catch err
        err isa ArgumentError && rethrow()
        throw(ArgumentError("setting `$name` cannot take $(repr(v))"))
    end
    throw(ArgumentError("setting `$name` has a type this reader does not handle: " *
                        "$(typeof(current))"))
end

"""
    load_settings!(obj, path)

Apply a TOML file to `obj`, field by field.

**An unknown key is an error, not a warning.** A typo that is ignored runs
something other than what the file says, and the file is the record of what was
run; the message lists the keys that exist, since the usual cause is a near
miss.
"""
function load_settings!(obj, path::AbstractString)
    isfile(path) || throw(ArgumentError("no such settings file: $path"))
    return load_settings!(obj, TOML.parsefile(path); source = path)
end

function load_settings!(obj, table::AbstractDict; source::AbstractString = "the table")
    keys_ = fieldnames(typeof(obj))
    for (k, v) in table
        sym = Symbol(k)
        sym in keys_ || throw(ArgumentError(
            "unknown setting `$k` in $source — known settings are: " *
            join(sort(string.(collect(keys_))), ", ")))
        setfield!(obj, sym, coerce_setting(sym, getfield(obj, sym), v))
    end
    return obj
end

toml_value(v::Bool) = string(v)
toml_value(v::Integer) = string(v)
toml_value(v::AbstractFloat) = string(v)
toml_value(v::Symbol) = "\"" * string(v) * "\""
toml_value(v::AbstractString) =
    "\"" * replace(String(v), "\\" => "\\\\", "\"" => "\\\"") * "\""
toml_value(v::NTuple{3,<:Real}) = "[" * join(v, ", ") * "]"
toml_value(v::AbstractVector{<:Real}) = "[" * join(v, ", ") * "]"

"""
    settings_toml(obj; width = 20)

`obj` as TOML — the same text [`load_settings!`](@ref) accepts, so writing it
out and reading it back is the identity.
"""
function settings_toml(obj; width::Integer = 20)
    io = IOBuffer()
    for k in fieldnames(typeof(obj))
        println(io, rpad(string(k), width), " = ", toml_value(getfield(obj, k)))
    end
    return String(take!(io))
end
