"""
Reading a trajectory CSV back in, for analysis and plotting (§7.4, stage two).

`scripts/run_pitch.jl` writes one row per sub-cycle with a header naming every
column (`t,x,y,z,vx,...`). The set of columns has grown as the run gained
things worth keeping (§ "あとから追加の処理が可能なように残しておきたい") and
will grow again, so the reader does not hard-code a schema: it returns
whatever columns the file actually has, keyed by their header name, and a
plotting script asks for the ones it wants and fails loudly if one is missing
rather than silently plotting garbage.

No dependency is added for this. The file is one header line and one numeric
value per field with no quoting or embedded commas — `CSV.jl` would be a
correct choice for a file that needed its full generality, but this one does
not, and the project's own convention (§7.4, `postprocess/vtk.jl`) is not to
pay for what a hand-rolled fifteen lines already covers.
"""

"""
    read_pitch_csv(path) -> Dict{Symbol,Vector{Float64}}

Read a trajectory CSV written by `scripts/run_pitch.jl`. Every column becomes
one entry, keyed by its header name as a `Symbol`; all values are parsed as
`Float64` regardless of the column's natural type (`steps`, `fresh` and
`recuts` are integers in the file but are just as usable as floats for
plotting, and keeping one parse path means one thing to get wrong).

Every column comes back the same length; a short or ragged row is an error
naming the row and the column count involved, since a plot built on
mismatched columns would fail far from here with a confusing shape error, or
worse, not fail and silently pair up the wrong rows.
"""
function read_pitch_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("$path is empty — no header to read"))
    names = Symbol.(split(lines[1], ','))
    ncols = length(names)
    cols = [Float64[] for _ in 1:ncols]
    @inbounds for (n, line) in enumerate(lines[2:end])
        isempty(line) && continue         # a trailing newline leaves one
        fields = split(line, ',')
        length(fields) == ncols ||
            throw(ArgumentError("$path row $(n+1) has $(length(fields)) fields, " *
                                "expected $ncols (from the header)"))
        for c in 1:ncols
            push!(cols[c], parse(Float64, fields[c]))
        end
    end
    return Dict(names[c] => cols[c] for c in 1:ncols)
end

"""
    require_columns(data, names...)

Check that every name in `names` is a key of `data`, raising an error that
lists exactly what is missing rather than letting the first `data[:missing]`
throw a bare `KeyError` with no context. Used by plotting code before it
builds a figure, so a CSV from an older run (missing a column this script
wants) fails with one clear line instead of midway through drawing.
"""
function require_columns(data::Dict{Symbol,Vector{Float64}}, names::Symbol...)
    missing_names = [n for n in names if !haskey(data, n)]
    isempty(missing_names) ||
        throw(ArgumentError("missing column(s) $(join(missing_names, ", ")) — " *
                            "this CSV may be from an older run; columns present: " *
                            join(sort(collect(keys(data))), ", ")))
    return nothing
end

"""
    moving_average(v, window)

Centred box-car average, `window` forced odd so the average has no lag.

The per-sub-cycle force coefficients in a trajectory CSV are the instantaneous
momentum-exchange force (§4.3): they fluctuate with the turbulence around the
mean the trajectory actually responds to, and a raw plot of them is mostly
that fluctuation (`scripts/analyze_pitch.jl` draws both — the raw series
faintly underneath, this on top — so the smoothing is visibly a reading aid
and not a substitute for the data). `window` is clamped to at least 1 so a
caller cannot request `mean` of an empty range.
"""
function moving_average(v::AbstractVector{<:Real}, window::Integer)
    window = max(1, window)
    iseven(window) && (window += 1)
    n = length(v)
    out = Vector{Float64}(undef, n)
    half = window ÷ 2
    @inbounds for i in 1:n
        lo, hi = max(1, i - half), min(n, i + half)
        s = 0.0
        for k in lo:hi
            s += v[k]
        end
        out[i] = s / (hi - lo + 1)
    end
    return out
end

"""
    default_smoothing_window(n)

A `moving_average` window sized to the row count when nothing more specific is
known: about 1/40th of the run, clamped to [5, 201] so a short debug run is not
over-smoothed into a flat line and a long production run does not get a window
so wide it washes out real trends over the second half.
"""
default_smoothing_window(n::Integer) = clamp(round(Int, n / 40), 5, 201)
