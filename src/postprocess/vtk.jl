"""
Flow-field snapshots, in a format ParaView opens, with no new dependency.

**Why by hand rather than through `WriteVTK.jl`.** The legacy VTK binary format
is a text header and a big-endian block of floats, which is fifty lines here and
one more package for whoever runs this to install there. The project's only
dependencies are the two weak ones the GPU needs, and a viewer format is not a
good reason to break that.

**Why this exists at all.** Until now a run produced a trajectory and a set of
force coefficients, and the flow that generated them was discarded at the end of
the last sub-cycle. That is enough to check a break, and no use at all for the
question the project is actually about — whether the seam is moving the
separation point — because that question is answered by looking at the flow.

**Size is the whole design constraint.** A production grid is 320³ nodes; five
fields of it in single precision is a gigabyte per snapshot, so a run that wrote
one every few sub-cycles would fill a disk before it reached the plate. Hence
`crop` and `stride`: the interesting region is the few diameters around the ball
(which, in the ball-following frame, is the middle of the box and stays there),
and a wake seen every second node is still a wake.
"""

"""
    write_vtk(path, scalars, vectors; spacing = 1, origin = (0, 0, 0), title = "")

Legacy VTK `STRUCTURED_POINTS`, binary, written as `Float32` whatever the
solver's precision — a viewer cannot show more than that, and doubling the file
to store it would be paid on every snapshot.

`scalars` and `vectors` are `name => array` pairs, shaped `(nx, ny, nz)` and
`(nx, ny, nz, 3)`. All must agree on the first three dimensions.
"""
function write_vtk(path::AbstractString,
                   scalars::AbstractVector{<:Pair{<:AbstractString,<:AbstractArray}},
                   vectors::AbstractVector{<:Pair{<:AbstractString,<:AbstractArray}} =
                       Pair{String,Array{Float32,4}}[];
                   spacing::Real = 1, origin::NTuple{3,<:Real} = (0, 0, 0),
                   title::AbstractString = "breaking-ball-lbm")
    isempty(scalars) && isempty(vectors) &&
        throw(ArgumentError("a snapshot with no fields in it is not worth writing"))
    dims = isempty(scalars) ? size(first(vectors).second)[1:3] : size(first(scalars).second)
    for (name, a) in scalars
        size(a) == dims || throw(DimensionMismatch("$name is $(size(a)), expected $dims"))
    end
    for (name, a) in vectors
        size(a)[1:3] == dims && size(a, 4) == 3 ||
            throw(DimensionMismatch("$name is $(size(a)), expected $((dims..., 3))"))
    end
    nx, ny, nz = dims
    n = nx * ny * nz

    # `hton` is the whole of the endianness story: legacy VTK binary is
    # big-endian by definition and every machine this runs on is not.
    put(io, x) = write(io, hton(Float32(x)))

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, title)
        println(io, "BINARY")
        println(io, "DATASET STRUCTURED_POINTS")
        println(io, "DIMENSIONS $nx $ny $nz")
        println(io, "ORIGIN $(origin[1]) $(origin[2]) $(origin[3])")
        println(io, "SPACING $spacing $spacing $spacing")
        println(io, "POINT_DATA $n")
        for (name, a) in scalars
            println(io, "SCALARS $name float 1")
            println(io, "LOOKUP_TABLE default")
            @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
                put(io, a[i, j, k])
            end
        end
        for (name, a) in vectors
            println(io, "VECTORS $name float")
            @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
                put(io, a[i, j, k, 1]); put(io, a[i, j, k, 2]); put(io, a[i, j, k, 3])
            end
        end
    end
    return path
end

"""
    snapshot_box(dims, centre, halfwidth)

The index range a crop covers, clamped to the grid and widened by one node so
the gradients at its edge have neighbours to use.

Returns `(ranges, trim)`: the range to copy, and the offset to drop afterwards,
which is one except where the crop ran into the edge of the grid and there was
nothing to widen into.
"""
function snapshot_box(dims::NTuple{3,<:Integer}, centre::NTuple{3,<:Real},
                      halfwidth::Real)
    ranges = ntuple(3) do d
        lo = max(1, floor(Int, centre[d] - halfwidth) - 1)
        hi = min(dims[d], ceil(Int, centre[d] + halfwidth) + 1)
        lo:hi
    end
    trim = ntuple(d -> (first(ranges[d]) == 1 ? 0 : 1,
                        last(ranges[d]) == dims[d] ? 0 : 1), 3)
    return ranges, trim
end

"""
    write_snapshot(path, g, solid; crop, stride, spacing, origin, fields)

Compute the fields worth looking at from an AA-pattern array in the even layout
and write them.

`g` may live on the device: only the cropped block is brought back, which is the
point of cropping at all — a production array is gigabytes and a snapshot of the
near wake is tens of megabytes.

`crop` is a half-width in nodes about `centre` (by default the middle of the
grid, which in the ball-following frame is where the ball is); `nothing` takes
everything. `stride` thins what is written after the derivatives are taken, so
the gradients are still the grid's own rather than a coarser grid's.
"""
function write_snapshot(path::AbstractString, g, solid::AbstractArray{Bool,3};
                        crop::Union{Real,Nothing} = nothing, stride::Integer = 1,
                        spacing::Real = 1, origin::NTuple{3,<:Real} = (0, 0, 0),
                        centre::Union{NTuple{3,<:Real},Nothing} = nothing,
                        title::AbstractString = "breaking-ball-lbm")
    dims = size(g)[1:3]
    size(solid) == dims ||
        throw(DimensionMismatch("solid is $(size(solid)), the lattice is $dims"))
    stride >= 1 || throw(ArgumentError("stride must be at least one, got $stride"))

    c = centre === nothing ? (dims .+ 1) ./ 2 : centre
    ranges, trim = crop === nothing ?
        (ntuple(d -> 1:dims[d], 3), ntuple(_ -> (0, 0), 3)) :
        snapshot_box(dims, c, crop)

    # One copy off the device, of the crop only — or of the whole thing when
    # that is what was asked for, without slicing first: indexing a device array
    # builds the slice *on the device*, and a full-size temporary is the one
    # allocation an 8 GB card holding the run has no room for.
    whole = all(d -> ranges[d] == 1:dims[d], 1:3)
    block = whole ? Array(g) : Array(g[ranges[1], ranges[2], ranges[3], :])
    mask = whole ? Array(solid) : Array(solid[ranges[1], ranges[2], ranges[3]])

    u = velocity_field(block; mask = mask)
    q = q_criterion(u)
    ω = vorticity(u)
    ρ = density_field(block; mask = mask)

    # Drop the halo the gradients needed, then thin.
    keep = ntuple(3) do d
        (1 + trim[d][1]):stride:(size(u, d) - trim[d][2])
    end
    sub3(a) = a[keep[1], keep[2], keep[3]]
    sub4(a) = a[keep[1], keep[2], keep[3], :]
    speed = sqrt.(sum(abs2, sub4(u); dims = 4)[:, :, :, 1])
    vort = sqrt.(sum(abs2, sub4(ω); dims = 4)[:, :, :, 1])

    off = ntuple(d -> origin[d] + spacing * (first(ranges[d]) - 1 + trim[d][1]), 3)
    return write_vtk(path,
                     ["density" => sub3(ρ), "speed" => speed,
                      "vorticity_magnitude" => vort, "q_criterion" => sub3(q),
                      "solid" => Float32.(sub3(mask))],
                     ["velocity" => sub4(u), "vorticity" => sub4(ω)];
                     spacing = spacing * stride, origin = off, title = title)
end

"""
    write_polylines(path, lines; names, title)

Legacy VTK `POLYDATA`, binary, holding one or more polylines given as vectors of
`(x, y, z)` in the same world coordinates [`write_snapshot`](@ref) uses.

**Why a separate file rather than a field in the volume.** A sphere looks the
same at every orientation, so the flow snapshots alone show a spinning ball as a
still picture — and where the seam is relative to the airflow is the entire
subject (§4.1). Marking seam cells in the volume would not work either: the seam
stands 0.79 mm proud of the cover and the deepest level's spacing is close to
that, so at the resolutions this project can reach the seam is a sub-cell
feature that no volume field can draw. As a curve it is exact at any resolution
and costs a few kilobytes a frame.

Each line carries an integer `line_id` cell scalar so a viewer can colour or
filter them apart — the seam and the spin axis go in one file and want different
widths. `names` is written into the title line, which is where a reader opening
the file by hand will look to find out which id is which.
"""
function write_polylines(path::AbstractString,
                         lines::AbstractVector{<:AbstractVector{<:NTuple{3,<:Real}}};
                         names::AbstractVector{<:AbstractString} = String[],
                         title::AbstractString = "breaking-ball-lbm")
    isempty(lines) && throw(ArgumentError("no lines to write"))
    any(isempty, lines) && throw(ArgumentError("a line with no points cannot be drawn"))
    isempty(names) || length(names) == length(lines) ||
        throw(DimensionMismatch("$(length(names)) names for $(length(lines)) lines"))

    npoints = sum(length, lines)
    header = isempty(names) ? title : string(title, " [", join(names, ", "), "]")
    put(io, x) = write(io, hton(Float32(x)))

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, header)
        println(io, "BINARY")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS $npoints float")
        for line in lines, p in line
            put(io, p[1]); put(io, p[2]); put(io, p[3])
        end
        # LINES is a count, a total size, and then one (n, indices...) record per
        # line — indices into the flat point list, zero-based.
        println(io)
        println(io, "LINES $(length(lines)) $(npoints + length(lines))")
        base = 0
        for line in lines
            write(io, hton(Int32(length(line))))
            for k in 0:(length(line) - 1)
                write(io, hton(Int32(base + k)))
            end
            base += length(line)
        end
        println(io)
        println(io, "CELL_DATA $(length(lines))")
        println(io, "SCALARS line_id float 1")
        println(io, "LOOKUP_TABLE default")
        for k in 1:length(lines)
            put(io, k)
        end
    end
    return path
end

"""
    read_polylines(path) -> (lines, ids, title)

The inverse of [`write_polylines`](@ref): read the points, line polylines and
`line_id` cell scalar back out of a file it wrote. `lines[k]` is a
`Vector{NTuple{3,Float64}}` and `ids[k]` its `line_id` (as written, `k` itself
unless a caller asked for something else).

This exists so a later pass can find, say, "the seam" inside a frame someone
else's run already wrote — without re-running the simulation or re-deriving
the geometry — by reading its own output back rather than assuming a naming
convention. [`ball_from_seam`](@ref) is the reason it was written.
"""
function read_polylines(path::AbstractString)
    data = read(path)
    pos = Ref(1)
    textline() = begin
        e = findnext(==(UInt8('\n')), data, pos[])
        e === nothing && throw(ArgumentError("$path: truncated header (no more newlines)"))
        s = String(data[pos[]:(e - 1)])
        pos[] = e + 1
        s
    end
    getfloats(n) = begin
        stop = pos[] + 4n - 1
        stop <= length(data) ||
            throw(ArgumentError("$path: truncated — expected $(4n) more bytes at byte $(pos[])"))
        v = ntoh.(reinterpret(Float32, data[pos[]:stop]))
        pos[] += 4n
        v
    end
    header = textline()
    header == "# vtk DataFile Version 3.0" ||
        throw(ArgumentError("$path: not a legacy VTK file (got \"$header\")"))
    title = textline()
    textline() == "BINARY" || throw(ArgumentError("$path: not written as BINARY"))
    textline() == "DATASET POLYDATA" ||
        throw(ArgumentError("$path: not POLYDATA (read_polylines only reads that)"))

    head = split(textline())
    (length(head) == 3 && head[1] == "POINTS" && head[3] == "float") ||
        throw(ArgumentError("$path: expected \"POINTS n float\", got \"$(join(head, ' '))\""))
    npoints = parse(Int, head[2])
    coords = ntoh.(reinterpret(Float32, data[pos[]:(pos[] + 12npoints - 1)]))
    pos[] += 12npoints
    points = [(Float64(coords[3k-2]), Float64(coords[3k-1]), Float64(coords[3k]))
              for k in 1:npoints]

    isempty(textline()) || throw(ArgumentError("$path: expected a blank line after POINTS"))
    head = split(textline())
    (length(head) == 3 && head[1] == "LINES") ||
        throw(ArgumentError("$path: expected \"LINES nlines total\", got \"$(join(head, ' '))\""))
    nlines = parse(Int, head[2])

    read_int32() = begin
        v = ntoh(reinterpret(Int32, data[pos[]:(pos[] + 3)])[1])
        pos[] += 4
        Int(v)
    end
    lines = Vector{Vector{NTuple{3,Float64}}}(undef, nlines)
    for k in 1:nlines
        len = read_int32()
        lines[k] = [points[read_int32() + 1] for _ in 1:len]   # written zero-based
    end

    isempty(textline()) || throw(ArgumentError("$path: expected a blank line after LINES"))
    head = split(textline())
    (length(head) == 2 && head[1] == "CELL_DATA" && parse(Int, head[2]) == nlines) ||
        throw(ArgumentError("$path: expected \"CELL_DATA $nlines\", got \"$(join(head, ' '))\""))
    startswith(textline(), "SCALARS line_id") ||
        throw(ArgumentError("$path: expected the line_id SCALARS block"))
    textline() == "LOOKUP_TABLE default" ||
        throw(ArgumentError("$path: expected \"LOOKUP_TABLE default\""))
    ids = Int.(getfloats(nlines))

    return lines, ids, title
end

"""
    ball_from_seam(lines, ids)

The ball's centre and radius, read off the seam curve rather than passed in
separately — so a later pass never has to know what scale or radius an earlier
one drew the seam at.

`write_polylines` gives the seam `line_id == 1` (the order `run_pitch.jl` calls
it in: seam first, spin axis second — see [`read_polylines`](@ref)). Every
point on it lies on the ball's surface by construction (`seam_world` rotates
and translates a curve that starts on a unit sphere), so the centroid is the
centre and the mean distance to it is the radius. The mean rather than any one
point's distance is what makes this insensitive to the single-precision
round-trip through the file.
"""
function ball_from_seam(lines::AbstractVector{<:AbstractVector{<:NTuple{3,<:Real}}},
                        ids::AbstractVector{<:Integer})
    k = findfirst(==(1), ids)
    k === nothing &&
        throw(ArgumentError("no line has line_id == 1 (the seam) — ids present: $ids"))
    seam = lines[k]
    isempty(seam) && throw(ArgumentError("the seam line has no points"))
    # `seam_polyline` closes the curve by repeating its first point at the end
    # (so it draws as a loop, not an arc with a gap); left in, that one point
    # would be counted twice and pull the mean toward it by O(R/n) — small
    # (a fraction of a millimetre at the usual few hundred samples) but free to
    # remove, since the repeat is bit-identical after the same transform and
    # the same Float32 round trip put it there.
    if length(seam) > 1 && seam[1] == seam[end]
        seam = @view seam[1:end-1]
    end
    n = length(seam)
    cx = sum(p[1] for p in seam) / n
    cy = sum(p[2] for p in seam) / n
    cz = sum(p[3] for p in seam) / n
    r = sum(sqrt((p[1] - cx)^2 + (p[2] - cy)^2 + (p[3] - cz)^2) for p in seam) / n
    return (cx, cy, cz), r
end

"""
    sphere_mesh(center, radius; slices = 24, stacks = 16) -> (points, triangles)

A UV-sphere as a triangle mesh: `stacks + 1` rings of `slices` points each
(pole to pole), triangulated with the two polar rings collapsed to a point and
their would-be degenerate triangles dropped rather than written as zero-area
ones. `points[i]` is an `(x, y, z)` and `triangles[j]` is `(a, b, c)`,
one-based indices into `points` — ready for [`write_triangle_mesh`](@ref).

A sphere has no preferred axis, so unlike the seam this needs no orientation:
the mesh looks the same whichever pole `slices`/`stacks` happen to run through.
"""
function sphere_mesh(center::NTuple{3,<:Real}, radius::Real;
                     slices::Integer = 24, stacks::Integer = 16)
    slices >= 3 || throw(ArgumentError("slices must be at least 3, got $slices"))
    stacks >= 2 || throw(ArgumentError("stacks must be at least 2, got $stacks"))
    radius > 0 || throw(ArgumentError("radius must be positive, got $radius"))

    T = promote_type(Float64, typeof(float(radius)))
    cx, cy, cz = T.(center)
    R = T(radius)

    # Ring i (0 at the north pole, stacks at the south pole) holds `slices`
    # points at co-latitude φ = π i / stacks; ring 0 and ring `stacks` each
    # collapse to a single physical point, repeated `slices` times so every
    # ring has the same point count and the index arithmetic below stays
    # uniform. `idx` wraps `j` so the last slice joins back to the first.
    idx(i, j) = i * slices + mod(j, slices) + 1
    points = Vector{NTuple{3,T}}(undef, (stacks + 1) * slices)
    for i in 0:stacks
        φ = T(pi) * i / stacks
        sφ, cφ = sincos(φ)
        for j in 0:(slices - 1)
            θ = 2 * T(pi) * j / slices
            sθ, cθ = sincos(θ)
            points[idx(i, j)] = (cx + R * sφ * cθ, cy + R * sφ * sθ, cz + R * cφ)
        end
    end

    triangles = Vector{NTuple{3,Int}}()
    sizehint!(triangles, 2 * slices * (stacks - 1))
    for i in 0:(stacks - 1), j in 0:(slices - 1)
        a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
        # At i == 0, a and b are both the (repeated) north pole: that triangle
        # has zero area and is dropped, keeping only the cap triangle (a,c,d).
        # The mirror image holds at the south pole, i == stacks - 1.
        i == 0 || push!(triangles, (a, b, c))
        i == stacks - 1 || push!(triangles, (a, c, d))
    end
    return points, triangles
end

"""
    write_triangle_mesh(path, points, triangles; title)

Legacy VTK `POLYDATA`, binary, holding a triangle mesh — the same file family
as [`write_polylines`](@ref), with a `POLYGONS` section instead of `LINES`.
`triangles` are one-based indices into `points`, written zero-based as the
format requires.
"""
function write_triangle_mesh(path::AbstractString,
                             points::AbstractVector{<:NTuple{3,<:Real}},
                             triangles::AbstractVector{<:NTuple{3,<:Integer}};
                             title::AbstractString = "breaking-ball-lbm")
    isempty(points) && throw(ArgumentError("no points to write"))
    isempty(triangles) && throw(ArgumentError("no triangles to write"))
    n = length(points)
    for t in triangles
        all(1 <= i <= n for i in t) ||
            throw(ArgumentError("triangle $t indexes outside 1:$n"))
    end
    put(io, x) = write(io, hton(Float32(x)))

    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, title)
        println(io, "BINARY")
        println(io, "DATASET POLYDATA")
        println(io, "POINTS $n float")
        for p in points
            put(io, p[1]); put(io, p[2]); put(io, p[3])
        end
        println(io)
        println(io, "POLYGONS $(length(triangles)) $(4 * length(triangles))")
        for (a, b, c) in triangles
            write(io, hton(Int32(3)))
            write(io, hton(Int32(a - 1)))
            write(io, hton(Int32(b - 1)))
            write(io, hton(Int32(c - 1)))
        end
        println(io)
    end
    return path
end

"""
    write_ball(path, center, radius; slices = 24, stacks = 16, title)

[`sphere_mesh`](@ref) followed by [`write_triangle_mesh`](@ref), for the common
case of just wanting the file.
"""
function write_ball(path::AbstractString, center::NTuple{3,<:Real}, radius::Real;
                    slices::Integer = 24, stacks::Integer = 16,
                    title::AbstractString = "breaking-ball-lbm")
    points, triangles = sphere_mesh(center, radius; slices = slices, stacks = stacks)
    return write_triangle_mesh(path, points, triangles; title = title)
end

"""
    read_vtk_series(path) -> Vector{NamedTuple{(:name, :time),Tuple{String,Float64}}}

Read back the `{"name": ..., "time": ...}` list a `.vtk.series` file holds
(`scripts/run_pitch.jl` writes one alongside every frame sequence).

Not a general JSON parser — this project's own writer is the only thing that
produces these files, in exactly this layout, so a few regular expressions
over the two fields it actually has are what a real parser would amount to
here anyway, without adding a dependency to read back three lines of a file
this project wrote in the first place.
"""
function read_vtk_series(path::AbstractString)
    text = read(path, String)
    names = [String(m.captures[1]) for m in eachmatch(r"\"name\"\s*:\s*\"([^\"]+)\"", text)]
    times = [parse(Float64, m.captures[1])
             for m in eachmatch(r"\"time\"\s*:\s*([-0-9.eE+]+)", text)]
    length(names) == length(times) ||
        throw(ArgumentError("$path: found $(length(names)) names but $(length(times)) " *
                            "times — not a series file this reader recognises"))
    return [(name = n, time = t) for (n, t) in zip(names, times)]
end
