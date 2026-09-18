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

    # One copy off the device, of the crop only.
    block = Array(g[ranges[1], ranges[2], ranges[3], :])
    mask = Array(solid[ranges[1], ranges[2], ranges[3]])

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
