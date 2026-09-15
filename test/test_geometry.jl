@testset "baseball geometry" begin
    seam = BaseballSeam(radius = 0.0374, amplitude = 0.7)
    R = seam.radius

    # Sample off the crossing points so no coordinate is exactly zero.
    offset_samples(n) = [seam_point(seam, 2π * (i - 0.5) / n) for i in 1:n]

    """Number of times the closed seam crosses the plane `component = 0`."""
    function crossings(component::Int; n::Int = 1000)
        vals = [p[component] for p in offset_samples(n)]
        @assert all(!iszero, vals)
        return count(i -> sign(vals[i]) != sign(vals[mod1(i + 1, n)]), 1:n)
    end

    @testset "curve lies on the sphere and closes" begin
        for u in range(0, 2π, length = 97)
            p = seam_point(seam, u)
            @test sqrt(p[1]^2 + p[2]^2 + p[3]^2) ≈ R rtol = 1e-12
        end
        p0 = seam_point(seam, 0.0)
        p1 = seam_point(seam, 2π)
        @test all(abs.(p0 .- p1) .< 1e-12)

        pts = seam_polyline(seam, 64)
        @test length(pts) == 65
        @test pts[end] == pts[1]
    end

    @testset "four-seam / two-seam orientation" begin
        # Spin about z: the seam crosses the spin equator (z = 0) four times per
        # revolution. Spin about x or y: twice. This is what distinguishes a
        # four-seam grip from a two-seam grip on a real ball.
        @test crossings(3) == 4     # z = 0 plane  → spin about z
        @test crossings(1) == 2     # x = 0 plane  → spin about x
        @test crossings(2) == 2     # y = 0 plane  → spin about y
    end

    @testset "symmetry: halves are directly congruent" begin
        # A 180° rotation about x maps the seam onto itself and swaps the two
        # cover pieces, exactly as for a real two-piece cover.
        poly = seam_polyline(seam, 4096)
        for u in range(0, 2π, length = 41)
            x, y, z = seam_point(seam, u)
            rotated = (x, -y, -z)
            @test BBL.distance_to_seam(rotated, poly) < 1e-5 * R
        end
    end

    @testset "no self-intersection" begin
        n = 512
        pts = [seam_point(seam, 2π * (i - 1) / n) for i in 1:n]
        # Minimum separation between points that are far apart along the curve.
        dmin = Inf
        for i in 1:n, j in 1:n
            sep = min(abs(i - j), n - abs(i - j))   # distance along the closed curve
            sep >= n ÷ 16 || continue
            d = sqrt(sum((pts[i] .- pts[j]) .^ 2))
            dmin = min(dmin, d)
        end
        @test dmin > 0.1R
    end

    @testset "seam length" begin
        L = seam_length(seam)
        @test L > 2π * R                    # longer than a great circle
        @test L < 3 * 2π * R                # but not pathologically so
        @test seam_length(seam, n = 8192) ≈ L rtol = 1e-4   # converged
    end

    geom = BaseballGeometry(diameter = 2R, seam_height = 0.00079, seam_amplitude = 0.7)

    @testset "signed distance field" begin
        h = geom.seam_height

        # Towards the poles the seam is far away, so the sphere sets the distance.
        @test sdf(geom, (0.0, 0.0, 0.3)) ≈ 0.3 - R rtol = 1e-9
        @test sdf(geom, (0.0, 0.0, -0.3)) ≈ 0.3 - R rtol = 1e-9

        # The seam passes through (R, 0, 0), so straight out along +x the nearest
        # solid is the crest of the ridge, one seam height further out.
        @test sdf(geom, (0.5, 0.0, 0.0)) ≈ 0.5 - R - h rtol = 1e-9

        # Deep inside, negative.
        @test sdf(geom, (0.0, 0.0, 0.0)) ≈ -R rtol = 1e-9
        @test sdf(geom, (0.5R, 0.0, 0.0)) < 0

        # On the crest of the ridge: the tube surface sits exactly `h` above the
        # sphere, and one further `h` out the distance is `h`.
        for u in range(0, 2π, length = 17)
            c = seam_point(seam, u)
            crest = c .* (1 + h / R)
            @test abs(sdf(geom, crest)) < 1e-5 * h
            above = c .* (1 + 2h / R)
            @test sdf(geom, above) ≈ h rtol = 1e-3
            @test sdf(geom, c) ≈ -h rtol = 1e-3       # buried half of the tube
        end

        # A point on the sphere far from any seam is on the surface.
        away = nothing
        poly = geom.polyline
        for u in range(0, 2π, length = 720)
            for v in range(-1.0, 1.0, length = 41)
                sinθ = sqrt(1 - v^2)
                p = (R * sinθ * cos(u), R * sinθ * sin(u), R * v)
                if BBL.distance_to_seam(p, poly) > 5h
                    away = p
                    break
                end
            end
            away === nothing || break
        end
        @test away !== nothing
        @test abs(sdf(geom, away)) < 1e-12

        # Eikonal property |∇ϕ| = 1 away from the medial axis.
        δ = 1e-6
        for p in ((0.06, 0.01, 0.02), (-0.05, 0.03, -0.01), (0.0, 0.045, 0.0))
            g = ntuple(3) do a
                pp = ntuple(b -> b == a ? p[b] + δ : p[b], 3)
                pm = ntuple(b -> b == a ? p[b] - δ : p[b], 3)
                (sdf(geom, pp) - sdf(geom, pm)) / 2δ
            end
            @test sqrt(sum(g .^ 2)) ≈ 1.0 rtol = 1e-4
        end
    end

    @testset "seam raises the surface by exactly its height" begin
        h = geom.seam_height
        c = seam_point(seam, 1.0)
        outward = c ./ R
        # Walk outwards along the radius through a seam point: the solid ends at R + h.
        inside(r) = sdf(geom, outward .* r) < 0
        @test inside(R + 0.9h)
        @test !inside(R + 1.1h)
    end

    @testset "voxelised volume matches the analytic sphere" begin
        dx = 2R / 48                       # 48 cells across the diameter
        dims = (64, 64, 64)
        ϕ = sdf_field(geom, dims, dx)
        v = solid_volume(ϕ, dx)
        v_exact = 4 / 3 * π * R^3
        @test v ≈ v_exact rtol = 0.02      # whole-cell counting on a coarse grid

        # The field agrees with the pointwise sdf wherever the band is active.
        center = (dims .+ 1) ./ 2
        for idx in ((30, 32, 32), (32, 40, 33), (20, 20, 20), (5, 5, 5))
            p = ((idx[1] - center[1]) * dx, (idx[2] - center[2]) * dx, (idx[3] - center[3]) * dx)
            if abs(sphere_sdf(geom, p)) <= geom.seam_height + 3dx
                @test ϕ[idx...] ≈ sdf(geom, p) rtol = 1e-12
            else
                @test ϕ[idx...] ≈ sphere_sdf(geom, p) rtol = 1e-12
            end
        end

        # Adding the seam can only add solid.
        smooth = BaseballGeometry(diameter = 2R, seam_height = 0.0)
        @test solid_volume(sdf_field(smooth, dims, dx), dx) <= v
    end
end
