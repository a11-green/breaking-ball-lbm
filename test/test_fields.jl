@testset "fields for the viewer" begin
    T = Float64
    dims = (24, 24, 24)

    """A velocity field built directly, so the derivative operators are tested alone."""
    function build(f)
        u = Array{T,4}(undef, dims..., 3)
        for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
            ux, uy, uz = f(T(i - 1), T(j - 1), T(k - 1))
            u[i, j, k, 1] = ux; u[i, j, k, 2] = uy; u[i, j, k, 3] = uz
        end
        return u
    end
    interior(a) = a[3:end-2, 3:end-2, 3:end-2, :]
    interior3(a) = a[3:end-2, 3:end-2, 3:end-2]

    @testset "a uniform flow has no vorticity and no Q" begin
        u = build((x, y, z) -> (0.03, -0.02, 0.01))
        @test maximum(abs.(vorticity(u))) < 1e-15
        @test maximum(abs.(q_criterion(u))) < 1e-15
        @test magnitude(u)[5, 5, 5] ≈ sqrt(0.03^2 + 0.02^2 + 0.01^2) rtol = 1e-14
    end

    @testset "solid-body rotation gives twice the rotation rate" begin
        # u = Ω × r with Ω along z. Central differences are exact on a linear
        # field, so this is a test of the operator and not of its accuracy — but
        # only away from the edges, since a linear field is not periodic and the
        # wrap differences across the jump.
        Ω = 0.004
        u = build((x, y, z) -> (-Ω * (y - 11.5), Ω * (x - 11.5), 0.0))
        ω = vorticity(u)
        @test maximum(abs.(interior(ω)[:, :, :, 3] .- 2Ω)) < 1e-14
        @test maximum(abs.(interior(ω)[:, :, :, 1])) < 1e-15
        # No strain at all, so Q is the whole of the rotation: Ω².
        @test maximum(abs.(interior3(q_criterion(u)) .- Ω^2)) < 1e-16
    end

    @testset "pure shear has Q of the opposite sign" begin
        # u_x = γ z is half rotation and half strain, and they cancel exactly:
        # a shear layer is not a vortex, which is the whole reason for Q.
        γ = 0.003
        u = build((x, y, z) -> (γ * z, 0.0, 0.0))
        @test maximum(abs.(interior3(q_criterion(u)))) < 1e-16
        @test maximum(abs.(interior(vorticity(u))[:, :, :, 2] .- γ)) < 1e-14
    end

    @testset "Taylor-Green vorticity converges at second order" begin
        # ω_z = 2 u₀ k cos(kx) cos(ky) for the embedded 2-D vortex.
        errs = Float64[]
        for n in (16, 32, 64)
            k = 2π / n
            u0 = 0.05
            d = (n, n, 4)
            u = Array{T,4}(undef, d..., 3)
            for c in 1:d[3], b in 1:d[2], a in 1:d[1]
                x, y = T(a - 1), T(b - 1)
                u[a, b, c, 1] = -u0 * cos(k * x) * sin(k * y)
                u[a, b, c, 2] = u0 * sin(k * x) * cos(k * y)
                u[a, b, c, 3] = 0.0
            end
            ω = vorticity(u)
            worst = 0.0
            for c in 1:d[3], b in 1:d[2], a in 1:d[1]
                x, y = T(a - 1), T(b - 1)
                exact = 2 * u0 * k * cos(k * x) * cos(k * y)
                worst = max(worst, abs(ω[a, b, c, 3] - exact))
            end
            push!(errs, worst / (2 * u0 * k))
        end
        orders = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
        @info "vorticity convergence" errs orders
        @test all(o -> 1.9 < o < 2.1, orders)
        @test errs[end] < 1e-2
    end

    @testset "the body is a hole, not a region of nonsense" begin
        ϕ = sphere_sdf_field(T, dims, 5.0)
        solid = solid_mask(ϕ)
        s = LBMState{T}(dims..., 0.8; lattice = D3Q27())
        init_equilibrium!(s, (i, j, k) -> (1.0, 0.02, 0.0, 0.0))
        g = to_cube_order!(similar(s.f), s.f)

        u = velocity_field(g; mask = solid)
        @test all(isnan, u[13, 13, 13, :])          # the centre is inside
        @test count(isnan, view(u, :, :, :, 1)) == count(solid)
        @test !any(isnan, u[1, 1, 1, :])

        # NaN spreads one node into the differences, which is the honest answer:
        # a central difference across a wall is not a gradient.
        ω = vorticity(u)
        @test count(isnan, view(ω, :, :, :, 3)) > count(solid)

        ρ = density_field(g; mask = solid)
        @test all(isfinite, [ρ[i, j, k] for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
                             if !solid[i, j, k]])
        @test ρ[1, 1, 1] ≈ 1.0 rtol = 1e-13
    end

    @testset "colour limits ignore the tail and slices are slices" begin
        a = fill(0.1, 10, 10, 10)
        a[5, 5, 5] = 1000.0                          # one wall-adjacent outlier
        lo, hi = field_limits(a; quantile = 0.99)
        @test hi ≈ 0.1 rtol = 1e-12                  # the outlier does not set the scale
        @test lo == -hi
        @test field_limits([NaN, NaN]) == (-1.0, 1.0)

        b = reshape(collect(1.0:27.0), 3, 3, 3)
        @test slice_field(b, 1, 2) == b[2, :, :]
        @test slice_field(b, 3, 3) == b[:, :, 3]
        @test_throws ArgumentError slice_field(b, 4, 1)
    end
end
