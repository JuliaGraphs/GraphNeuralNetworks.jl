@testitem "Pooling" setup=[TestModuleLux] begin
    using .TestModuleLux
    @testset "GlobalPool" begin

        rng = StableRNG(1234)
        g = rand_graph(rng, 10, 40)
        in_dims = 3
        x = randn(rng, Float32, in_dims, 10)

        @testset "GCNConv" begin
            l = GlobalPool(mean)
            test_lux_layer(rng, l, g, x, sizey=(in_dims,1))
        end
    end
end
