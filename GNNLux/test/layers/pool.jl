@testitem "Pooling" setup=[TestModuleLux] begin
    using .TestModuleLux
    @testset "Pooling" begin

        rng = StableRNG(1234)
        g = rand_graph(rng, 10, 40)
        in_dims = 3
        x = randn(rng, Float32, in_dims, 10)

        @testset "GlobalPool" begin
            l = GlobalPool(mean)
            test_lux_layer(rng, l, g, x, sizey=(in_dims,1))
        end
        @testset "GlobalAttentionPool" begin
            fgate = Dense(in_dims, 1)
            ffeat = Dense(in_dims, in_dims)
            l = GlobalAttentionPool(fgate, ffeat)
            test_lux_layer(rng, l, g, x, sizey=(in_dims,1), container=true)
        end
    end
end
