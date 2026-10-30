@testitem "layers/basic" setup=[TestModuleLux] begin
    using .TestModuleLux
    
    rng = StableRNG(17)
    g = rand_graph(rng, 10, 40)
    x = randn(rng, Float32, 3, 10)        

    @testset "GNNLayer" begin
        @test GNNLayer <: LuxCore.AbstractLuxLayer
    end

    @testset "GNNContainerLayer" begin
        @test GNNContainerLayer <: LuxCore.AbstractLuxContainerLayer
    end

    @testset "GNNChain" begin
        @test GNNChain <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
        c = GNNChain(GraphConv(3 => 5, tanh), GCNConv(5 => 3))
        test_lux_layer(rng, c, g, x, outputsize=(3,), container=true)

        # The returned state has the same structure as the input state,
        # so it can be fed back into the chain (as in a training loop).
        c = GNNChain(GCNConv(3 => 5, relu), Dropout(0.5), Dense(5 => 3))
        ps, st = LuxCore.setup(rng, c)
        y, st′ = c(g, x, ps, st)
        @test typeof(st′) == typeof(st)
        @test st′.layers.layer_2.rng != st.layers.layer_2.rng
        y′, st″ = c(g, x, ps, st′)
        @test typeof(st″) == typeof(st)
        @test y′ != y
    end
end
