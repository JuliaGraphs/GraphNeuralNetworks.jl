@testitem "Pooling" setup=[TestModuleLux] begin
    using .TestModuleLux
    @testset "Pooling" begin

        rng = StableRNG(1234)
        @testset "GlobalPool" begin
            g = rand_graph(rng, 10, 40)
            in_dims = 3
            x = randn(rng, Float32, in_dims, 10)
            l = GlobalPool(mean)
            test_lux_layer(rng, l, g, x, sizey=(in_dims,1))
        end
        @testset "GlobalAttentionPool" begin
            n = 10
            chin = 6
            chout = 5
            ng = 3
            g = batch([GNNGraph(rand_graph(rng, 10, 40),
                            ndata = rand(Float32, chin, n)) for i in 1:ng])

            fgate = Dense(chin, 1)
            ffeat = Dense(chin, chout)
            l = GlobalAttentionPool(fgate, ffeat)
    
            test_lux_layer(rng, l, g, g.ndata.x, sizey=(chout,ng), container=true)
        end

        @testset "TopKPool" begin
            N = 10
            k, in_channel = 4, 7
            X = rand(in_channel, N)
            ps = (;)
            st = (;)
            for T in [Bool, Float64]
                adj = rand(T, N, N)
                p = GNNLux.TopKPool(adj, k, in_channel)
                @test eltype(p.p) === Float32
                @test size(p.p) == (in_channel,)
                @test eltype(p.Ã) === T
                @test size(p.Ã) == (k, k)
                y, st = p(X, ps, st)
                @test size(y) == (in_channel, k)
            end
        end
    end
end
