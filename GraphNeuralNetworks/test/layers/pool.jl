@testitem "GlobalPool" setup=[TestModule] begin
    using .TestModule
    @testset "GlobalPool $GRAPH_T" for GRAPH_T in GRAPH_TYPES
        p = GlobalPool(+)
        n = 10
        chin = 6
        X = rand(Float32, 6, n)
        g = GNNGraph(random_regular_graph(n, 4), ndata = X, graph_type = GRAPH_T)
        u = p(g, X)
        @test u ≈ sum(X, dims = 2)

        ng = 3
        g = Flux.batch([GNNGraph(random_regular_graph(n, 4),
                                    ndata = rand(Float32, chin, n),
                                    graph_type = GRAPH_T)
                        for i in 1:ng])
        u = p(g, g.ndata.x)
        @test size(u) == (chin, ng)
        @test u[:, [1]] ≈ sum(g.ndata.x[:, 1:n], dims = 2)
        @test p(g).gdata.u == u

        test_gradients(p, g, g.x, rtol = 1e-5)
    end
end

@testitem "GlobalAttentionPool" setup=[TestModule] begin
    using .TestModule
    @testset "GlobalAttentionPool $GRAPH_T" for GRAPH_T in GRAPH_TYPES
        n = 10
        chin = 6
        chout = 5
        ng = 3

        fgate = Dense(chin, 1)
        ffeat = Dense(chin, chout)
        p = GlobalAttentionPool(fgate, ffeat)
        @test length(Flux.trainables(p)) == 4

        g = Flux.batch([GNNGraph(random_regular_graph(n, 4),
                                    ndata = rand(Float32, chin, n),
                                    graph_type = GRAPH_T)
                        for i in 1:ng])

        @test size(p(g, g.x)) == (chout, ng)
        test_gradients(p, g, g.x, rtol = 1e-5)
    end
end

@testitem "TopKPool" setup=[TestModule] begin
    using .TestModule
    N = 10
    k, in_channel = 4, 7
    X = rand(in_channel, N)
    for T in [Bool, Float64]
        adj = rand(T, N, N)
        p = TopKPool(adj, k, in_channel)
        @test eltype(p.p) === Float32
        @test size(p.p) == (in_channel,)
        @test eltype(p.Ã) === T
        @test size(p.Ã) == (k, k)
        y = p(X)
        @test size(y) == (in_channel, k)
    end
end


@testitem "topk_index" begin
    X = [8, 7, 6, 5, 4, 3, 2, 1]
    @test topk_index(X, 4) == [1, 2, 3, 4]
    @test topk_index(X', 4) == [1, 2, 3, 4]
end

@testitem "Set2Set" setup=[TestModule] begin
    using .TestModule
    @testset "Set2Set $GRAPH_T" for GRAPH_T in GRAPH_TYPES
            
        n_in = 3
        n_iters = 2
        n_layers = 1 #TODO test with more layers
        g = batch([rand_graph(10, 40, graph_type = GRAPH_T) for _ in 1:5])
        g = GNNGraph(g, ndata = rand(Float32, n_in, g.num_nodes))
        l = Set2Set(n_in, n_iters, n_layers)
        y = l(g, node_features(g))
        @test size(y) == (2 * n_in, g.num_graphs)
        
        ## TODO the numerical gradient seems to be 3 times smaller than zygote one
        # test_gradients(l, g, g.x, rtol = 1e-4, atol=1e-4)
    end
end
