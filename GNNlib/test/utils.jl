@testitem "utils" setup=[TestModuleGNNlib] begin
    using .TestModuleGNNlib
    # TODO test all graph types
    GRAPH_T = :coo
    De, Dx = 3, 2
    g = MLUtils.batch([rand_graph(10, 60, bidirected=true,
                                ndata = rand(Dx, 10),
                                edata = rand(De, 30),
                                graph_type = GRAPH_T) for i in 1:5])
    x = g.ndata.x
    e = g.edata.e

    @testset "reduce_nodes" begin
        r = reduce_nodes(mean, g, x)
        @test size(r) == (Dx, g.num_graphs)
        @test r[:, 2] ≈ mean(getgraph(g, 2).ndata.x, dims = 2)

        r2 = reduce_nodes(mean, graph_indicator(g), x)
        @test r2 == r
    end

    @testset "reduce_nodes" begin
        g = rand_bipartite_heterograph((5, 10), 20)
        x = (
            A = [Float32(i) for j = 1:1, i = 1:g.num_nodes[:A]],
            B = [Float32(0) for j = 1:2, _ = 1:g.num_nodes[:B]],
        )
        expected = sum(i for i = 1:g.num_nodes[:A])
        result = reduce_nodes(+, :A, g, x)
        @test result == [expected;;]
    end

    @testset "reduce_edges" begin
        r = reduce_edges(mean, g, e)
        @test size(r) == (De, g.num_graphs)
        @test r[:, 2] ≈ mean(getgraph(g, 2).edata.e, dims = 2)
    end

    @testset "softmax_nodes" begin
        r = softmax_nodes(g, x)
        @test size(r) == size(x)
        @test r[:, 1:10] ≈ softmax(getgraph(g, 1).ndata.x, dims = 2)
    end

    @testset "softmax_edges" begin
        r = softmax_edges(g, e)
        @test size(r) == size(e)
        @test r[:, 1:60] ≈ softmax(getgraph(g, 1).edata.e, dims = 2)
    end

    @testset "broadcast_nodes" begin
        z = rand(4, g.num_graphs)
        r = broadcast_nodes(g, z)
        @test size(r) == (4, g.num_nodes)
        @test r[:, 1] ≈ z[:, 1]
        @test r[:, 10] ≈ z[:, 1]
        @test r[:, 11] ≈ z[:, 2]
    end

    @testset "broadcast_edges" begin
        z = rand(4, g.num_graphs)
        r = broadcast_edges(g, z)
        @test size(r) == (4, g.num_edges)
        @test r[:, 1] ≈ z[:, 1]
        @test r[:, 60] ≈ z[:, 1]
        @test r[:, 61] ≈ z[:, 2]
    end

    @testset "softmax_edge_neighbors" begin
        s = [1, 2, 3, 4]
        t = [5, 5, 6, 6]
        g2 = GNNGraph(s, t)
        e2 = randn(Float32, 3, g2.num_edges)
        z = softmax_edge_neighbors(g2, e2)
        @test size(z) == size(e2)
        @test z[:, 1:2] ≈ NNlib.softmax(e2[:, 1:2], dims = 2)
        @test z[:, 3:4] ≈ NNlib.softmax(e2[:, 3:4], dims = 2)
    end
end

