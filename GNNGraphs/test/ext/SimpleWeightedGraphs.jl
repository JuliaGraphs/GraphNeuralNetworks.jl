@testitem "simple_weighted_graph" begin
    using SimpleWeightedGraphs
    srcs = [1, 2, 1]
    dsts = [2, 3, 3]
    wts = [0.5, 0.8, 2.0]
    g = SimpleWeightedGraph(srcs, dsts, wts)
    gd = SimpleWeightedDiGraph(srcs, dsts, wts)
    gnn_g = GNNGraph(g)
    gnn_gd = GNNGraph(gd)
    @test get_edge_weight(gnn_g)  == [0.5, 2, 0.5, 0.8, 2.0, 0.8] 
    @test get_edge_weight(gnn_gd) == [0.5, 2, 0.8]
end
