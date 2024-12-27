@testitem "intersect" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 20, graph_type = GRAPH_T)
        @test intersect(g, g).num_edges == 20
    end
end
