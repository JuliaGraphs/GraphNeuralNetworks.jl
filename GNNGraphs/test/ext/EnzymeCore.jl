@testitem "_to_coo_graph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    import NNlib

    rng = MersenneTwister(17)
    for graph_type in GRAPH_TYPES
        g = rand_graph(rng, 6, 10; graph_type)
        gcoo = GNNGraphs._to_coo_graph(g)
        gref = GNNGraph(g, graph_type = :coo)
        @test get_graph_type(gcoo) == :coo
        @test gcoo.num_nodes == gref.num_nodes == g.num_nodes
        @test gcoo.num_edges == gref.num_edges == g.num_edges
        @test edge_index(gcoo) == edge_index(gref)
        @test get_edge_weight(gcoo) == get_edge_weight(gref)
        @test is_coalesced(gcoo) == false
        @test gcoo.ndata === g.ndata
        @test gcoo.edata === g.edata
        @test gcoo.gdata === g.gdata
    end

    # Zygote keeps differentiating through the helper (no ChainRules rule on it).
    g = rand_graph(rng, 6, 10, graph_type = :dense)
    x = randn(rng, Float32, 3, g.num_nodes)
    loss(x, g) = sum(abs2, NNlib.gather(x, edge_index(GNNGraphs._to_coo_graph(g))[1]))
    gz = gradient(x -> loss(x, g), x)[1]
    gfd = ngradient(x -> loss(x, g), x)[1]
    @test gz ≈ gfd rtol = 1e-4
end

@testitem "GNNGraphsEnzymeCoreExt marks _to_coo_graph inactive" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using EnzymeCore: EnzymeRules
    using Enzyme: Enzyme, Const
    import NNlib

    # Loading EnzymeCore triggers the extension, which registers the rule.
    @test hasmethod(EnzymeRules.inactive,
                    Tuple{typeof(GNNGraphs._to_coo_graph), GNNGraph})

    # On Julia 1.10 Enzyme itself fails in its activity analysis for this pattern
    # (MethodError: active_reg(::TypeVar, ::UInt64)), so differentiate only on >= 1.12.
    if VERSION >= v"1.12"
        rng = MersenneTwister(17)
        loss(x, g) = sum(abs2, NNlib.gather(x, edge_index(GNNGraphs._to_coo_graph(g))[1]))
        # Differentiating through the conversion crashed Enzyme before the inactive rule.
        for graph_type in (:dense, :sparse)
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse),
                                   loss, x, Const(g))
            gfd = ngradient(x -> loss(x, g), x)[1]
            @test grad[1] ≈ gfd rtol = 1e-4
        end
    end
end
