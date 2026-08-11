@testitem "GNNGraphsMooncakeExt zero-derivative rules" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    import Mooncake
    import NNlib

    # Loading Mooncake triggers the extension.
    @test Base.get_extension(GNNGraphs, :GNNGraphsMooncakeExt) !== nothing

    # Mooncake is only exercised on Julia >= 1.12, as in the other test suites.
    if VERSION >= v"1.12"
        function mooncake_gradient(f, x)
            cache = Mooncake.prepare_gradient_cache(f, x)
            _, grads = Mooncake.value_and_gradient!!(cache, f, x)
            return grads[2]
        end

        rng = MersenneTwister(17)

        # Gradients through `add_self_loops` (zero-derivative rule).
        for graph_type in GRAPH_TYPES
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            loss(x) = sum(abs2, NNlib.gather(x, edge_index(add_self_loops(g))[1]))
            gm = mooncake_gradient(loss, x)
            gz = gradient(loss, x)[1]
            gfd = ngradient(loss, x)[1]
            @test gm≈gz rtol=1e-4
            @test gm≈gfd rtol=1e-4
        end

        # Gradients through `edge_index` on adjacency-matrix graphs
        # (`to_coo`, traced by Mooncake with only `_findnz_idx` marked).
        for graph_type in (:dense, :sparse)
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            loss(x) = sum(abs2, NNlib.gather(x, edge_index(g)[1]))
            gm = mooncake_gradient(loss, x)
            gfd = ngradient(loss, x)[1]
            @test gm≈gfd rtol=1e-4
        end

        # Gradients through `_to_coo_graph`, used by the GCNConv/SGConv/TAGConv
        # adjacency-matrix fallbacks in GNNlib.
        for graph_type in (:dense, :sparse)
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            loss(x) = sum(abs2,
                NNlib.gather(x, edge_index(GNNGraphs._to_coo_graph(g))[1]))
            gm = mooncake_gradient(loss, x)
            gfd = ngradient(loss, x)[1]
            @test gm≈gfd rtol=1e-4
        end

        # Float adjacency: edge weights stay differentiable through `_edge_values`.
        adjf = Float32[0 1 0 2; 1 0 3 0; 0 3 0 1; 2 0 1 0]
        gf = GNNGraph(adjf, graph_type = :dense)
        xf = randn(rng, Float32, 3, 4)
        loss_f(x) = sum(abs2, NNlib.gather(x, edge_index(gf)[1]) .* get_edge_weight(gf)')
        gm = mooncake_gradient(loss_f, xf)
        gz = gradient(loss_f, xf)[1]
        @test gm≈gz rtol=1e-4
    end
end
