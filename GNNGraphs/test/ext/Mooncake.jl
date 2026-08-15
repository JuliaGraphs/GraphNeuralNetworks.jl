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
            @test gm≈gradient(loss, x)[1] rtol=1e-4
            @test gm≈ngradient(loss, x)[1] rtol=1e-4
        end

        # Gradients through `edge_index` on adjacency-matrix graphs.
        for graph_type in (:dense, :sparse)
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            loss(x) = sum(abs2, NNlib.gather(x, edge_index(g)[1]))
            gm = mooncake_gradient(loss, x)
            @test gm≈gradient(loss, x)[1] rtol=1e-4
            @test gm≈ngradient(loss, x)[1] rtol=1e-4
        end

        # Float adjacency: edge weights stay on the AD path.
        A = Float32[0 1 0 2; 1 0 3 0; 0 3 0 1; 2 0 1 0]
        loss_w(A) = sum(abs2, get_edge_weight(GNNGraph(A, graph_type = :dense)))
        gw = mooncake_gradient(loss_w, copy(A))
        @test gw≈2 .* A rtol=1e-4
        @test gw≈gradient(loss_w, A)[1] rtol=1e-4
    end
end

@testitem "GNNGraphsMooncakeExt on GPU" setup=[GraphsTestModule] tags=[:gpu] begin
    using .GraphsTestModule
    import Mooncake
    import NNlib

    dev = gpu_device(force = true)
    # Mooncake's GPU rules are CUDA-only.
    if VERSION >= v"1.12" && dev isa CUDADevice
        function mooncake_gradient(f, x)
            cache = Mooncake.prepare_gradient_cache(f, x)
            _, grads = Mooncake.value_and_gradient!!(cache, f, x)
            return grads[2]
        end

        rng = MersenneTwister(17)

        # Without the rules Mooncake errors here instead of returning a gradient.
        for graph_type in (:coo, :dense)
            g = rand_graph(rng, 6, 10; graph_type)
            x = randn(rng, Float32, 3, g.num_nodes)
            g_gpu, x_gpu = dev(g), dev(x)
            loss(x) = sum(abs2, NNlib.gather(x, edge_index(add_self_loops(g))[1]))
            loss_gpu(x) = sum(abs2, NNlib.gather(x, edge_index(add_self_loops(g_gpu))[1]))
            @test Array(mooncake_gradient(loss_gpu, x_gpu))≈mooncake_gradient(loss, x) rtol=1e-4
        end

        # Float adjacency: edge weights stay on the AD path.
        A = Float32[0 1 0 2; 1 0 3 0; 0 3 0 1; 2 0 1 0]
        loss_w(A) = sum(abs2, get_edge_weight(GNNGraph(A, graph_type = :dense)))
        @test Array(mooncake_gradient(loss_w, dev(A)))≈2 .* A rtol=1e-4
    end
end
