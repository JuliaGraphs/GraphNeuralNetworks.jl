@testsnippet TolSnippet begin
    RTOL_LOW = 1e-2
    RTOL_HIGH = 1e-5
    ATOL_LOW = 1e-3
end

@testitem "GCNConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    @testset "basic" begin
        l = GCNConv(D_IN => D_OUT)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end

        l = GCNConv(D_IN => D_OUT, tanh, bias = false)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end

        l = GCNConv(D_IN => D_OUT, add_self_loops = false)
        for g in TEST_GRAPHS
            has_isolated_nodes(g) && continue
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end
    end

    @testset "edge weights & custom normalization $GRAPH_T" for GRAPH_T in GRAPH_TYPES
        s = [2, 3, 1, 3, 1, 2]
        t = [1, 1, 2, 2, 3, 3]
        w = Float32[1, 2, 3, 4, 5, 6]
        g = GNNGraph((s, t, w), ndata = ones(Float32, 1, 3), graph_type = GRAPH_T)
        x = g.ndata.x
        custom_norm_fn(d) = 1 ./ sqrt.(d)  
        l = GCNConv(1 => 1, add_self_loops = false, use_edge_weight = true)
        l.weight .= 1
        d = degree(g, dir = :in, edge_weight = true)
        y = l(g, x)
        @test y[1, 1] ≈ w[1] / √(d[1] * d[2]) + w[2] / √(d[1] * d[3])
        @test y[1, 2] ≈ w[3] / √(d[2] * d[1]) + w[4] / √(d[2] * d[3])
        @test y ≈ l(g, x, w; norm_fn = custom_norm_fn) # checking without custom

        # test gradient with respect to edge weights
        w = rand(Float32, 6)
        x = rand(Float32, 1, 3)
        g = GNNGraph((s, t, w), ndata = x, graph_type = GRAPH_T, edata = w)
        l = GCNConv(1 => 1, add_self_loops = false, use_edge_weight = true)
        @test gradient(w -> sum(l(g, x, w)), w)[1] isa AbstractVector{Float32}   # redundant test but more explicit
        @test size(l(g, x, w)) == (1, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end

    @testset "conv_weight" begin
        l = GraphNeuralNetworks.GCNConv(D_IN => D_OUT)
        w = zeros(Float32, D_OUT, D_IN)
        
        for g in TEST_GRAPHS
            x = ones(Float32, D_IN, g.num_nodes)
            @test l(g, x, conv_weight = w) == zeros(Float32, D_OUT, g.num_nodes)
            x = rand(Float32, D_IN, g.num_nodes)
            @test l(g, x, conv_weight = w) == w * x
        end
    end
end


@testitem "GCNConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = GCNConv(D_IN => D_OUT)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "ChebConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    k = 2
    l = ChebConv(D_IN => D_OUT, k)
    @test size(l.weight) == (D_OUT, D_IN, k)
    @test size(l.bias) == (D_OUT,)
    @test l.k == k
    for g in TEST_GRAPHS
        g = add_self_loops(g)
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_LOW)
    end

    @testset "bias=false" begin
        @test length(Flux.trainables(ChebConv(2 => 3, 3))) == 2
        @test length(Flux.trainables(ChebConv(2 => 3, 3, bias = false))) == 1
    end
end


@testitem "ChebConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    k = 2
    l = ChebConv(D_IN => D_OUT, k)
    for g in TEST_GRAPHS
        has_isolated_nodes(g) && continue
        gpu_backend() == "AMDGPU" && continue # TODO skipping since julia crashes
        broken = gpu_backend() == "AMDGPU"
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes) broken=broken
        
        if gpu_backend() == "AMDGPU"
            broken = true
        elseif gpu_backend() == "CUDA" && get_graph_type(g) in [:coo, :sparse]
            broken = true
        else
            broken = false
        end
        @test test_gradients(
            l, g, g.x, rtol = RTOL_LOW, test_gpu = true, compare_finite_diff = false
        ) broken=broken
    end   
end

@testitem "GraphConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = GraphConv(D_IN => D_OUT)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end

    l = GraphConv(D_IN => D_OUT, tanh, bias = false, aggr = mean)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end

    @testset "bias=false" begin
        @test length(Flux.trainables(GraphConv(2 => 3))) == 3
        @test length(Flux.trainables(GraphConv(2 => 3, bias = false))) == 2
    end
end


@testitem "GraphConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = GraphConv(D_IN => D_OUT)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end


@testitem "GATConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    for heads in (1, 2), concat in (true, false)
        l = GATConv(D_IN => D_OUT; heads, concat, dropout=0)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (concat ? heads * D_OUT : D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_LOW)
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATConv((D_IN, ein) => D_OUT, add_self_loops = false, dropout=0)
        g = GNNGraph(TEST_GRAPHS[1], edata = rand(Float32, ein, TEST_GRAPHS[1].num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW)
    end

    @testset "num params" begin
        l = GATConv(2 => 3, add_self_loops = false)
        @test length(Flux.trainables(l)) == 3
        l = GATConv((2, 4) => 3, add_self_loops = false)
        @test length(Flux.trainables(l)) == 4
        l = GATConv((2, 4) => 3, add_self_loops = false, bias = false)
        @test length(Flux.trainables(l)) == 3
    end
end

@testitem "GATConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    for heads in (1, 2), concat in (true, false)
        l = GATConv(D_IN => D_OUT; heads, concat, dropout=0)
        for g in TEST_GRAPHS
            g.graph isa AbstractSparseMatrix && continue
            @test size(l(g, g.x)) == (concat ? heads * D_OUT : D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_LOW, test_gpu = true, compare_finite_diff = false)
        end
    end
end

@testitem "GATv2Conv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    for heads in (1, 2), concat in (true, false)
        l = GATv2Conv(D_IN => D_OUT, tanh; heads, concat, dropout=0)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (concat ? heads * D_OUT : D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_LOW, atol=ATOL_LOW)
        end
    end

    @testset "edge features" begin
        ein = 3
        l = GATv2Conv((D_IN, ein) => D_OUT, add_self_loops = false, dropout=0)
        g = GNNGraph(TEST_GRAPHS[1], edata = rand(Float32, ein, TEST_GRAPHS[1].num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW, atol=ATOL_LOW)
    end

    @testset "num params" begin
        l = GATv2Conv(2 => 3, add_self_loops = false)
        @test length(Flux.trainables(l)) == 5
        l = GATv2Conv((2, 4) => 3, add_self_loops = false)
        @test length(Flux.trainables(l)) == 6
        l = GATv2Conv((2, 4) => 3, add_self_loops = false, bias = false)
        @test length(Flux.trainables(l)) == 4
    end
end

@testitem "GATv2Conv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    for heads in (1, 2), concat in (true, false)
        l = GATv2Conv(D_IN => D_OUT, tanh; heads, concat, dropout=0)
        for g in TEST_GRAPHS
            g.graph isa AbstractSparseMatrix && continue
            @test size(l(g, g.x)) == (concat ? heads * D_OUT : D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_LOW, atol=ATOL_LOW, test_gpu = true, compare_finite_diff = false)
        end
    end
end

@testitem "GatedGraphConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    num_layers = 3
    l = GatedGraphConv(D_OUT, num_layers)
    @test size(l.weight) == (D_OUT, D_OUT, num_layers)

    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end
end


@testitem "GatedGraphConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    num_layers = 3
    l = GatedGraphConv(D_OUT, num_layers)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "EdgeConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = EdgeConv(Dense(2 * D_IN, D_OUT), aggr = +)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end
end

@testitem "EdgeConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = EdgeConv(Dense(2 * D_IN, D_OUT), aggr = +)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "GINConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    nn = Dense(D_IN, D_OUT)

    l = GINConv(nn, 0.01, aggr = mean)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end

    @test !in(:eps, Flux.trainable(l))
end

@testitem "GINConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    nn = Dense(D_IN, D_OUT)
    l = GINConv(nn, 0.01, aggr = mean)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "NNConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    edim = 10
    nn = Dense(edim, D_OUT * D_IN)

    l = NNConv(D_IN => D_OUT, nn, tanh, bias = true, aggr = +)
    for g in TEST_GRAPHS
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH)
    end
end

@testitem "NNConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    edim = 10
    nn = Dense(edim, D_OUT * D_IN)
    l = NNConv(D_IN => D_OUT, nn, tanh, bias = true, aggr = +)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "SAGEConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = SAGEConv(D_IN => D_OUT)
    @test l.aggr == mean

    l = SAGEConv(D_IN => D_OUT, tanh, bias = false, aggr = +)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end
end

@testitem "SAGEConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = SAGEConv(D_IN => D_OUT)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "ResGatedGraphConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = ResGatedGraphConv(D_IN => D_OUT, tanh, bias = true)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end
end

@testitem "ResGatedGraphConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = ResGatedGraphConv(D_IN => D_OUT, tanh, bias = true)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "CGConv" setup=[TolSnippet, TestModule] begin
    using .TestModule

    edim = 10
    l = CGConv((D_IN, edim) => D_OUT, tanh, residual = false, bias = true)
    for g in TEST_GRAPHS
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH)
    end

    # no edge features
    l1 = CGConv(D_IN => D_OUT, tanh, residual = false, bias = true)
    g1 = TEST_GRAPHS[1]
    @test l1(g1, g1.ndata.x) == l1(g1).ndata.x
    @test l1(g1, g1.ndata.x, nothing) == l1(g1).ndata.x
end

@testitem "CGConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    edim = 10
    l = CGConv((D_IN, edim) => D_OUT, tanh, residual = false, bias = true)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        g = GNNGraph(g, edata = rand(Float32, edim, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "AGNNConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = AGNNConv(trainable=false, add_self_loops=false)
    @test l.β == [1.0f0]
    @test l.add_self_loops == false
    @test l.trainable == false
    Flux.trainable(l) == (;)

    l = AGNNConv(init_beta=2.0f0)
    @test l.β == [2.0f0]
    @test l.add_self_loops == true
    @test l.trainable == true 
    Flux.trainable(l) == (; β = [1f0])
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_IN, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH)
    end
end

@testitem "AGNNConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = AGNNConv(trainable=false, add_self_loops=false)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_IN, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "MEGNetConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    l = MEGNetConv(D_IN => D_OUT, aggr = +)
    for g in TEST_GRAPHS
        g = GNNGraph(g, edata = rand(Float32, D_IN, g.num_edges))
        y = l(g, g.x, g.e)
        @test size(y[1]) == (D_OUT, g.num_nodes)
        @test size(y[2]) == (D_OUT, g.num_edges)
        function loss(l, g, x, e)
            y = l(g, x, e)
            return mean(y[1]) + sum(y[2])
        end
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW; loss)
    end
end

@testitem "MEGNetConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = MEGNetConv(D_IN => D_OUT, aggr = +)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        g = GNNGraph(g, edata = rand(Float32, D_IN, g.num_edges))
        y = l(g, g.x, g.e)
        @test size(y[1]) == (D_OUT, g.num_nodes)
        @test size(y[2]) == (D_OUT, g.num_edges)
        function loss(l, g, x, e)
            y = l(g, x, e)
            return mean(y[1]) + sum(y[2])
        end
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW; loss, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "GMMConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    ein_channel = 10
    K = 5
    l = GMMConv((D_IN, ein_channel) => D_OUT, K = K)
    for g in TEST_GRAPHS
        g = GNNGraph(g, edata = rand(Float32, ein_channel, g.num_edges))
        y = l(g, g.x, g.e)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH)
    end
end

@testitem "GMMConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    ein_channel = 10
    K = 5
    l = GMMConv((D_IN, ein_channel) => D_OUT, K = K)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        g = GNNGraph(g, edata = rand(Float32, ein_channel, g.num_edges))
        y = l(g, g.x, g.e)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end

@testitem "SGConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = SGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end

        l = SGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end
    end
end

@testitem "SGConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    k = 2
    l = SGConv(D_IN => D_OUT, k, add_self_loops = true)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end
end

@testitem "TAGConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    K = [1, 2, 3]
    for k in K
        l = TAGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end

        l = TAGConv(D_IN => D_OUT, k, add_self_loops = true)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end
    end
end

@testitem "TAGConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    k = 2
    l = TAGConv(D_IN => D_OUT, k, add_self_loops = true)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end
end

@testitem "EGNNConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    #TODO test gradient
    #TODO test gpu
    @testset "EGNNConv $GRAPH_T" for GRAPH_T in GRAPH_TYPES
        hin = 5
        hout = 5
        hidden = 5
        l = EGNNConv(hin => hout, hidden)
        g = rand_graph(10, 20, graph_type = GRAPH_T)
        x = rand(Float32, D_IN, g.num_nodes)
        h = randn(Float32, hin, g.num_nodes)
        hnew, xnew = l(g, h, x)
        @test size(hnew) == (hout, g.num_nodes)
        @test size(xnew) == (D_IN, g.num_nodes)
    end
end

@testitem "TransformerConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    ein = 2
    heads = 3
    # used like in Kool et al., 2019
    l = TransformerConv(D_IN * heads => D_IN; heads, add_self_loops = true,
                        root_weight = false, ff_channels = 10, skip_connection = true,
                        batch_norm = false)
    # batch_norm=false here for tests to pass; true in paper
    for g in TEST_GRAPHS
        g = GNNGraph(g, ndata = rand(Float32, D_IN * heads, g.num_nodes))
        @test size(l(g, g.x)) == (D_IN * heads, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_LOW)
    end
    # used like in Shi et al., 2021 
    l = TransformerConv((D_IN, ein) => D_IN; heads, gating = true,
                        bias_qkv = true)
    for g in TEST_GRAPHS
        g = GNNGraph(g, edata = rand(Float32, ein, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_IN * heads, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW)
    end
    # test averaging heads
    l = TransformerConv(D_IN => D_IN; heads, concat = false,
                        bias_root = false,
                        root_weight = false)
    for g in TEST_GRAPHS
        @test size(l(g, g.x)) == (D_IN, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_LOW)
    end
end

@testitem "TransformerConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    ein = 2
    heads = 3

    # used like in Shi et al., 2021 
    l = TransformerConv((D_IN, ein) => D_IN; heads, gating = true,
                        bias_qkv = true)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        g = GNNGraph(g, edata = rand(Float32, ein, g.num_edges))
        @test size(l(g, g.x, g.e)) == (D_IN * heads, g.num_nodes)
        test_gradients(l, g, g.x, g.e, rtol = RTOL_LOW, test_gpu = true, compare_finite_diff = false)
    end
end


@testitem "DConv" setup=[TolSnippet, TestModule] begin
    using .TestModule
    K = [1, 2, 3] # for different number of hops       
    for k in K
        l = DConv(D_IN => D_OUT, k)
        for g in TEST_GRAPHS
            @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
            test_gradients(l, g, g.x, rtol = RTOL_HIGH)
        end
    end
end

@testitem "DConv GPU" setup=[TolSnippet, TestModule] tags=[:gpu] begin
    using .TestModule
    l = DConv(D_IN => D_OUT, 2)
    for g in TEST_GRAPHS
        g.graph isa AbstractSparseMatrix && continue
        @test size(l(g, g.x)) == (D_OUT, g.num_nodes)
        test_gradients(l, g, g.x, rtol = RTOL_HIGH, test_gpu = true, compare_finite_diff = false)
    end   
end
