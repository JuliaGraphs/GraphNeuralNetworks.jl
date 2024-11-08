@testitem "layers/conv" setup=[SharedTestSetup] begin
    rng = StableRNG(1234)
    g = rand_graph(rng, 10, 40)
    in_dims = 3
    out_dims = 5
    x = randn(rng, Float32, in_dims, 10)

    @testset "TransformerConv" begin
        x = randn(rng, Float32, 6, 10)
        ein = 2 
        e = randn(rng, Float32, ein, g.num_edges)
    
        l = TransformerConv((6, ein) => 8, heads = 2, gating = true, bias_qkv = true)
        test_lux_layer(rng, l, g, x, outputsize = (16,), e = e, container = true)
    
        # l = TransformerConv((16, ein) => 16, heads = 2, concat = false, skip_connection = true)
        # test_lux_layer(rng, l, g, x, outputsize = (16,), e = e, container = true)
    end

    @testset "GCNConv" begin
        l = GCNConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "ChebConv" begin
        l = ChebConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "GraphConv" begin
        l = GraphConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "AGNNConv" begin
        l = AGNNConv(init_beta=1.0f0)
        test_lux_layer(rng, l, g, x, sizey=(in_dims, 10))
    end

    @testset "EdgeConv" begin
        nn = Chain(Dense(2*in_dims => 2, tanh), Dense(2 => out_dims))
        l = EdgeConv(nn, aggr = +)
        test_lux_layer(rng, l, g, x, sizey=(out_dims,10), container=true)
    end

    @testset  "CGConv" begin
        l = CGConv(in_dims => in_dims, residual = true)
        test_lux_layer(rng, l, g, x, outputsize=(in_dims,), container=true)
    end

    @testset "DConv" begin
        l = DConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(5,))
    end

    @testset "EGNNConv" begin
        hin = 6
        hout = 7
        hidden = 8
        l = EGNNConv(hin => hout, hidden)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        h = randn(rng, Float32, hin, g.num_nodes)
        (hnew, xnew), stnew = l(g, h, x, ps, st)
        @test size(hnew) == (hout, g.num_nodes)
        @test size(xnew) == (in_dims, g.num_nodes)
    end

    @testset "GATConv" begin
        x = randn(rng, Float32, 6, 10)

        l = GATConv(6 => 8, heads=2)
        test_lux_layer(rng, l, g, x, outputsize=(16,))

        l = GATConv(6 => 8, heads=2, concat=false, dropout=0.5)
        test_lux_layer(rng, l, g, x, outputsize=(8,))

        #TODO test edge
    end

    @testset "GATv2Conv" begin
        x = randn(rng, Float32, 6, 10)

        l = GATv2Conv(6 => 8, heads=2)
        test_lux_layer(rng, l, g, x, outputsize=(16,))

        l = GATv2Conv(6 => 8, heads=2, concat=false, dropout=0.5)
        test_lux_layer(rng, l, g, x, outputsize=(8,))

        #TODO test edge
    end

    @testset "SGConv" begin
        l = SGConv(in_dims => out_dims, 2)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "GatedGraphConv" begin
        l = GatedGraphConv(in_dims, 3)
        test_lux_layer(rng, l, g, x, outputsize=(in_dims,))
    end

    @testset "GINConv" begin
        nn = Chain(Dense(in_dims => out_dims, tanh), Dense(out_dims => out_dims))
        l = GINConv(nn, 0.5)
        test_lux_layer(rng, l, g, x, sizey=(out_dims,g.num_nodes), container=true)
    end

    @testset "MEGNetConv" begin
        l = MEGNetConv(in_dims => out_dims)
    
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
    
        e = randn(rng, Float32, in_dims, g.num_edges) 
        (x_new, e_new), st_new = l(g, x, e, ps, st)   
    
        @test size(x_new) == (out_dims, g.num_nodes)
        @test size(e_new) == (out_dims, g.num_edges)
    end

    @testset "NNConv" begin
        n_in = 3
        n_in_edge = 10
        n_out = 5

        s = [1,1,2,3]
        t = [2,3,1,1]
        g2 = GNNGraph(s, t)

        nn = Dense(n_in_edge => n_out * n_in)
        l = NNConv(n_in => n_out, nn, tanh, aggr = +)
        x = randn(Float32, n_in, g2.num_nodes)
        e = randn(Float32, n_in_edge, g2.num_edges)
        test_lux_layer(rng, l, g2, x; outputsize=(n_out,), e, container=true)
    end

    @testset "GMMConv" begin
        ein_dims = 4 
        e = randn(rng, Float32, ein_dims, g.num_edges)
        l = GMMConv((in_dims, ein_dims) => out_dims, tanh; K = 2, residual = false)
        test_lux_layer(rng, l, g, x; outputsize=(out_dims,), e)
    end

    @testset "ResGatedGraphConv" begin
        l = ResGatedGraphConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end

    @testset "SAGEConv" begin
        l = SAGEConv(in_dims => out_dims, tanh)
        test_lux_layer(rng, l, g, x, outputsize=(out_dims,))
    end
end
