@testitem "HeteroGraphConv" setup=[TestModule] begin
    using .TestModule
    d, n = 3, 5
    g = rand_bipartite_heterograph((n, 2*n), 15)
    hg = rand_bipartite_heterograph((2,3), 6)

    model = HeteroGraphConv([(:A,:to,:B) => GraphConv(d => d), 
                            (:B,:to,:A) => GraphConv(d => d)])

    for x in [
                (A = rand(Float32, d, n), B = rand(Float32, d, 2n)),
                Dict(:A => rand(Float32, d, n), :B => rand(Float32, d, 2n)) 
             ]
        # x = (A = rand(Float32, d, n), B = rand(Float32, d, 2n))
        x = Dict(:A => rand(Float32, d, n), :B => rand(Float32, d, 2n)) 
      
        y = model(g, x)

        grad, dx = gradient((model, x) -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model, x)
        yhat, (ngrad, ndx) = finitediff_withgradient((model, x) -> sum(model(g, x)[1]) + sum(model(g, x)[2].^2), model, x)

        @test grad.layers[1].weight1 ≈ ngrad.layers[1].weight1  rtol=1e-3
        @test grad.layers[1].weight2 ≈ ngrad.layers[1].weight2  rtol=1e-3
        @test grad.layers[1].bias ≈ ngrad.layers[1].bias        rtol=1e-3
        @test grad.layers[2].weight1 ≈ ngrad.layers[2].weight1  rtol=1e-3
        @test grad.layers[2].weight2 ≈ ngrad.layers[2].weight2  rtol=1e-3
        @test grad.layers[2].bias ≈ ngrad.layers[2].bias        rtol=1e-3

        @test dx[:A] ≈ ndx[:A]        rtol=1e-3
        @test dx[:B] ≈ ndx[:B]        rtol=1e-3
    end

    @testset "Constructor from pairs" begin
        layer = HeteroGraphConv((:A, :to, :B) => GraphConv(64 => 32, tanh),
                                (:B, :to, :A) => GraphConv(64 => 32, tanh));
        @test length(layer.etypes) == 2
    end

    @testset "Destination node aggregation" begin
        # deterministic setup to validate the aggregation
        d, n = 3, 5
        g = GNNHeteroGraph(((:A, :to, :B) => ([1, 1, 2, 3], [1, 2, 2, 3]),
                (:B, :to, :A) => ([1, 1, 2, 3], [1, 2, 2, 3]),
                (:C, :to, :A) => ([1, 1, 2, 3], [1, 2, 2, 3])); num_nodes = Dict(:A => n, :B => n, :C => n))
        model = HeteroGraphConv([
                (:A, :to, :B) => GraphConv(d => d, init = ones, bias = false),
                (:B, :to, :A) => GraphConv(d => d, init = ones, bias = false),
                (:C, :to, :A) => GraphConv(d => d, init = ones, bias = false)]; aggr = +)
        x = (A = rand(Float32, d, n), B = rand(Float32, d, n), C = rand(Float32, d, n))
        y = model(g, x)
        weights = ones(Float32, d, d)

        ### Test default summation aggregation
        # B2 has 2 edges from A and itself (sense check)
        expected = sum(weights * x.A[:, [1, 2]]; dims = 2) .+ weights * x.B[:, [2]]
        output = y.B[:, [2]]
        @test expected ≈ output

        # B5 has only itself
        @test weights * x.B[:, [5]] ≈ y.B[:, [5]]

        # A1 has 1 edge from B, 1 from C and twice itself
        expected = sum(weights * x.B[:, [1]] + weights * x.C[:, [1]]; dims = 2) .+
                   2 * weights * x.A[:, [1]]
        output = y.A[:, [1]]
        @test expected ≈ output

        # A2 has 2 edges from B, 2 from C and twice itself
        expected = sum(weights * x.B[:, [1, 2]] + weights * x.C[:, [1, 2]]; dims = 2) .+
                   2 * weights * x.A[:, [2]]
        output = y.A[:, [2]]
        @test expected ≈ output

        # A5 has only itself but twice
        @test 2 * weights * x.A[:, [5]] ≈ y.A[:, [5]]

        #### Test different aggregation function
        model2 = HeteroGraphConv([
                (:A, :to, :B) => GraphConv(d => d, init = ones, bias = false),
                (:B, :to, :A) => GraphConv(d => d, init = ones, bias = false),
                (:C, :to, :A) => GraphConv(d => d, init = ones, bias = false)]; aggr = -)
        y2 = model2(g, x)
        # B no change
        @test y.B ≈ y2.B

        # A1 has 1 edge from B, 1 from C, itself cancels out
        expected = sum(weights * x.B[:, [1]] - weights * x.C[:, [1]]; dims = 2)
        output = y2.A[:, [1]]
        @test expected ≈ output

        # A2 has 2 edges from B, 2 from C, itself cancels out
        expected = sum(weights * x.B[:, [1, 2]] - weights * x.C[:, [1, 2]]; dims = 2)
        output = y2.A[:, [2]]
        @test expected ≈ output
    end

    @testset "CGConv" begin
        x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv( (:A, :to, :B) => CGConv(4 => 2, tanh),
                                    (:B, :to, :A) => CGConv(4 => 2, tanh));
        y = layers(hg, x); 
        @test size(y.A) == (2,2) && size(y.B) == (2,3)
    end

    @testset "EdgeConv" begin
        x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv( (:A, :to, :B) => EdgeConv(Dense(2 * 4, 2), aggr = +),
                                    (:B, :to, :A) => EdgeConv(Dense(2 * 4, 2), aggr = +));
        y = layers(hg, x); 
        @test size(y.A) == (2,2) && size(y.B) == (2,3)
    end
  
    @testset "SAGEConv" begin
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => SAGEConv(4 => 2, tanh, bias = false, aggr = +),
                                 (:B, :to, :A) => SAGEConv(4 => 2, tanh, bias = false, aggr = +));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "GATConv" begin
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => GATConv(4 => 2),
                                 (:B, :to, :A) => GATConv(4 => 2));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "GINConv" begin
        x = (A = rand(4, 2), B = rand(4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => GINConv(Dense(4, 2), 0.4),
                                    (:B, :to, :A) => GINConv(Dense(4, 2), 0.4));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "ResGatedGraphConv" begin   
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => ResGatedGraphConv(4 => 2),
                                 (:B, :to, :A) => ResGatedGraphConv(4 => 2));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end

    @testset "GATv2Conv" begin
        x = (A = rand(Float32, 4, 2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv((:A, :to, :B) => GATv2Conv(4 => 2),
                                 (:B, :to, :A) => GATv2Conv(4 => 2));
        y = layers(hg, x); 
        @test size(y.A) == (2, 2) && size(y.B) == (2, 3)
    end
    
    @testset "GCNConv" begin
        g = rand_bipartite_heterograph((2,3), 6)
        x = (A = rand(Float32, 4,2), B = rand(Float32, 4, 3))
        layers = HeteroGraphConv( (:A, :to, :B) => GCNConv(4 => 2, tanh),
                                    (:B, :to, :A) => GCNConv(4 => 2, tanh));
        y = layers(g, x);
        @test size(y.A) == (2,2) && size(y.B) == (2,3)
    end

    @testset "NNConv" begin
        nA, nB = 2, 3
        nedges = 6
        g = rand_bipartite_heterograph((nA, nB), nedges;
            edata = Dict((:A, :to, :B) => (e = rand(Float32, 5, nedges),),
                        (:B, :to, :A) => (e = rand(Float32, 5, nedges),))
        )
        x = (A = rand(Float32, 4, nA), B = rand(Float32, 4, nB))
        nn = Dense(5 => 4 * 2)
        layers = HeteroGraphConv((:A, :to, :B) => NNConv(4 => 2, nn, tanh),
                                 (:B, :to, :A) => NNConv(4 => 2, nn, tanh))
        y = layers(g, x)
        @test size(y.A) == (2, nA) && size(y.B) == (2, nB)
    end
end
