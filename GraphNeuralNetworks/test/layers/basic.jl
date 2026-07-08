@testitem "GNNChain" setup=[TestModule] begin
    using .TestModule
    @testset "GNNChain $GRAPH_T" for GRAPH_T in GRAPH_TYPES
        n, din, d, dout = 10, 3, 4, 2
        deg = 4

        g = GNNGraph(random_regular_graph(n, deg),
                        graph_type = GRAPH_T,
                        ndata = randn(Float32, din, n))
        x = g.ndata.x

        gnn = GNNChain(GCNConv(din => d),
                        LayerNorm(d),
                        x -> tanh.(x),
                        GraphConv(d => d, tanh),
                        Dropout(0.5),
                        Dense(d, dout))

        Flux.testmode!(gnn)

        test_gradients(gnn, g, x, rtol = 1e-5)

        @testset "constructor with names" begin
            m = GNNChain(GCNConv(din => d),
                            LayerNorm(d),
                            x -> tanh.(x),
                            Dense(d, dout))

            m2 = GNNChain(enc = m,
                            dec = DotDecoder())

            @test m2[:enc] === m
            @test m2(g, x) == m2[:dec](g, m2[:enc](g, x))
        end

        @testset "constructor with vector" begin
            m = GNNChain(GCNConv(din => d),
                            LayerNorm(d),
                            x -> tanh.(x),
                            Dense(d, dout))
            m2 = GNNChain([m.layers...])
            @test m2(g, x) == m(g, x)
        end

        @testset "Parallel" begin
            AddResidual(l) = Parallel(+, identity, l)

            gnn = GNNChain(GraphConv(din => d, tanh),
                            LayerNorm(d),
                            AddResidual(GraphConv(d => d, tanh)),
                            BatchNorm(d),
                            Dense(d, dout))

            Flux.trainmode!(gnn)

            test_gradients(gnn, g, x, rtol = 1e-4, atol=1e-4)
        end
    end

    @testset "Only graph input" begin
        nin, nout = 2, 4
        ndata = rand(Float32, nin, 3)
        edata = rand(Float32, nin, 3)
        g = GNNGraph([1, 1, 2], [2, 3, 3], ndata = ndata, edata = edata)
        m = NNConv(nin => nout, Dense(2, nin * nout, tanh))
        chain = GNNChain(m)
        y = m(g, g.ndata.x, g.edata.e)
        @test m(g).ndata.x == y
        @test chain(g).ndata.x == y
    end
end

@testitem "WithGraph" setup=[TestModule] begin
    using .TestModule
    x = rand(Float32, 2, 3)
    g = GNNGraph([1, 2, 3], [2, 3, 1], ndata = x)
    model = SAGEConv(2 => 3)
    wg = WithGraph(model, g)
    # No need to feed the graph to `wg`
    @test wg(x) == model(g, x)
    @test Flux.trainables(wg) == Flux.trainables(model)
    g2 = GNNGraph([1, 1, 2, 3], [2, 4, 1, 1])
    x2 = rand(Float32, 2, 4)
    # WithGraph will ignore the internal graph if fed with a new one. 
    @test wg(g2, x2) == model(g2, x2)

    wg = WithGraph(model, g, traingraph = false)
    @test length(Flux.trainables(wg)) == length(Flux.trainables(model))

    wg = WithGraph(model, g, traingraph = true)
    @test length(Flux.trainables(wg)) == length(Flux.trainables(model)) + length(Flux.trainables(g))
end

@testitem "Flux.restructure" setup=[TestModule] begin
    using .TestModule
    chain = GNNChain(GraphConv(2 => 2))
    params, restructure = Flux.destructure(chain)
    @test restructure(params) isa GNNChain
end
