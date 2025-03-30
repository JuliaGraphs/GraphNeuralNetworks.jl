@testmodule TemporalConvTestModule begin
    using GraphNeuralNetworks
    using Statistics
    export in_channel, out_channel, N, timesteps, g, tg, cell_loss,
            RTOL_LOW, ATOL_LOW, RTOL_HIGH

    RTOL_LOW = 1e-2
    ATOL_LOW = 1e-3
    RTOL_HIGH = 1e-5

    in_channel = 3
    out_channel = 5
    N = 4
    timesteps = 5

    cell_loss(cell, g, x...) = mean(cell(g, x...)[1])

    g = GNNGraph(rand_graph(N, 8),
                  ndata = rand(Float32, in_channel, N),
                  graph_type = :coo)

    tg = TemporalSnapshotsGNNGraph([g for _ in 1:timesteps])

end

@testitem "TGCNCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    
    # Test with default activation function
    cell = GraphNeuralNetworks.TGCNCell(in_channel => out_channel)
    y, h = cell(g, g.x)
    @test y === h
    @test size(h) == (out_channel, g.num_nodes)
    # with no initial state
    test_gradients(cell, g, g.x, loss=cell_loss, rtol=RTOL_HIGH)
    # with initial state
    test_gradients(cell, g, g.x, h, loss=cell_loss, rtol=RTOL_HIGH)
    
    # Test with custom activation function
    custom_activation = tanh
    cell_custom = GraphNeuralNetworks.TGCNCell(in_channel => out_channel, act = custom_activation)
    y_custom, h_custom = cell_custom(g, g.x)
    @test y_custom === h_custom
    @test size(h_custom) == (out_channel, g.num_nodes)
    # Test that outputs differ when using different activation functions
    @test !isapprox(y, y_custom, rtol=RTOL_HIGH)
    # with no initial state
    test_gradients(cell_custom, g, g.x, loss=cell_loss, rtol=RTOL_HIGH)
    # with initial state
    test_gradients(cell_custom, g, g.x, h_custom, loss=cell_loss, rtol=RTOL_HIGH)
end

@testitem "TGCN" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    
    # Test with default activation function
    layer = TGCN(in_channel => out_channel)
    x = rand(Float32, in_channel, timesteps, g.num_nodes)
    state0 = rand(Float32, out_channel, g.num_nodes)
    y = layer(g, x)
    @test layer isa GNNRecurrence
    @test size(y) == (out_channel, timesteps, g.num_nodes)
    # with no initial state
    test_gradients(layer, g, x, rtol = RTOL_HIGH)
    # with initial state
    test_gradients(layer, g, x, state0, rtol = RTOL_HIGH)

    # Test with custom activation function
    custom_activation = tanh
    layer_custom = TGCN(in_channel => out_channel, act = custom_activation)
    y_custom = layer_custom(g, x)
    @test layer_custom isa GNNRecurrence
    @test size(y_custom) == (out_channel, timesteps, g.num_nodes)
    # Test that outputs differ when using different activation functions
    @test !isapprox(y, y_custom, rtol = RTOL_HIGH)
    # with no initial state
    test_gradients(layer_custom, g, x, rtol = RTOL_HIGH)
    # with initial state
    test_gradients(layer_custom, g, x, state0, rtol = RTOL_HIGH)

    # interplay with GNNChain
    model = GNNChain(TGCN(in_channel => out_channel), Dense(out_channel, 1))
    y = model(g, x)
    @test size(y) == (1, timesteps, g.num_nodes)
    test_gradients(model, g, x, rtol = RTOL_HIGH, atol = ATOL_LOW)
end

@testitem "GConvLSTMCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    cell = GConvLSTMCell(in_channel => out_channel, 2)
    y, (h, c) = cell(g, g.x)
    @test y === h
    @test size(h) == (out_channel, g.num_nodes)
    @test size(c) == (out_channel, g.num_nodes)
    # with no initial state
    test_gradients(cell, g, g.x, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(cell, g, g.x, (h, c), loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
end

@testitem "GConvLSTM" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    layer = GConvLSTM(in_channel => out_channel, 2)
    @test layer isa GNNRecurrence
    x = rand(Float32, in_channel, timesteps, g.num_nodes)
    state0 = (rand(Float32, out_channel, g.num_nodes), rand(Float32, out_channel, g.num_nodes))
    y = layer(g, x)
    @test size(y) == (out_channel, timesteps, g.num_nodes)
    # with no initial state
    test_gradients(layer, g, x, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(layer, g, x, state0, rtol=RTOL_LOW, atol=ATOL_LOW)

    # interplay with GNNChain
    model = GNNChain(GConvLSTM(in_channel => out_channel, 2), Dense(out_channel, 1))
    y = model(g, x)
    @test size(y) == (1, timesteps, g.num_nodes)
    test_gradients(model, g, x, rtol = RTOL_LOW, atol = ATOL_LOW)
end

@testitem "GConvGRUCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    cell = GConvGRUCell(in_channel => out_channel, 2)
    y, h = cell(g, g.x)
    @test y === h
    @test size(h) == (out_channel, g.num_nodes)
    # with no initial state
    test_gradients(cell, g, g.x, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(cell, g, g.x, h, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
end


@testitem "GConvGRU" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    layer = GConvGRU(in_channel => out_channel, 2)
    @test layer isa GNNRecurrence
    x = rand(Float32, in_channel, timesteps, g.num_nodes)
    state0 = rand(Float32, out_channel, g.num_nodes)
    y = layer(g, x)
    @test size(y) == (out_channel, timesteps, g.num_nodes)
    # with no initial state
    test_gradients(layer, g, x, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(layer, g, x, state0, rtol=RTOL_LOW, atol=ATOL_LOW)

    # interplay with GNNChain
    model = GNNChain(GConvGRU(in_channel => out_channel, 2), Dense(out_channel, 1))
    y = model(g, x)
    @test size(y) == (1, timesteps, g.num_nodes)
    test_gradients(model, g, x, rtol = RTOL_LOW, atol = ATOL_LOW)
end

@testitem "DCGRUCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    cell = DCGRUCell(in_channel => out_channel, 2)
    y, h = cell(g, g.x)
    @test y === h
    @test size(h) == (out_channel, g.num_nodes)
    # with no initial state
    test_gradients(cell, g, g.x, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(cell, g, g.x, h, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
end

@testitem "DCGRU" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    layer = DCGRU(in_channel => out_channel, 2)
    @test layer isa GNNRecurrence
    x = rand(Float32, in_channel, timesteps, g.num_nodes)
    state0 = rand(Float32, out_channel, g.num_nodes)
    y = layer(g, x)
    @test size(y) == (out_channel, timesteps, g.num_nodes)
    # with no initial state
    test_gradients(layer, g, x, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(layer, g, x, state0, rtol=RTOL_LOW, atol=ATOL_LOW)

    # interplay with GNNChain
    model = GNNChain(DCGRU(in_channel => out_channel, 2), Dense(out_channel, 1))
    y = model(g, x)
    @test size(y) == (1, timesteps, g.num_nodes)
    test_gradients(model, g, x, rtol = RTOL_LOW, atol = ATOL_LOW)
end

@testitem "EvolveGCNOCell" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    cell = EvolveGCNOCell(in_channel => out_channel)
    y, state = cell(g, g.x)
    @test size(y) == (out_channel, g.num_nodes)
    # with no initial state
    test_gradients(cell, g, g.x, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(cell, g, g.x, state, loss=cell_loss, rtol=RTOL_LOW, atol=ATOL_LOW)
end

@testitem "EvolveGCNO" setup=[TemporalConvTestModule, TestModule] begin
    using .TemporalConvTestModule, .TestModule
    layer = EvolveGCNO(in_channel => out_channel)
    @test layer isa GNNRecurrence
    x = rand(Float32, in_channel, timesteps, g.num_nodes)
    state0 = Flux.initialstates(layer)
    y = layer(g, x)
    @test size(y) == (out_channel, timesteps, g.num_nodes)
    # with no initial state
    test_gradients(layer, g, x, rtol=RTOL_LOW, atol=ATOL_LOW)
    # with initial state
    test_gradients(layer, g, x, state0, rtol=RTOL_LOW, atol=ATOL_LOW)

    # interplay with GNNChain
    model = GNNChain(EvolveGCNO(in_channel => out_channel), Dense(out_channel, 1))
    y = model(g, x)
    @test size(y) == (1, timesteps, g.num_nodes)
    test_gradients(model, g, x, rtol=RTOL_LOW, atol=ATOL_LOW)
end

# @testitem "GINConv" setup=[TemporalConvTestModule, TestModule] begin
#     using .TemporalConvTestModule, .TestModule
#     ginconv = GINConv(Dense(in_channel => out_channel),0.3)
#     @test length(ginconv(tg, tg.ndata.x)) == S
#     @test size(ginconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
#     @test length(Flux.gradient(x ->sum(sum(ginconv(tg, x))), tg.ndata.x)[1]) == S    
# end

# @testitem "GraphConv" setup=[TemporalConvTestModule, TestModule] begin
#     using .TemporalConvTestModule, .TestModule
#     graphconv = GraphConv(in_channel => out_channel, tanh)
#     @test length(graphconv(tg, tg.ndata.x)) == S
#     @test size(graphconv(tg, tg.ndata.x)[1]) == (out_channel, N) 
#     @test length(Flux.gradient(x ->sum(sum(graphconv(tg, x))), tg.ndata.x)[1]) == S    
# end

