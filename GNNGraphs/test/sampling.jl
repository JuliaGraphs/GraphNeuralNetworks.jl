@testitem "sample_neighbors" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        GRAPH_T != :coo && continue # TODO
    # replace = false
        dir = :in
        nodes = 2:3
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        sg = sample_neighbors(g, nodes; dir)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(degree(g, i; dir) for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        @test length(union(sg.edata.EID)) == length(sg.edata.EID)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(t .∈ Ref(nodes))
        for i in nodes
            @test sort(neighbors(sg, i; dir)) == sort(neighbors(g, i; dir))
        end

        # replace = true
        dir = :out
        nodes = 2:3
        K = 2
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        sg = sample_neighbors(g, nodes, K; dir, replace = true)
        @test sg.num_nodes == 10
        @test sg.num_edges == sum(K for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        adjlist = adjacency_list(g; dir)
        s, t = edge_index(sg)
        @test all(s .∈ Ref(nodes))
        for i in nodes
            @test issubset(neighbors(sg, i; dir), adjlist[i])
        end

        # dropnodes = true
        dir = :in
        nodes = 2:3
        g = rand_graph(10, 40, bidirected = false, graph_type = GRAPH_T)
        g = GNNGraph(g, ndata = (x1 = rand(10),), edata = (e1 = rand(40),))
        sg = sample_neighbors(g, nodes; dir, dropnodes = true)
        @test sg.num_edges == sum(degree(g, i; dir) for i in nodes)
        @test size(sg.edata.EID) == (sg.num_edges,)
        @test size(sg.ndata.NID) == (sg.num_nodes,)
        @test sg.edata.e1 == g.edata.e1[sg.edata.EID]
        @test sg.ndata.x1 == g.ndata.x1[sg.ndata.NID]
        @test length(union(sg.ndata.NID)) == length(sg.ndata.NID)
    end
end

@testitem "induced_subgraph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using MLUtils: getobs
    s = [1, 2]
    t = [2, 3]
    
    graph = GNNGraph((s, t), ndata = (; x=rand(Float32, 32, 3), y=rand(Float32, 3)), edata = rand(Float32, 2))
    
    nodes = [1, 2, 3]
    subgraph = Graphs.induced_subgraph(graph, nodes)
    
    @test subgraph.num_nodes == 3  
    @test subgraph.num_edges == 2  
    @test subgraph.ndata.x == graph.ndata.x
    @test subgraph.ndata.y == graph.ndata.y
    @test subgraph.edata == graph.edata
    
    nodes = [1, 2]
    subgraph = Graphs.induced_subgraph(graph, nodes)

    @test subgraph.num_nodes == 2 
    @test subgraph.num_edges == 1 
    @test subgraph.ndata == getobs(graph.ndata, [1, 2])
    @test isapprox(getobs(subgraph.edata.e, 1), getobs(graph.edata.e, 1); atol=1e-6)

    graph = GNNGraph(2)
    graph = add_edges(graph, ([2], [1]))
    nodes = [1]
    subgraph = Graphs.induced_subgraph(graph, nodes)
    
    @test subgraph.num_nodes == 1 
    @test subgraph.num_edges == 0 
end

@testitem "NeighborLoader"  setup=[GraphsTestModule] begin
    using .GraphsTestModule
    # Helper function to create a simple graph with node features using GNNGraph
    function create_test_graph()
        source = [1, 2, 3, 4]  # Define source nodes of edges
        target = [2, 3, 4, 5]  # Define target nodes of edges
        node_features = rand(Float32, 5, 5)  # Create random node features (5 features for 5 nodes)

        return GNNGraph(source, target, ndata = node_features)  # Create a GNNGraph with edges and features
    end


    # 1. Basic functionality: Check neighbor sampling and subgraph creation
    @testset "Basic functionality" begin
        g = create_test_graph()

        # Define NeighborLoader with 2 neighbors per layer, 2 layers, batch size 2
        loader = NeighborLoader(g; num_neighbors=[2, 2], input_nodes=[1, 2], num_layers=2, batch_size=2)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph is not empty
        @test !isempty(mini_batch_gnn.graph)

        num_sampled_nodes = mini_batch_gnn.num_nodes

        @test num_sampled_nodes == 2

        # Test if there are edges in the subgraph
        @test mini_batch_gnn.num_edges > 0
    end

    # 2. Edge case: Single node with no neighbors
    @testset "Single node with no neighbors" begin
        g = SimpleDiGraph(1)  # A graph with a single node and no edges
        node_features = rand(Float32, 5, 1)
        graph = GNNGraph(g, ndata = node_features)

        loader = NeighborLoader(graph; num_neighbors=[2], input_nodes=[1], num_layers=1)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph contains only one node
        @test size(mini_batch_gnn.x, 2) == 1
    end

    # 3. Edge case: A node with no outgoing edges (isolated node)
    @testset "Node with no outgoing edges" begin
        g = SimpleDiGraph(2)  # Graph with 2 nodes, no edges
        node_features = rand(Float32, 5, 2)
        graph = GNNGraph(g, ndata = node_features)

        loader = NeighborLoader(graph; num_neighbors=[1], input_nodes=[1, 2], num_layers=1)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph contains the input nodes only (as no neighbors can be sampled)
        @test size(mini_batch_gnn.x, 2) == 2  # Only two isolated nodes
    end

    # 4. Edge case: A fully connected graph
    @testset "Fully connected graph" begin
        g = SimpleDiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        add_edge!(g, 3, 1)
        node_features = rand(Float32, 5, 3)
        graph = GNNGraph(g, ndata = node_features)

        loader = NeighborLoader(graph; num_neighbors=[2, 2], input_nodes=[1], num_layers=2)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if all nodes are included in the mini-batch since it's fully connected
        @test size(mini_batch_gnn.x, 2) == 3  # All nodes should be included
    end

    # 5. Edge case: More layers than the number of neighbors
    @testset "More layers than available neighbors" begin
        g = SimpleDiGraph(3)
        add_edge!(g, 1, 2)
        add_edge!(g, 2, 3)
        node_features = rand(Float32, 5, 3)
        graph = GNNGraph(g, ndata = node_features)

        # Test with 3 layers but only enough connections for 2 layers
        loader = NeighborLoader(graph; num_neighbors=[1, 1, 1], input_nodes=[1], num_layers=3)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph contains all available nodes
        @test size(mini_batch_gnn.x, 2) == 1
    end

    # 6. Edge case: Large batch size greater than the number of input nodes
    @testset "Large batch size" begin
        g = create_test_graph()

        # Define NeighborLoader with a larger batch size than input nodes
        loader = NeighborLoader(g; num_neighbors=[2], input_nodes=[1, 2], num_layers=1, batch_size=10)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph is not empty
        @test !isempty(mini_batch_gnn.graph)

        # Test if the correct number of nodes are sampled
        @test size(mini_batch_gnn.x, 2) == length(unique([1, 2]))  # Nodes [1, 2] are expected
    end

    # 7. Edge case: No neighbors sampled (num_neighbors = [0]) and 1 layer
    @testset "No neighbors sampled" begin
        g = create_test_graph()

        # Define NeighborLoader with 0 neighbors per layer, 1 layer, batch size 2
        loader = NeighborLoader(g; num_neighbors=[0], input_nodes=[1, 2], num_layers=1, batch_size=2)

        mini_batch_gnn, next_state = iterate(loader)

        # Test if the mini-batch graph contains only the input nodes
        @test size(mini_batch_gnn.x, 2) == 2  # No neighbors should be sampled, only nodes 1 and 2 should be in the graph
    end
end
