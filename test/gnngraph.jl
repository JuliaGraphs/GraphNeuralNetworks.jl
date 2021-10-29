@testset "GNNGraph" begin
    @testset "symmetric graph" begin
        s = [1, 1, 2, 2, 3, 3, 4, 4]
        t = [2, 4, 1, 3, 2, 4, 1, 3]
        adj_mat =  [0  1  0  1
                    1  0  1  0
                    0  1  0  1
                    1  0  1  0]
        adj_list_out =  [[2,4], [1,3], [2,4], [1,3]]
        adj_list_in =  [[2,4], [1,3], [2,4], [1,3]]

        # core functionality
        g = GNNGraph(s, t; graph_type=GRAPH_T)
        if TEST_GPU
            g_gpu = g |> gpu
        end

        @test g.num_edges == 8
        @test g.num_nodes == 4
        @test nv(g) == g.num_nodes
        @test ne(g) == g.num_edges
        @test collect(edges(g)) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2, 4] 
        @test sort(inneighbors(g, 1)) == [2, 4] 
        @test is_directed(g) == true
        s1, t1 = GraphNeuralNetworks.sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t
        @test vertices(g) == 1:g.num_nodes
        
        @test sort.(adjacency_list(g; dir=:in)) == adj_list_in
        @test sort.(adjacency_list(g; dir=:out)) == adj_list_out

        @testset "adjacency_matrix" begin
            @test adjacency_matrix(g) == adj_mat
            @test adjacency_matrix(g; dir=:in) == adj_mat
            @test adjacency_matrix(g; dir=:out) == adj_mat
            
            if TEST_GPU
                # See https://github.com/JuliaGPU/CUDA.jl/pull/1093
                mat_gpu = adjacency_matrix(g_gpu)
                @test mat_gpu isa ACUMatrix{Int}
                @test Array(mat_gpu) == adj_mat 
            end
        end
        
        @testset "normalized_laplacian" begin
            mat = normalized_laplacian(g)
            if TEST_GPU
                mat_gpu = normalized_laplacian(g_gpu)
                @test mat_gpu isa ACUMatrix{Float32}
                @test Array(mat_gpu) == mat 
            end
        end


        @testset "scaled_laplacian" begin
            if TEST_GPU
                @test_broken begin 
                    mat = scaled_laplacian(g)
                    mat_gpu = scaled_laplacian(g_gpu)
                    @test mat_gpu isa ACUMatrix{Float32}
                    @test Array(mat_gpu) == mat
                end
            end
        end

        @testset "constructors" begin
            adjacency_matrix(g; dir=:out) == adj_mat
            adjacency_matrix(g; dir=:in) == adj_mat
        end 

        @testset "degree" begin
            @test degree(g, dir=:out) == vec(sum(adj_mat, dims=2))
            @test degree(g, dir=:in) == vec(sum(adj_mat, dims=1))

            if TEST_GPU
                d = degree(g)
                d_gpu = degree(g_gpu)
                @test d_gpu isa CuVector
                @test Array(d_gpu) == d
            end
        end

        if TEST_GPU
            @testset "functor" begin                
                s_cpu, t_cpu = edge_index(g)
                s_gpu, t_gpu = edge_index(g_gpu)
                @test s_gpu isa CuVector{Int}
                @test Array(s_gpu) == s_cpu
                @test t_gpu isa CuVector{Int}
                @test Array(t_gpu) == t_cpu
            end
        end
    end

    @testset "asymmetric graph" begin
        s = [1, 2, 3, 4]
        t = [2, 3, 4, 1]
        adj_mat_out =  [0  1  0  0
                        0  0  1  0
                        0  0  0  1
                        1  0  0  0]
        adj_list_out =  [[2], [3], [4], [1]]


        adj_mat_in =   [0  0  0  1
                        1  0  0  0
                        0  1  0  0
                        0  0  1  0]
        adj_list_in =  [[4], [1], [2], [3]]

        # core functionality
        g = GNNGraph(s, t; graph_type=GRAPH_T)
        if TEST_GPU
            g_gpu = g |> gpu
        end

        @test g.num_edges == 4
        @test g.num_nodes == 4
        @test collect(edges(g)) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2] 
        @test sort(inneighbors(g, 1)) == [4] 
        @test is_directed(g) == true
        @test is_directed(typeof(g)) == true
        s1, t1 = GraphNeuralNetworks.sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t

        # adjacency
        @test adjacency_matrix(g) ==  adj_mat_out
        @test adjacency_list(g) ==  adj_list_out
        @test adjacency_matrix(g, dir=:out) ==  adj_mat_out
        @test adjacency_list(g, dir=:out) ==  adj_list_out
        @test adjacency_matrix(g, dir=:in) ==  adj_mat_in
        @test adjacency_list(g, dir=:in) ==  adj_list_in

        @testset "degree" begin
            @test degree(g, dir=:out) == vec(sum(adj_mat_out, dims=2))
            @test degree(g, dir=:in) == vec(sum(adj_mat_out, dims=1))
        end
    end

    @testset "Graphs constructor" begin
        lg = random_regular_graph(10, 4)
        @test !Graphs.is_directed(lg)
        g = GNNGraph(lg)
        @test g.num_edges == 2*ne(lg) # g in undirected
        @test Graphs.is_directed(g)
        for e in Graphs.edges(lg)
            i, j = src(e), dst(e)
            @test has_edge(g, i, j)
            @test has_edge(g, j, i)            
        end
    end

    @testset "add self-loops" begin
        A = [1  1  0  0
             0  0  1  0
             0  0  0  1
             1  0  0  0]
        A2 =   [2  1  0  0
                0  1  1  0
                0  0  1  1
                1  0  0  1]

        g = GNNGraph(A; graph_type=GRAPH_T)
        fg2 = add_self_loops(g)
        @test adjacency_matrix(g) == A
        @test g.num_edges == sum(A)
        @test adjacency_matrix(fg2) == A2
        @test fg2.num_edges == sum(A2)
    end

    @testset "batch"  begin
        #TODO add graph_type=GRAPH_T
        g1 = GNNGraph(random_regular_graph(10,2), ndata=rand(16,10))
        g2 = GNNGraph(random_regular_graph(4,2), ndata=rand(16,4))
        g3 = GNNGraph(random_regular_graph(7,2), ndata=rand(16,7))
        
        g12 = Flux.batch([g1, g2])
        g12b = blockdiag(g1, g2)
        
        g123 = Flux.batch([g1, g2, g3])
        @test g123.graph_indicator == [fill(1, 10); fill(2, 4); fill(3, 7)]

        s, t = edge_index(g123)
        @test s == [edge_index(g1)[1]; 10 .+ edge_index(g2)[1]; 14 .+ edge_index(g3)[1]] 
        @test t == [edge_index(g1)[2]; 10 .+ edge_index(g2)[2]; 14 .+ edge_index(g3)[2]] 
        @test node_features(g123)[:,11:14] ≈ node_features(g2) 

        # scalar graph features
        g1 = GNNGraph(random_regular_graph(10,2), gdata=rand())
        g2 = GNNGraph(random_regular_graph(4,2), gdata=rand())
        g3 = GNNGraph(random_regular_graph(4,2), gdata=rand())
        g123 = Flux.batch([g1, g2, g3])
        @test g123.gdata.u == [g1.gdata.u, g2.gdata.u, g3.gdata.u]
    end

    @testset "getgraph"  begin
        g1 = GNNGraph(random_regular_graph(10,2), ndata=rand(16,10), graph_type=GRAPH_T)
        g2 = GNNGraph(random_regular_graph(4,2), ndata=rand(16,4), graph_type=GRAPH_T)
        g3 = GNNGraph(random_regular_graph(7,2), ndata=rand(16,7), graph_type=GRAPH_T)
        g = Flux.batch([g1, g2, g3])
        
        g2b, nodemap = getgraph(g, 2, nmap=true)
        s, t = edge_index(g2b)
        @test s == edge_index(g2)[1]
        @test t == edge_index(g2)[2] 
        @test node_features(g2b) ≈ node_features(g2) 

        g2c = getgraph(g, 2)
        @test g2c isa GNNGraph{typeof(g.graph)}

        g1b, nodemap = getgraph(g1, 1, nmap=true)
        @test g1b === g1
        @test nodemap == 1:g1.num_nodes
    end

    @testset "Features" begin
        g = GNNGraph(sprand(10, 10, 0.3), graph_type=GRAPH_T)
        
        # default names
        X = rand(10, g.num_nodes)
        E = rand(10, g.num_edges)
        U = rand(10, g.num_graphs)
        
        g = GNNGraph(g, ndata=X, edata=E, gdata=U)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U

        # Check no args
        g = GNNGraph(g)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U


        # multiple features names
        g = GNNGraph(g, ndata=(x2=2X, g.ndata...), edata=(e2=2E, g.edata...), gdata=(u2=2U, g.gdata...))
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U
        @test g.ndata.x2 ≈ 2X
        @test g.edata.e2 ≈ 2E
        @test g.gdata.u2 ≈ 2U

        # Dimension checks
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=rand(29), graph_type=GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=rand(2, 29), graph_type=GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata=(; x=rand(30), y=rand(29)), graph_type=GRAPH_T)

        # Copy features on reverse edge
        e = rand(30)
        g = GNNGraph(erdos_renyi(10,  30), edata=e, graph_type=GRAPH_T)
        @test g.edata.e == [e; e]


        # Attach non array data
        g = GNNGraph(erdos_renyi(10,  30), edata="ciao", graph_type=GRAPH_T)
        @test g.edata.e == "ciao"
    end 

    @testset "LearnBase and DataLoader compat" begin
        n, m, num_graphs = 10, 30, 50
        X = rand(10, n)
        E = rand(10, 2m)
        U = rand(10, 1)
        g = Flux.batch([GNNGraph(erdos_renyi(n, m), ndata=X, edata=E, gdata=U) 
                        for _ in 1:num_graphs])
        
        @test LearnBase.getobs(g, 3) == getgraph(g, 3)
        @test LearnBase.getobs(g, 3:5) == getgraph(g, 3:5)
        @test StatsBase.nobs(g) == g.num_graphs
        
        d = Flux.Data.DataLoader(g, batchsize = 2, shuffle=false)
        @test first(d) == getgraph(g, 1:2)
    end

    @testset "Graphs.jl integration" begin
        g = GNNGraph(erdos_renyi(10, 20))
        @test g isa Graphs.AbstractGraph
    end
end


