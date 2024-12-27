@testitem "add self-loops" setup=[GraphsTestModule] begin
    using .GraphsTestModule

    A = [1 1 0 0
         0 0 1 0
         0 0 0 1
         1 0 0 0]
    A2 = [2 1 0 0
          0 1 1 0
          0 0 1 1
          1 0 0 1]

    for GRAPH_T in GRAPH_TYPES
        g = GNNGraph(A; graph_type = GRAPH_T)
        fg2 = add_self_loops(g)
        @test adjacency_matrix(g) == A
        @test g.num_edges == sum(A)
        @test adjacency_matrix(fg2) == A2
        @test fg2.num_edges == sum(A2)
    end
end

@testitem "batch" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES 
        g1 = GNNGraph(random_regular_graph(10, 2), ndata = rand(16, 10),
                    graph_type = GRAPH_T)
        g2 = GNNGraph(random_regular_graph(4, 2), ndata = rand(16, 4), graph_type = GRAPH_T)
        g3 = GNNGraph(random_regular_graph(7, 2), ndata = rand(16, 7), graph_type = GRAPH_T)

        g12 = MLUtils.batch([g1, g2])
        g12b = blockdiag(g1, g2)
        @test g12 == g12b

        g123 = MLUtils.batch([g1, g2, g3])
        @test g123.graph_indicator == [fill(1, 10); fill(2, 4); fill(3, 7)]

        # Allow wider eltype
        g123 = MLUtils.batch(GNNGraph[g1, g2, g3])
        @test g123.graph_indicator == [fill(1, 10); fill(2, 4); fill(3, 7)]


        s, t = edge_index(g123)
        @test s == [edge_index(g1)[1]; 10 .+ edge_index(g2)[1]; 14 .+ edge_index(g3)[1]]
        @test t == [edge_index(g1)[2]; 10 .+ edge_index(g2)[2]; 14 .+ edge_index(g3)[2]]
        @test node_features(g123)[:, 11:14] ≈ node_features(g2)

        # scalar graph features
        g1 = GNNGraph(g1, gdata = rand())
        g2 = GNNGraph(g2, gdata = rand())
        g3 = GNNGraph(g3, gdata = rand())
        g123 = MLUtils.batch([g1, g2, g3])
        @test g123.gdata.u == [g1.gdata.u, g2.gdata.u, g3.gdata.u]

        # Batch of batches
        g123123 = MLUtils.batch([g123, g123])
        @test g123123.graph_indicator ==
            [fill(1, 10); fill(2, 4); fill(3, 7); fill(4, 10); fill(5, 4); fill(6, 7)]
        @test g123123.num_graphs == 6
    end
end

@testitem "unbatch" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g1 = rand_graph(10, 20, graph_type = GRAPH_T)
        g2 = rand_graph(5, 10, graph_type = GRAPH_T)
        g12 = MLUtils.batch([g1, g2])
        gs = MLUtils.unbatch([g1, g2])
        @test length(gs) == 2
        @test gs[1].num_nodes == 10
        @test gs[1].num_edges == 20
        @test gs[1].num_graphs == 1
        @test gs[2].num_nodes == 5
        @test gs[2].num_edges == 10
        @test gs[2].num_graphs == 1
    end
end

@testitem "batch/unbatch roundtrip" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        n = 20
        c = 3
        ngraphs = 10
        gs = [rand_graph(n, c * n, ndata = rand(2, n), edata = rand(3, c * n),
                        graph_type = GRAPH_T)
            for _ in 1:ngraphs]
        gall = MLUtils.batch(gs)
        gs2 = MLUtils.unbatch(gall)
        @test gs2[1] == gs[1]
        @test gs2[end] == gs[end]
    end
end

@testitem "getgraph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g1 = GNNGraph(random_regular_graph(10, 2), ndata = rand(16, 10),
                    graph_type = GRAPH_T)
        g2 = GNNGraph(random_regular_graph(4, 2), ndata = rand(16, 4), graph_type = GRAPH_T)
        g3 = GNNGraph(random_regular_graph(7, 2), ndata = rand(16, 7), graph_type = GRAPH_T)
        g = MLUtils.batch([g1, g2, g3])

        g2b, nodemap = getgraph(g, 2, nmap = true)
        s, t = edge_index(g2b)
        @test s == edge_index(g2)[1]
        @test t == edge_index(g2)[2]
        @test node_features(g2b) ≈ node_features(g2)

        g2c = getgraph(g, 2)
        @test g2c isa GNNGraph{typeof(g.graph)}

        g1b, nodemap = getgraph(g1, 1, nmap = true)
        @test g1b === g1
        @test nodemap == 1:(g1.num_nodes)
    end 
end

@testitem "remove_edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s = [1, 1, 2, 3]
            t = [2, 3, 4, 5]
            w = [0.1, 0.2, 0.3, 0.4]
            edata = ['a', 'b', 'c', 'd']
            g = GNNGraph(s, t, w, edata = edata, graph_type = GRAPH_T)    

            # single edge removal
            gnew = remove_edges(g, [1])
            new_s, new_t = edge_index(gnew)
            @test gnew.num_edges == 3
            @test new_s == s[2:end]
            @test new_t == t[2:end]
            
            # multiple edge removal
            gnew = remove_edges(g, [1,2,4])
            new_s, new_t = edge_index(gnew)
            new_w = get_edge_weight(gnew)
            new_edata = gnew.edata.e
            @test gnew.num_edges == 1
            @test new_s == [2]
            @test new_t == [4]
            @test new_w == [0.3]
            @test new_edata == ['c']

            # drop with probability
            gnew = remove_edges(g, Float32(1.0))
            @test gnew.num_edges == 0

            gnew = remove_edges(g, Float32(0.0))
            @test gnew.num_edges == g.num_edges
        end
    end
end

@testitem "add_edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s = [1, 1, 2, 3]
            t = [2, 3, 4, 5]
            g = GNNGraph(s, t, graph_type = GRAPH_T)
            snew = [1]
            tnew = [4]
            gnew = add_edges(g, snew, tnew)
            @test gnew.num_edges == 5
            @test sort(inneighbors(gnew, 4)) == [1, 2]

            gnew2 = add_edges(g, (snew, tnew))
            @test gnew2 == gnew
            @test get_edge_weight(gnew2) === nothing

            g = GNNGraph(s, t, edata = (e1 = rand(2, 4), e2 = rand(3, 4)), graph_type = GRAPH_T)
            # @test_throws ErrorException add_edges(g, snew, tnew)
            gnew = add_edges(g, snew, tnew, edata = (e1 = ones(2, 1), e2 = zeros(3, 1)))
            @test all(gnew.edata.e1[:, 5] .== 1)
            @test all(gnew.edata.e2[:, 5] .== 0)

            @testset "adding new nodes" begin
                g = GNNGraph()
                g = add_edges(g, ([1,3], [2, 1]))
                @test g.num_nodes == 3
                @test g.num_edges == 2
                @test sort(inneighbors(g, 1)) == [3]
                @test sort(outneighbors(g, 1)) == [2]
            end
            @testset "also add weights" begin
                s = [1, 1, 2, 3]
                t = [2, 3, 4, 5]
                w = [1.0, 2.0, 3.0, 4.0]
                snew = [1]
                tnew = [4]
                wnew = [5.]

                g = GNNGraph((s, t), graph_type = GRAPH_T)
                gnew = add_edges(g, (snew, tnew, wnew))
                @test get_edge_weight(gnew) == [ones(length(s)); wnew]
                
                g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                gnew = add_edges(g, (snew, tnew, wnew))
                @test get_edge_weight(gnew) == [w; wnew]
            end
        end
    end
end

@testitem "perturb_edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s, t = [1, 2, 3, 4, 5], [2, 3, 4, 5, 1]
            g = GNNGraph((s, t), graph_type = GRAPH_T)
            rng = MersenneTwister(42)
            g_per = perturb_edges(rng, g, 0.5)
            @test g_per.num_edges == 8
        end
    end
end

@testitem "remove_nodes" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES 
        if GRAPH_T == :coo
            #single node
            s = [1, 1, 2, 3]
            t = [2, 3, 4, 5]
            eweights = [0.1, 0.2, 0.3, 0.4]
            ndata = [1.0, 2.0, 3.0, 4.0, 5.0]
            edata = ['a', 'b', 'c', 'd']

            g = GNNGraph(s, t, eweights, ndata = ndata, edata = edata, graph_type = GRAPH_T)

            gnew = remove_nodes(g, [1])

            snew = [1, 2]
            tnew = [3, 4]
            eweights_new = [0.3, 0.4]
            ndata_new = [2.0, 3.0, 4.0, 5.0]
            edata_new = ['c', 'd']

            stest, ttest = edge_index(gnew)
            eweightstest = get_edge_weight(gnew)
            ndatatest = gnew.ndata.x
            edatatest = gnew.edata.e


            @test gnew.num_edges == 2
            @test gnew.num_nodes == 4
            @test snew == stest
            @test tnew == ttest
            @test eweights_new == eweightstest
            @test ndata_new == ndatatest
            @test edata_new == edatatest

            # multiple nodes
            s = [1, 5, 2, 3]
            t = [2, 3, 4, 5]
            eweights = [0.1, 0.2, 0.3, 0.4]
            ndata = [1.0, 2.0, 3.0, 4.0, 5.0]
            edata = ['a', 'b', 'c', 'd']

            g = GNNGraph(s, t, eweights, ndata = ndata, edata = edata, graph_type = GRAPH_T)

            gnew = remove_nodes(g, [1,4])
            snew = [3,2]
            tnew = [2,3]
            eweights_new = [0.2,0.4]
            ndata_new = [2.0,3.0,5.0]
            edata_new = ['b','d']

            stest, ttest = edge_index(gnew)
            eweightstest = get_edge_weight(gnew)
            ndatatest = gnew.ndata.x
            edatatest = gnew.edata.e

            @test gnew.num_edges == 2
            @test gnew.num_nodes == 3
            @test snew == stest
            @test tnew == ttest
            @test eweights_new == eweightstest
            @test ndata_new == ndatatest
            @test edata_new == edatatest
        end
    end 
end

@testitem "remove_nodes(g, p)" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            Random.seed!(42)
            s = [1, 1, 2, 3]
            t = [2, 3, 4, 5]
            g = GNNGraph(s, t, graph_type = GRAPH_T)    
            
            gnew = remove_nodes(g, 0.5)
            @test gnew.num_nodes == 3

            gnew = remove_nodes(g, 1.0)
            @test gnew.num_nodes == 0

            gnew = remove_nodes(g, 0.0)
            @test gnew.num_nodes == 5
        end
    end
end

@testitem "add_nodes" setup=[GraphsTestModule] begin 
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            g = rand_graph(6, 4, ndata = rand(2, 6), graph_type = GRAPH_T)
            gnew = add_nodes(g, 5, ndata = ones(2, 5))
            @test gnew.num_nodes == g.num_nodes + 5
            @test gnew.num_edges == g.num_edges
            @test gnew.num_graphs == g.num_graphs
            @test all(gnew.ndata.x[:, 7:11] .== 1)
        end
    end
end

@testitem "remove_self_loops" setup=[GraphsTestModule] begin
    using .GraphsTestModule 
    for GRAPH_T in GRAPH_TYPES 
        if GRAPH_T == :coo # add_edges and set_edge_weight only implemented for coo
            g = rand_graph(10, 20, graph_type = GRAPH_T)
            g1 = add_edges(g, [1:5;], [1:5;])
            @test g1.num_edges == g.num_edges + 5
            g2 = remove_self_loops(g1)
            @test g2.num_edges == g.num_edges
            @test sort_edge_index(edge_index(g2)) == sort_edge_index(edge_index(g))

            # with edge features and weights
            g1 = GNNGraph(g1, edata = (e1 = ones(3, g1.num_edges), e2 = 2 * ones(g1.num_edges)))
            g1 = set_edge_weight(g1, 3 * ones(g1.num_edges))
            g2 = remove_self_loops(g1)
            @test g2.num_edges == g.num_edges
            @test sort_edge_index(edge_index(g2)) == sort_edge_index(edge_index(g))
            @test size(get_edge_weight(g2)) == (g2.num_edges,)
            @test size(g2.edata.e1) == (3, g2.num_edges)
            @test size(g2.edata.e2) == (g2.num_edges,)
        end 
    end
end

@testitem "remove_multi_edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            g = rand_graph(10, 20, graph_type = GRAPH_T)
            s, t = edge_index(g)
            g1 = add_edges(g, s[1:5], t[1:5])
            @test g1.num_edges == g.num_edges + 5
            g2 = remove_multi_edges(g1, aggr = +)
            @test g2.num_edges == g.num_edges
            @test sort_edge_index(edge_index(g2)) == sort_edge_index(edge_index(g))

            # Default aggregation is +
            g1 = GNNGraph(g1, edata = (e1 = ones(3, g1.num_edges), e2 = 2 * ones(g1.num_edges)))
            g1 = set_edge_weight(g1, 3 * ones(g1.num_edges))
            g2 = remove_multi_edges(g1)
            @test g2.num_edges == g.num_edges
            @test sort_edge_index(edge_index(g2)) == sort_edge_index(edge_index(g))
            @test count(g2.edata.e1[:, i] == 2 * ones(3) for i in 1:(g2.num_edges)) == 5
            @test count(g2.edata.e2[i] == 4 for i in 1:(g2.num_edges)) == 5
            w2 = get_edge_weight(g2)
            @test count(w2[i] == 6 for i in 1:(g2.num_edges)) == 5
        end
    end 
end

@testitem "negative_sample" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            n, m = 10, 30
            g = rand_graph(n, m, bidirected = true, graph_type = GRAPH_T)

            # check bidirected=is_bidirected(g) default
            gneg = negative_sample(g, num_neg_edges = 20)
            @test gneg.num_nodes == g.num_nodes
            @test gneg.num_edges == 20
            @test is_bidirected(gneg)
            @test intersect(g, gneg).num_edges == 0
        end
    end 
end

@testitem "rand_edge_split" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            n, m = 100, 300

            g = rand_graph(n, m, bidirected = true, graph_type = GRAPH_T)
            # check bidirected=is_bidirected(g) default
            g1, g2 = rand_edge_split(g, 0.9)
            @test is_bidirected(g1)
            @test is_bidirected(g2)
            @test intersect(g1, g2).num_edges == 0
            @test g1.num_edges + g2.num_edges == g.num_edges
            @test g2.num_edges < 50

            g = rand_graph(n, m, bidirected = false, graph_type = GRAPH_T)
            # check bidirected=is_bidirected(g) default
            g1, g2 = rand_edge_split(g, 0.9)
            @test !is_bidirected(g1)
            @test !is_bidirected(g2)
            @test intersect(g1, g2).num_edges == 0
            @test g1.num_edges + g2.num_edges == g.num_edges
            @test g2.num_edges < 50

            g1, g2 = rand_edge_split(g, 0.9, bidirected = false)
            @test !is_bidirected(g1)
            @test !is_bidirected(g2)
            @test intersect(g1, g2).num_edges == 0
            @test g1.num_edges + g2.num_edges == g.num_edges
            @test g2.num_edges < 50
        end
    end
end

@testitem "set_edge_weight" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 20, graph_type = GRAPH_T)
        w = rand(20)

        gw = set_edge_weight(g, w)
        @test get_edge_weight(gw) == w

        # now from weighted graph
        s, t = edge_index(g)
        g2 = GNNGraph(s, t, rand(20), graph_type = GRAPH_T)
        gw2 = set_edge_weight(g2, w)
        @test get_edge_weight(gw2) == w
    end
end

@testitem "to_bidirected" setup=[GraphsTestModule] begin
    using .GraphsTestModule 
    for GRAPH_T in GRAPH_TYPES 
        if GRAPH_T == :coo
            s, t = [1, 2, 3, 3, 4], [2, 3, 4, 4, 4]
            w = [1.0, 2.0, 3.0, 4.0, 5.0]
            e = [10.0, 20.0, 30.0, 40.0, 50.0]
            g = GNNGraph(s, t, w, edata = e)

            g2 = to_bidirected(g)
            @test g2.num_nodes == g.num_nodes
            @test g2.num_edges == 7
            @test is_bidirected(g2)
            @test !has_multi_edges(g2)

            s2, t2 = edge_index(g2)
            w2 = get_edge_weight(g2)
            @test s2 == [1, 2, 2, 3, 3, 4, 4]
            @test t2 == [2, 1, 3, 2, 4, 3, 4]
            @test w2 == [1, 1, 2, 2, 3.5, 3.5, 5]
            @test g2.edata.e == [10.0, 10.0, 20.0, 20.0, 35.0, 35.0, 50.0]
        end 
    end
end

@testitem "to_unidirected" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s = [1, 2, 3, 4, 4]
            t = [2, 3, 4, 3, 4]
            w = [1.0, 2.0, 3.0, 4.0, 5.0]
            e = [10.0, 20.0, 30.0, 40.0, 50.0]
            g = GNNGraph(s, t, w, edata = e)

            g2 = to_unidirected(g)
            @test g2.num_nodes == g.num_nodes
            @test g2.num_edges == 4
            @test !has_multi_edges(g2)

            s2, t2 = edge_index(g2)
            w2 = get_edge_weight(g2)
            @test s2 == [1, 2, 3, 4]
            @test t2 == [2, 3, 4, 4]
            @test w2 == [1, 2, 3.5, 5]
            @test g2.edata.e == [10.0, 20.0, 35.0, 50.0]
        end 
    end
end

@testitem "Graphs.Graph from GNNGraph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 20, graph_type = GRAPH_T)

        G = Graphs.Graph(g)
        @test nv(G) == g.num_nodes
        @test ne(G) == g.num_edges ÷ 2

        DG = Graphs.DiGraph(g)
        @test nv(DG) == g.num_nodes
        @test ne(DG) == g.num_edges
    end
end

@testitem "random_walk_pe" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        s = [1, 2, 2, 3]
        t = [2, 1, 3, 2]
        ndata = [-1, 0, 1]
        g = GNNGraph(s, t, graph_type = GRAPH_T, ndata = ndata)
        output = random_walk_pe(g, 3)
        @test output == [0.0 0.0 0.0
                        0.5 1.0 0.5
                        0.0 0.0 0.0]
    end
end

@testitem "HeteroGraphs" setup = [GraphsTestModule] begin
    using .GraphsTestModule
    @testset "batch" begin
        gs = [rand_bipartite_heterograph((10, 15), 20) for _ in 1:5]
        g = MLUtils.batch(gs)
        @test g.num_nodes[:A] == 50
        @test g.num_nodes[:B] == 75
        @test g.num_edges[(:A,:to,:B)] == 100
        @test g.num_edges[(:B,:to,:A)] == 100
        @test g.num_graphs == 5
        @test g.graph_indicator == Dict(:A => vcat([fill(i, 10) for i in 1:5]...),
                                        :B => vcat([fill(i, 15) for i in 1:5]...))

        for gi in gs
            gi.ndata[:A].x = ones(2, 10)
            gi.ndata[:A].y = zeros(10)
            gi.edata[(:A,:to,:B)].e = fill(2, 20)
            gi.gdata.u = 7
        end
        g = MLUtils.batch(gs)
        @test g.ndata[:A].x == ones(2, 50)
        @test g.ndata[:A].y == zeros(50)
        @test g.edata[(:A,:to,:B)].e == fill(2, 100)
        @test g.gdata.u == fill(7, 5)

        # Allow for wider eltype 
        g = MLUtils.batch(GNNHeteroGraph[g for g in gs])
        @test g.ndata[:A].x == ones(2, 50)
        @test g.ndata[:A].y == zeros(50)
        @test g.edata[(:A,:to,:B)].e == fill(2, 100)
        @test g.gdata.u == fill(7, 5)
    end

    @testset "batch non-similar edge types" begin
        gs = [rand_heterograph((:A =>10, :B => 14), ((:A, :to1, :A) => 5, (:A, :to1, :B) => 20)),
            rand_heterograph((:A => 10, :B => 15), ((:A, :to1, :B) => 5, (:B, :to2, :B) => 16)),
            rand_heterograph((:B => 15, :C => 5), ((:C, :to1, :B) => 5, (:B, :to2, :C) => 21)),
            rand_heterograph((:A => 10, :B => 10, :C => 10), ((:A, :to1, :C) => 5, (:A, :to1, :B) => 5)),
            rand_heterograph((:C => 20), ((:C, :to3, :C) => 10))
        ]
        g = MLUtils.batch(gs)

        @test g.num_nodes[:A] == 10 + 10 + 10
        @test g.num_nodes[:B] == 14 + 15 + 15 + 10
        @test g.num_nodes[:C] == 5 + 10 + 20
        @test g.num_edges[(:A,:to1,:A)] == 5
        @test g.num_edges[(:A,:to1,:B)] == 20 + 5 + 5
        @test g.num_edges[(:A,:to1,:C)] == 5

        @test g.num_edges[(:B,:to2,:B)] == 16
        @test g.num_edges[(:B,:to2,:C)] == 21

        @test g.num_edges[(:C,:to1,:B)] == 5
        @test g.num_edges[(:C,:to3,:C)] == 10
        @test length(keys(g.num_edges)) == 7
        @test g.num_graphs == 5

        function ndata_if_key(g, key, subkey, value)
            if haskey(g.ndata, key)
                g.ndata[key][subkey] = reduce(hcat, fill(value, g.num_nodes[key]))
            end
        end

        function edata_if_key(g, key, subkey, value)
            if haskey(g.edata, key)
                g.edata[key][subkey] = reduce(hcat, fill(value, g.num_edges[key]))
            end
        end

        for gi in gs
            ndata_if_key(gi, :A, :x, [0])
            ndata_if_key(gi, :A, :y, ones(2))
            ndata_if_key(gi, :B, :x, ones(3))
            ndata_if_key(gi, :C, :y, zeros(4))
            edata_if_key(gi, (:A,:to1,:B), :x, [0])
            gi.gdata.u = 7
        end

        g = MLUtils.batch(gs)

        @test g.ndata[:A].x == reduce(hcat, fill(0, 10 + 10 + 10))
        @test g.ndata[:A].y == ones(2, 10 + 10 + 10)
        @test g.ndata[:B].x == ones(3, 14 + 15 + 15 + 10)
        @test g.ndata[:C].y == zeros(4, 5 + 10 + 20)

        @test g.edata[(:A,:to1,:B)].x == reduce(hcat, fill(0, 20 + 5 + 5))

        @test g.gdata.u == fill(7, 5)

        # Allow for wider eltype 
        g = MLUtils.batch(GNNHeteroGraph[g for g in gs])
        @test g.ndata[:A].x == reduce(hcat, fill(0, 10 + 10 + 10))
        @test g.ndata[:A].y == ones(2, 10 + 10 + 10)
        @test g.ndata[:B].x == ones(3, 14 + 15 + 15 + 10)
        @test g.ndata[:C].y == zeros(4, 5 + 10 + 20)

        @test g.edata[(:A,:to1,:B)].x == reduce(hcat, fill(0, 20 + 5 + 5))

        @test g.gdata.u == fill(7, 5)
    end

    @testset "add_edges" begin
        hg = rand_bipartite_heterograph((2, 2), (4, 0), bidirected=false)
        hg = add_edges(hg, (:B,:to,:A), [1, 1], [1,2])
        @test hg.num_edges == Dict((:A,:to,:B) => 4, (:B,:to,:A) => 2)
        @test has_edge(hg, (:B,:to,:A), 1, 1)
        @test has_edge(hg, (:B,:to,:A), 1, 2)
        @test !has_edge(hg, (:B,:to,:A), 2, 1)
        @test !has_edge(hg, (:B,:to,:A), 2, 2)

        @testset "new nodes" begin
            hg = rand_bipartite_heterograph((2, 2), 3)
            hg = add_edges(hg, (:C,:rel,:B) => ([1, 3], [1,2]))
            @test hg.num_nodes == Dict(:A => 2, :B => 2, :C => 3)
            @test hg.num_edges == Dict((:A,:to,:B) => 3, (:B,:to,:A) => 3, (:C,:rel,:B) => 2)
            s, t = edge_index(hg, (:C,:rel,:B))
            @test s == [1, 3]
            @test t == [1, 2]

            hg = add_edges(hg, (:D,:rel,:F) => ([1, 3], [1,2]))
            @test hg.num_nodes == Dict(:A => 2, :B => 2, :C => 3, :D => 3, :F => 2)
            @test hg.num_edges == Dict((:A,:to,:B) => 3, (:B,:to,:A) => 3, (:C,:rel,:B) => 2, (:D,:rel,:F) => 2)
            s, t = edge_index(hg, (:D,:rel,:F))
            @test s == [1, 3]
            @test t == [1, 2]
        end

        @testset "also add weights" begin
            hg = GNNHeteroGraph((:user, :rate, :movie) => ([1,1,2,3], [7,13,5,7], [0.1, 0.2, 0.3, 0.4]))
            hgnew = add_edges(hg, (:user, :like, :actor) => ([1, 2], [3, 4], [0.5, 0.6]))
            @test hgnew.num_nodes[:user] == 3
            @test hgnew.num_nodes[:movie] == 13
            @test hgnew.num_nodes[:actor] == 4
            @test hgnew.num_edges == Dict((:user, :rate, :movie) => 4, (:user, :like, :actor) => 2)
            @test get_edge_weight(hgnew, (:user, :rate, :movie)) == [0.1, 0.2, 0.3, 0.4]
            @test get_edge_weight(hgnew, (:user, :like, :actor)) == [0.5, 0.6]

            hgnew2 = add_edges(hgnew, (:user, :like, :actor) => ([6, 7], [8, 10], [0.7, 0.8]))
            @test hgnew2.num_nodes[:user] == 7
            @test hgnew2.num_nodes[:movie] == 13
            @test hgnew2.num_nodes[:actor] == 10
            @test hgnew2.num_edges == Dict((:user, :rate, :movie) => 4, (:user, :like, :actor) => 4)
            @test get_edge_weight(hgnew2, (:user, :rate, :movie)) == [0.1, 0.2, 0.3, 0.4]
            @test get_edge_weight(hgnew2, (:user, :like, :actor)) == [0.5, 0.6, 0.7, 0.8]
        end
    end

    @testset "add self-loops heterographs" begin
        g = rand_heterograph((:A =>10, :B => 14), ((:A, :to1, :A) => 5, (:A, :to1, :B) => 20))
        # Case in which haskey(g.graph, edge_t) passes
        g = add_self_loops(g, (:A, :to1, :A))

        @test g.num_edges[(:A, :to1, :A)] == 5 + 10
        @test g.num_edges[(:A, :to1, :B)] == 20
        # This test should not use length(keys(g.num_edges)) since that may be undefined behavior
        @test sum(1 for k in keys(g.num_edges) if g.num_edges[k] != 0) == 2

        # Case in which haskey(g.graph, edge_t) fails
        g = add_self_loops(g, (:A, :to3, :A))

        @test g.num_edges[(:A, :to1, :A)] == 5 + 10
        @test g.num_edges[(:A, :to1, :B)] == 20
        @test g.num_edges[(:A, :to3, :A)] == 10
        @test sum(1 for k in keys(g.num_edges) if g.num_edges[k] != 0) == 3

        # Case with edge weights
        g = GNNHeteroGraph(Dict((:A, :to1, :A) => ([1, 2, 3], [3, 2, 1], [2, 2, 2]), (:A, :to2, :B) => ([1, 4, 5], [1, 2, 3])))
        n = g.num_nodes[:A]
        g = add_self_loops(g, (:A, :to1, :A))
        
        @test g.graph[(:A, :to1, :A)][3] == vcat([2, 2, 2], fill(1, n))
    end
end

@testitem "ppr_diffusion" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s = [1, 1, 2, 3]
            t = [2, 3, 4, 5]
            eweights = [0.1, 0.2, 0.3, 0.4]

            g = GNNGraph(s, t, eweights)

            g_new = ppr_diffusion(g)
            w_new = get_edge_weight(g_new)

            check_ew = Float32[0.012749999
                            0.025499998
                            0.038249996
                            0.050999995]

            @test w_new ≈ check_ew
        end
    end
end
