@testitem "rand_graph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    n, m = 10, 20
    m2 = m ÷ 2
    x = rand(3, n)
    e = rand(4, m2)
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(n, m, ndata = x, edata = e, graph_type = GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m
        @test g.ndata.x === x
        if GRAPH_T == :coo
            s, t = edge_index(g)
            @test s[1:m2] == t[(m2 + 1):end]
            @test t[1:m2] == s[(m2 + 1):end]
            @test g.edata.e[:, 1:m2] == e
            @test g.edata.e[:, (m2 + 1):end] == e
        end

        rng = MersenneTwister(17)
        g = rand_graph(rng, n, m, bidirected = false, graph_type = GRAPH_T)
        @test g.num_nodes == n
        @test g.num_edges == m

        rng = MersenneTwister(17)
        g2 = rand_graph(rng, n, m, bidirected = false, graph_type = GRAPH_T)
        @test edge_index(g2) == edge_index(g)

        ew = rand(m2)
        rng = MersenneTwister(17)
        g = rand_graph(rng, n, m, bidirected = true, graph_type = GRAPH_T, edge_weight = ew)
        @test get_edge_weight(g) == [ew; ew] broken=(GRAPH_T != :coo)
        
        ew = rand(m)
        rng = MersenneTwister(17)
        g = rand_graph(n, m, bidirected = false, graph_type = GRAPH_T, edge_weight = ew)
        @test get_edge_weight(g) == ew broken=(GRAPH_T != :coo)
    end
end

@testitem "knn_graph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        n, k = 10, 3
        x = rand(3, n)
        g = knn_graph(x, k; graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n * k
        @test degree(g, dir = :in) == fill(k, n)
        @test has_self_loops(g) == false

        g = knn_graph(x, k; dir = :out, self_loops = true, graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == n * k
        @test degree(g, dir = :out) == fill(k, n)
        @test has_self_loops(g) == true

        graph_indicator = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        g = knn_graph(x, k; graph_indicator, graph_type = GRAPH_T)
        @test g.num_graphs == 2
        s, t = edge_index(g)
        ne = n * k ÷ 2
        @test all(1 .<= s[1:ne] .<= 5)
        @test all(1 .<= t[1:ne] .<= 5)
        @test all(6 .<= s[(ne + 1):end] .<= 10)
        @test all(6 .<= t[(ne + 1):end] .<= 10)
    end
end

@testitem "radius_graph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        n, r = 10, 0.5
        x = rand(3, n)
        g = radius_graph(x, r; graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test has_self_loops(g) == false

        g = radius_graph(x, r; dir = :out, self_loops = true, graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test has_self_loops(g) == true

        graph_indicator = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        g = radius_graph(x, r; graph_indicator, graph_type = GRAPH_T)
        @test g.num_graphs == 2
        s, t = edge_index(g)
        @test (s .> 5) == (t .> 5)
    end
end

@testitem "rand_bipartite_heterograph" begin
    g = rand_bipartite_heterograph((10, 15), (20, 20))
    @test g.num_nodes == Dict(:A => 10, :B => 15)
    @test g.num_edges == Dict((:A, :to, :B) => 20, (:B, :to, :A) => 20)
    sA, tB = edge_index(g, (:A, :to, :B))
    for (s, t) in zip(sA, tB)
        @test 1 <= s <= 10
        @test 1 <= t <= 15
        @test has_edge(g, (:A,:to,:B), s, t)
        @test has_edge(g, (:B,:to,:A), t, s)
    end

    g = rand_bipartite_heterograph((2, 2), (4, 0), bidirected=false)
    @test has_edge(g, (:A,:to,:B), 1, 1)
    @test !has_edge(g, (:B,:to,:A), 1, 1)
end

@testitem "rand_temporal_radius_graph" begin
    using Statistics
    number_nodes = 30
    number_snapshots = 5
    r = 0.1
    speed = 0.1
    tg = rand_temporal_radius_graph(number_nodes, number_snapshots, speed, r)
    @test tg.num_nodes == [number_nodes for i in 1:number_snapshots]
    @test tg.num_snapshots == number_snapshots
    r2 = 0.95
    tg2 = rand_temporal_radius_graph(number_nodes, number_snapshots, speed, r2)
    @test mean(mean(degree.(tg.snapshots)))<=mean(mean(degree.(tg2.snapshots)))
end

@testitem "rand_temporal_hyperbolic_graph" begin
    using Statistics
    @test GNNGraphs._hyperbolic_distance([1.0,1.0],[1.0,1.0];ζ=1)==0
    @test GNNGraphs._hyperbolic_distance([0.23,0.11],[0.98,0.55];ζ=1) == GNNGraphs._hyperbolic_distance([0.98,0.55],[0.23,0.11];ζ=1)
    number_nodes = 30
    number_snapshots = 5
    α, R, speed, ζ = 1, 1, 0.1, 1
  
    tg = rand_temporal_hyperbolic_graph(number_nodes, number_snapshots; α, R, speed, ζ)
    @test tg.num_nodes == [number_nodes for i in 1:number_snapshots]
    @test tg.num_snapshots == number_snapshots
    R = 10
    tg1 = rand_temporal_hyperbolic_graph(number_nodes, number_snapshots; α, R, speed, ζ)
    @test mean(mean(degree.(tg1.snapshots)))<=mean(mean(degree.(tg.snapshots)))
end
