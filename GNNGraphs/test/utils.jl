@testitem "edge encoding/decoding" begin
    # not is_bidirected
    n = 5
    s = [1, 1, 2, 3, 3, 4, 5]
    t = [1, 3, 1, 1, 2, 5, 5]

    # directed=true
    idx, maxid = GNNGraphs.edge_encoding(s, t, n)
    @test maxid == n^2
    @test idx == [1, 3, 6, 11, 12, 20, 25]

    sdec, tdec = GNNGraphs.edge_decoding(idx, n)
    @test sdec == s
    @test tdec == t

    n1, m1 = 10, 30
    g = rand_graph(n1, m1)
    s1, t1 = edge_index(g)
    idx, maxid = GNNGraphs.edge_encoding(s1, t1, n1)
    sdec, tdec = GNNGraphs.edge_decoding(idx, n1)
    @test sdec == s1
    @test tdec == t1

    # directed=false
    idx, maxid = GNNGraphs.edge_encoding(s, t, n, directed = false)
    @test maxid == n * (n + 1) ÷ 2
    @test idx == [1, 3, 2, 3, 7, 14, 15]

    mask = s .> t
    snew = copy(s)
    tnew = copy(t)
    snew[mask] .= t[mask]
    tnew[mask] .= s[mask]
    sdec, tdec = GNNGraphs.edge_decoding(idx, n, directed = false)
    @test sdec == snew
    @test tdec == tnew

    n1, m1 = 6, 8
    g = rand_graph(n1, m1)
    s1, t1 = edge_index(g)
    idx, maxid = GNNGraphs.edge_encoding(s1, t1, n1, directed = false)
    sdec, tdec = GNNGraphs.edge_decoding(idx, n1, directed = false)
    mask = s1 .> t1
    snew = copy(s1)
    tnew = copy(t1)
    snew[mask] .= t1[mask]
    tnew[mask] .= s1[mask]
    @test sdec == snew
    @test tdec == tnew

    @testset "directed=false, self_loops=false" begin
        n = 5
        edges = [(1,2), (3,1), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
        s = [e[1] for e in edges]
        t = [e[2] for e in edges]
        g = GNNGraph(s, t)
        idx, idxmax = GNNGraphs.edge_encoding(s, t, n, directed=false, self_loops=false)
        @test idxmax == n * (n - 1) ÷ 2
        @test idx == 1:idxmax
        
        snew, tnew = GNNGraphs.edge_decoding(idx, n, directed=false, self_loops=false)
        @test snew == [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
        @test tnew == [2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    end

    @testset "directed=false, self_loops=false" begin
        n = 5
        edges = [(1,2), (3,1), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
        s = [e[1] for e in edges]
        t = [e[2] for e in edges]

        idx, idxmax = GNNGraphs.edge_encoding(s, t, n, directed=false, self_loops=false)
        @test idxmax == n * (n - 1) ÷ 2
        @test idx == 1:idxmax
        
        snew, tnew = GNNGraphs.edge_decoding(idx, n, directed=false, self_loops=false)
        @test snew == [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
        @test tnew == [2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
    end

    @testset "directed=true, self_loops=false" begin
        n = 5
        edges = [(1,2), (3,1), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
        s = [e[1] for e in edges]
        t = [e[2] for e in edges]

        idx, idxmax = GNNGraphs.edge_encoding(s, t, n, directed=true, self_loops=false)
        @test idxmax == n^2 - n
        @test idx == [1, 9, 3, 4, 6, 7, 8, 11, 12, 16]
        snew, tnew = GNNGraphs.edge_decoding(idx, n, directed=true, self_loops=false)
        @test snew == s
        @test tnew == t
    end
end

@testitem "color_refinement" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using StableRNGs
    for GRAPH_T in GRAPH_TYPES
        rng = StableRNG(17)
        g = rand_graph(rng, 10, 20, graph_type = GRAPH_T)
        x0 = ones(Int, 10)
        x, ncolors, niters = color_refinement(g, x0)
        @test ncolors == 9
        @test niters == 2
        @test x == [6, 7, 8, 9, 10, 11, 12, 6, 13, 14]
        
        x2, _, _ = color_refinement(g)
        @test x2 == x
    end
end
