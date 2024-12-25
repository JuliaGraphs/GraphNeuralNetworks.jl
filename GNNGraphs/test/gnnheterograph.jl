

@testset "Empty constructor" begin
    g = GNNHeteroGraph()
    @test isempty(g.num_nodes)
    g = add_edges(g, (:user, :like, :actor) => ([1,2,3,3,3], [3,5,1,9,4]))
    @test g.num_nodes[:user] == 3
    @test g.num_nodes[:actor] == 9
    @test g.num_edges[(:user, :like, :actor)] == 5
end

@testset "Constructor from pairs" begin
    hg = GNNHeteroGraph((:A, :e1, :B) => ([1,2,3,4], [3,2,1,5]))
    @test hg.num_nodes == Dict(:A => 4, :B => 5)
    @test hg.num_edges == Dict((:A, :e1, :B) => 4)

    hg = GNNHeteroGraph((:A, :e1, :B) => ([1,2,3], [3,2,1]),
                        (:A, :e2, :C) => ([1,2,3], [4,5,6]))
    @test hg.num_nodes == Dict(:A => 3, :B => 3, :C => 6)
    @test hg.num_edges == Dict((:A, :e1, :B) => 3, (:A, :e2, :C) => 3)
end

@testset "Generation" begin
    hg = rand_heterograph(Dict(:A => 10, :B => 20),
                            Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10))

    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10)
    @test hg.graph_indicator === nothing
    @test hg.num_graphs == 1
    @test hg.ndata isa Dict{Symbol, DataStore}
    @test hg.edata isa Dict{Tuple{Symbol, Symbol, Symbol}, DataStore}
    @test isempty(hg.gdata)
    @test hg.gdata._n == -1 # no constraints on gdata
    @test sort(hg.ntypes) == [:A, :B]
    @test sort(hg.etypes) == [(:A, :rel1, :B), (:B, :rel2, :A)]
end

@testset "features" begin
    hg = rand_heterograph(Dict(:A => 10, :B => 20),
                            Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
                            ndata = Dict(:A => rand(2, 10),
                                        :B => (x = rand(3, 20), y = rand(4, 20))),
                            edata = Dict((:A, :rel1, :B) => rand(5, 30)),
                            gdata = 1)

    @test size(hg.ndata[:A].x) == (2, 10)
    @test size(hg.ndata[:B].x) == (3, 20)
    @test size(hg.ndata[:B].y) == (4, 20)
    @test size(hg.edata[(:A, :rel1, :B)].e) == (5, 30)
    @test hg.gdata == DataStore(u = 1)

end

@testset "indexing syntax" begin
    g = GNNHeteroGraph((:user, :rate, :movie) => ([1,1,2,3], [7,13,5,7]))
    g[:movie].z = rand(Float32, 64, 13);
    g[:user, :rate, :movie].e = rand(Float32, 64, 4);
    g[:user].x = rand(Float32, 64, 3);
    @test size(g.ndata[:user].x) == (64, 3)
    @test size(g.ndata[:movie].z) == (64, 13)
    @test size(g.edata[(:user, :rate, :movie)].e) == (64, 4) 
end


@testset "simplified constructor" begin
    hg = rand_heterograph((:A => 10, :B => 20),
                            ((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
                            ndata = (:A => rand(2, 10),
                                    :B => (x = rand(3, 20), y = rand(4, 20))),
                            edata = (:A, :rel1, :B) => rand(5, 30),
                            gdata = 1)

    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10)
    @test hg.graph_indicator === nothing
    @test hg.num_graphs == 1
    @test size(hg.ndata[:A].x) == (2, 10)
    @test size(hg.ndata[:B].x) == (3, 20)
    @test size(hg.ndata[:B].y) == (4, 20)
    @test size(hg.edata[(:A, :rel1, :B)].e) == (5, 30)
    @test hg.gdata == DataStore(u = 1)

    nA, nB = 10, 20
    edges1 = rand(1:nA, 20), rand(1:nB, 20)
    edges2 = rand(1:nB, 30), rand(1:nA, 30)
    hg = GNNHeteroGraph(((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2))
    @test hg.num_edges == Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)

    nA, nB = 10, 20
    edges1 = rand(1:nA, 20), rand(1:nB, 20)
    edges2 = rand(1:nB, 30), rand(1:nA, 30)
    hg = GNNHeteroGraph(((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2);
                        num_nodes = (:A => nA, :B => nB))
    @test hg.num_nodes == Dict(:A => 10, :B => 20)
    @test hg.num_edges == Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)
end

@testset "num_edge_types / num_node_types" begin
    hg = rand_heterograph((:A => 10, :B => 20),
            ((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
            ndata = (:A => rand(2, 10),
                    :B => (x = rand(3, 20), y = rand(4, 20))),
            edata = (:A, :rel1, :B) => rand(5, 30),
            gdata = 1)
    @test num_edge_types(hg) == 2
    @test num_node_types(hg) == 2

    g = rand_graph(10, 20)
    @test num_edge_types(g) == 1
    @test num_node_types(g) == 1
end

@testset "numobs" begin
    hg = rand_heterograph((:A => 10, :B => 20),
            ((:A, :rel1, :B) => 30, (:B, :rel2, :A) => 10),
            ndata = (:A => rand(2, 10),
                    :B => (x = rand(3, 20), y = rand(4, 20))),
            edata = (:A, :rel1, :B) => rand(5, 30),
            gdata = 1)
    @test MLUtils.numobs(hg) == 1
end

@testset "get/set node features" begin
    d, n = 3, 5
    g = rand_bipartite_heterograph((n, 2*n), 15)
    g[:A].x = rand(Float32, d, n)
    g[:B].y = rand(Float32, d, 2*n)

    @test size(g[:A].x) == (d, n)
    @test size(g[:B].y) == (d, 2*n)
end

@testset "add_edges" begin
    d, n = 3, 5
    g = rand_bipartite_heterograph((n, 2 * n), 15)
    s, t = [1, 2, 3], [3, 2, 1]
    ## Keep the same ntypes - construct with args
    g1 = add_edges(g, (:A, :rel1, :B), s, t)
    @test num_node_types(g1) == 2
    @test num_edge_types(g1) == 3
    for i in eachindex(s, t)
        @test has_edge(g1, (:A, :rel1, :B), s[i], t[i])
    end
    # no change to num_nodes
    @test g1.num_nodes[:A] == n
    @test g1.num_nodes[:B] == 2n

    ## Keep the same ntypes - construct with a pair
    g2 = add_edges(g, (:A, :rel1, :B) => (s, t))
    @test num_node_types(g2) == 2
    @test num_edge_types(g2) == 3
    for i in eachindex(s, t)
        @test has_edge(g2, (:A, :rel1, :B), s[i], t[i])
    end
    # no change to num_nodes
    @test g2.num_nodes[:A] == n
    @test g2.num_nodes[:B] == 2n

    ## New ntype with num_nodes (applies only to the new ntype) and edata
    edata = rand(Float32, d, length(s))
    g3 = add_edges(g,
        (:A, :rel1, :C) => (s, t);
        num_nodes = Dict(:A => 1, :B => 1, :C => 10),
        edata)
    @test num_node_types(g3) == 3
    @test num_edge_types(g3) == 3
    for i in eachindex(s, t)
        @test has_edge(g3, (:A, :rel1, :C), s[i], t[i])
    end
    # added edata
    @test g3.edata[(:A, :rel1, :C)].e == edata
    # no change to existing num_nodes
    @test g3.num_nodes[:A] == n
    @test g3.num_nodes[:B] == 2n
    # new num_nodes added as per kwarg
    @test g3.num_nodes[:C] == 10
end

@testset "add self loops" begin
    g1 = GNNHeteroGraph((:A, :to, :B) => ([1,2,3,4], [3,2,1,5]))
    g2 = add_self_loops(g1, (:A, :to, :B))
    @test g2.num_edges[(:A, :to, :B)] === g1.num_edges[(:A, :to, :B)]
    g1 = GNNHeteroGraph((:A, :to, :A) => ([1,2,3,4], [3,2,1,5]))
    g2 = add_self_loops(g1, (:A, :to, :A))
    @test g2.num_edges[(:A, :to, :A)] === g1.num_edges[(:A, :to, :A)] + g1.num_nodes[(:A)]
end

## Cannot test this because DataStore is not an ordered collection
## Uncomment when/if it will be based on OrderedDict
# @testset "show" begin
#     num_nodes = Dict(:A => 10, :B => 20);
#     edges1 = rand(1:num_nodes[:A], 20), rand(1:num_nodes[:B], 20)
#     edges2 = rand(1:num_nodes[:B], 30), rand(1:num_nodes[:A], 30)
#     eindex = ((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2)
#     ndata = Dict(:A => (x = rand(2, num_nodes[:A]), y = rand(3, num_nodes[:A])),:B => rand(10, num_nodes[:B]))
#     edata= Dict((:A, :rel1, :B) => (x = rand(2, 20), y = rand(3, 20)),(:B, :rel2, :A) => rand(10, 30))
#     hg1 = GNNHeteroGraph(eindex; num_nodes)
#     hg2 = GNNHeteroGraph(eindex; num_nodes, ndata,edata)
#     hg3 = GNNHeteroGraph(eindex; num_nodes, ndata)
#     @test sprint(show, hg1) == "GNNHeteroGraph(Dict(:A => 10, :B => 20), Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30))"
#     @test sprint(show, hg2) == sprint(show, hg1)
#     @test sprint(show, MIME("text/plain"), hg1; context=:compact => true) == "GNNHeteroGraph(Dict(:A => 10, :B => 20), Dict((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30))"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => true) == sprint(show, MIME("text/plain"), hg1;context=:compact => true)
#     @test sprint(show, MIME("text/plain"), hg1; context=:compact => false) == "GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => false) == "GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)\n ndata:\n\t:A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})\n\t:B  =>  x = 10×20 Matrix{Float64}\n edata:\n\t(:A, :rel1, :B)  =>  (x = 2×20 Matrix{Float64}, y = 3×20 Matrix{Float64})\n\t(:B, :rel2, :A)  =>  e = 10×30 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), hg3; context=:compact => false) =="GNNHeteroGraph:\n num_nodes: (:A => 10, :B => 20)\n num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)\n ndata:\n\t:A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})\n\t:B  =>  x = 10×20 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), hg2; context=:compact => false) != sprint(show, MIME("text/plain"), hg3; context=:compact => false)
# end
