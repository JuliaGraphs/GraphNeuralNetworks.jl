# TODO test that the graph type is preserved
# when constructing a GNNGraph from another

@testitem "Constructor: adjacency matrix" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        A = sprand(10, 10, 0.5)
        sA, tA, vA = findnz(A)

        g = GNNGraph(A, graph_type = GRAPH_T)
        s, t = edge_index(g)
        v = get_edge_weight(g)
        @test s == sA
        @test t == tA
        @test v == vA

        g = GNNGraph(Matrix(A), graph_type = GRAPH_T)
        s, t = edge_index(g)
        v = get_edge_weight(g)
        @test s == sA
        @test t == tA
        @test v == vA

        g = GNNGraph([0 0 0
                        0 0 1
                        0 1 0], graph_type = GRAPH_T)
        @test g.num_nodes == 3
        @test g.num_edges == 2

        g = GNNGraph([0 1 0
                        1 0 0
                        0 0 0], graph_type = GRAPH_T)
        @test g.num_nodes == 3
        @test g.num_edges == 2
    end
end

@testitem "Constructor: integer" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = GNNGraph(10, graph_type = GRAPH_T)
        @test g.num_nodes == 10
        @test g.num_edges == 0

        g2 = rand_graph(10, 30, graph_type = GRAPH_T)
        G = typeof(g2)
        g = G(10)
        @test g.num_nodes == 10
        @test g.num_edges == 0

        g = GNNGraph(graph_type = GRAPH_T)
        @test g.num_nodes == 0
    end
end

@testitem "symmetric graph" setup=[GraphsTestModule] tags=[:gpu] begin
    using .GraphsTestModule
    dev = gpu_device()
    TEST_GPU = dev != cpu_device()
    for GRAPH_T in GRAPH_TYPES
        s = [1, 1, 2, 2, 3, 3, 4, 4]
        t = [2, 4, 1, 3, 2, 4, 1, 3]
        adj_mat = [0 1 0 1
                    1 0 1 0
                    0 1 0 1
                    1 0 1 0]
        adj_list_out = [[2, 4], [1, 3], [2, 4], [1, 3]]
        adj_list_in = [[2, 4], [1, 3], [2, 4], [1, 3]]

        # core functionality
        g = GNNGraph(s, t; graph_type = GRAPH_T)
        if TEST_GPU
            g_gpu = g |> dev
        end

        @test g.num_edges == 8
        @test g.num_nodes == 4
        @test nv(g) == g.num_nodes
        @test ne(g) == g.num_edges
        @test Tuple.(collect(edges(g))) |> sort == collect(zip(s, t)) |> sort
        @test sort(outneighbors(g, 1)) == [2, 4]
        @test sort(inneighbors(g, 1)) == [2, 4]
        @test is_directed(g) == true
        s1, t1 = sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t
        @test vertices(g) == 1:(g.num_nodes)

        @test sort.(adjacency_list(g; dir = :in)) == adj_list_in
        @test sort.(adjacency_list(g; dir = :out)) == adj_list_out

        @testset "adjacency_matrix" begin
            @test adjacency_matrix(g) == adj_mat
            @test adjacency_matrix(g; dir = :in) == adj_mat
            @test adjacency_matrix(g; dir = :out) == adj_mat

            if TEST_GPU && !(dev isa MetalDevice) && GRAPH_T != :sparse
                # See https://github.com/JuliaGPU/CUDA.jl/pull/1093
                mat_gpu = adjacency_matrix(g_gpu)
                @test mat_gpu isa AbstractMatrix{Int}
                @test get_device(mat_gpu) isa AbstractGPUDevice
                # Convert to float first because poor Int support in CUSPARSE, throws an error
                @test Array(Float32.(mat_gpu)) == Float32.(adj_mat)
            end
        end

        @testset "normalized_laplacian" begin
            mat = normalized_laplacian(g)
            if TEST_GPU && !(dev isa MetalDevice) && GRAPH_T != :sparse && GRAPH_T != :coo
                mat_gpu = normalized_laplacian(g_gpu)
                @test mat_gpu isa AbstractMatrix{Float32}
                @test get_device(mat_gpu)isa AbstractGPUDevice
                @test Array(mat_gpu) == mat
            end
        end

        @testset "scaled_laplacian" begin 
            if TEST_GPU && !(dev isa MetalDevice) && GRAPH_T != :sparse && GRAPH_T != :coo
                mat = scaled_laplacian(g)
                mat_gpu = scaled_laplacian(g_gpu)
                @test mat_gpu isa AbstractMatrix{Float32}
                @test get_device(mat_gpu) isa AbstractGPUDevice
                @test Array(mat_gpu) ≈ mat
            end 
        end

        @testset "constructors" begin
            adjacency_matrix(g; dir = :out) == adj_mat
            adjacency_matrix(g; dir = :in) == adj_mat
        end

        if TEST_GPU
            @testset "functor" begin
                s_cpu, t_cpu = edge_index(g)
                s_gpu, t_gpu = edge_index(g_gpu)
                @test s_gpu isa AbstractVector{<:Integer}
                @test get_device(s_gpu) isa AbstractGPUDevice
                @test Array(s_gpu) == s_cpu
                @test t_gpu isa AbstractVector{<:Integer}
                @test get_device(t_gpu) isa AbstractGPUDevice
                @test Array(t_gpu) == t_cpu
            end
        end
    end
end

@testitem "asymmetric graph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    s = [1, 2, 3, 4]
    t = [2, 3, 4, 1]
    adj_mat_out = [0 1 0 0
                    0 0 1 0
                    0 0 0 1
                    1 0 0 0]
    adj_list_out = [[2], [3], [4], [1]]

    adj_mat_in = [0 0 0 1
                    1 0 0 0
                    0 1 0 0
                    0 0 1 0]
    adj_list_in = [[4], [1], [2], [3]]

    for GRAPH_T in GRAPH_TYPES
        g = GNNGraph(s, t; graph_type = GRAPH_T)

        @test g.num_edges == 4
        @test g.num_nodes == 4
        @test length(edges(g)) == 4
        @test sort(outneighbors(g, 1)) == [2]
        @test sort(inneighbors(g, 1)) == [4]
        @test is_directed(g) == true
        @test is_directed(typeof(g)) == true
        s1, t1 = sort_edge_index(edge_index(g))
        @test s1 == s
        @test t1 == t

        # adjacency
        @test adjacency_matrix(g) == adj_mat_out
        @test adjacency_list(g) == adj_list_out
        @test adjacency_matrix(g, dir = :out) == adj_mat_out
        @test adjacency_list(g, dir = :out) == adj_list_out
        @test adjacency_matrix(g, dir = :in) == adj_mat_in
        @test adjacency_list(g, dir = :in) == adj_list_in
    end
end

@testitem "zero" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(4, 6, graph_type = GRAPH_T)
        G = typeof(g)
        @test zero(G) == G(0)
    end
end

@testitem "Graphs.jl constructor" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        lg = random_regular_graph(10, 4)
        @test !Graphs.is_directed(lg)
        g = GNNGraph(lg, graph_type = GRAPH_T)
        @test g.num_edges == 2 * ne(lg) # g in undirected
        @test Graphs.is_directed(g)
        for e in Graphs.edges(lg)
            i, j = src(e), dst(e)
            @test has_edge(g, i, j)
            @test has_edge(g, j, i)
        end

        @testset "SimpleGraph{Int32}" begin
            g = GNNGraph(SimpleGraph{Int32}(6), graph_type = GRAPH_T)
            @test g.num_nodes == 6
        end
    end
end

@testitem "Constructor: empty" setup=[GraphsTestModule] begin
    g = GNNGraph(ndata=ones(2, 1))
    @test g.num_nodes == 1
    @test g.num_edges == 0
    @test g.ndata.x == ones(2, 1)

    g = GNNGraph(num_nodes=1)
    @test g.num_nodes == 1
    @test g.num_edges == 0
    @test isempty(g.ndata) 

    g = GNNGraph((Int[], Int[]); ndata=(; a=[1]))
    @test g.num_nodes == 1
    @test g.num_edges == 0
    @test g.ndata.a == [1]

    g = GNNGraph((Int[], Int[]); ndata=(;a=[1]), edata=(;b=Int[]), num_nodes=1)
    @test g.num_nodes == 1
    @test g.num_edges == 0
    @test g.ndata.a == [1]
    @test g.edata.b == Int[]

    g = GNNGraph(; edata=(;b=Int[]))
    @test g.num_nodes == 0
    @test g.num_edges == 0
    @test g.edata.b == Int[]
end


@testitem "Features" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = GNNGraph(sprand(10, 10, 0.3), graph_type = GRAPH_T)

        # default names
        X = rand(10, g.num_nodes)
        E = rand(10, g.num_edges)
        U = rand(10, g.num_graphs)

        g = GNNGraph(g, ndata = X, edata = E, gdata = U)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U
        @test g.x === g.ndata.x
        @test g.e === g.edata.e
        @test g.u === g.gdata.u

        # Check no args
        g = GNNGraph(g)
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U

        # multiple features names
        g = GNNGraph(g, ndata = (x2 = 2X, g.ndata...), edata = (e2 = 2E, g.edata...),
                        gdata = (u2 = 2U, g.gdata...))
        @test g.ndata.x === X
        @test g.edata.e === E
        @test g.gdata.u === U
        @test g.ndata.x2 ≈ 2X
        @test g.edata.e2 ≈ 2E
        @test g.gdata.u2 ≈ 2U
        @test g.x === g.ndata.x
        @test g.e === g.edata.e
        @test g.u === g.gdata.u
        @test g.x2 === g.ndata.x2
        @test g.e2 === g.edata.e2
        @test g.u2 === g.gdata.u2

        # Dimension checks
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata = rand(29),
                                                graph_type = GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30), edata = rand(2, 29),
                                                graph_type = GRAPH_T)
        @test_throws AssertionError GNNGraph(erdos_renyi(10, 30),
                                                edata = (; x = rand(30), y = rand(29)),
                                                graph_type = GRAPH_T)

        # Copy features on reverse edge
        e = rand(30)
        g = GNNGraph(erdos_renyi(10, 30), edata = e, graph_type = GRAPH_T)
        @test g.edata.e == [e; e]

        # non-array global
        g = rand_graph(10, 30, gdata = "ciao", graph_type = GRAPH_T)
        @test g.gdata.u == "ciao"

        # vectors stays vectors
        g = rand_graph(10, 30, ndata = rand(10),
                        edata = rand(30),
                        gdata = (u = rand(2), z = rand(1), q = 1),
                        graph_type = GRAPH_T)
        @test size(g.ndata.x) == (10,)
        @test size(g.edata.e) == (30,)
        @test size(g.gdata.u) == (2, 1)
        @test size(g.gdata.z) == (1,)
        @test g.gdata.q === 1

        # Error for non-array ndata
        @test_throws AssertionError rand_graph(10, 30, ndata = "ciao", graph_type = GRAPH_T)
        @test_throws AssertionError rand_graph(10, 30, ndata = 1, graph_type = GRAPH_T)

        # Error for Ambiguous getproperty
        g = rand_graph(10, 20, ndata = rand(2, 10), edata = (; x = rand(3, 20)),
                        graph_type = GRAPH_T)
        @test size(g.ndata.x) == (2, 10)
        @test size(g.edata.x) == (3, 20)
        @test_throws ArgumentError g.x
    end
end

@testitem "MLUtils and DataLoader compat" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        n, m, num_graphs = 10, 30, 50
        X = rand(10, n)
        E = rand(10, m)
        U = rand(10, 1)
        data = [rand_graph(n, m, ndata = X, edata = E, gdata = U, graph_type = GRAPH_T)
                for _ in 1:num_graphs]
        g = MLUtils.batch(data)

        @testset "batch then pass to dataloader" begin
            @test MLUtils.getobs(g, 3) == getgraph(g, 3)
            @test MLUtils.getobs(g, 3:5) == getgraph(g, 3:5)
            @test MLUtils.numobs(g) == g.num_graphs

            d = MLUtils.DataLoader(g, batchsize = 2, shuffle = false)
            @test first(d) == getgraph(g, 1:2)
        end

        @testset "pass to dataloader and no automatic collation" begin
            @test MLUtils.getobs(data, 3) == data[3]
            @test MLUtils.getobs(data, 3:5) isa Vector{<:GNNGraph}
            @test MLUtils.getobs(data, 3:5) == [data[3], data[4], data[5]]
            @test MLUtils.numobs(data) == g.num_graphs

            d = MLUtils.DataLoader(data, batchsize = 2, shuffle = false)
            @test first(d) == [data[1], data[2]]
        end
    end
end

@testitem "Graphs.jl integration" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = GNNGraph(erdos_renyi(10, 20), graph_type = GRAPH_T)
        @test g isa Graphs.AbstractGraph
    end
end

@testitem "==" setup = [GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g1 = rand_graph(5, 6, ndata = rand(5), edata = rand(6), graph_type = GRAPH_T)
        @test g1 == g1
        @test g1 == deepcopy(g1)
        @test g1 !== deepcopy(g1)

        g2 = GNNGraph(g1, graph_type = GRAPH_T)
        @test g1 == g2
        @test g1 === g2 # this is true since GNNGraph is immutable

        g2 = GNNGraph(g1, ndata = rand(5), graph_type = GRAPH_T)
        @test g1 != g2
        @test g1 !== g2

        g2 = GNNGraph(g1, edata = rand(6), graph_type = GRAPH_T)
        @test g1 != g2
        @test g1 !== g2
    end
end

@testitem "hash" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g1 = rand_graph(5, 6, ndata = rand(5), edata = rand(6), graph_type = GRAPH_T)
        @test hash(g1) == hash(g1)
        @test hash(g1) == hash(deepcopy(g1))
        @test hash(g1) == hash(GNNGraph(g1, ndata = g1.ndata, graph_type = GRAPH_T))
        @test hash(g1) == hash(GNNGraph(g1, ndata = g1.ndata, graph_type = GRAPH_T))
        @test hash(g1) != hash(GNNGraph(g1, ndata = rand(5), graph_type = GRAPH_T))
        @test hash(g1) != hash(GNNGraph(g1, edata = rand(6), graph_type = GRAPH_T))
    end
end

@testitem "copy" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g1 = rand_graph(10, 4, ndata = rand(2, 10), graph_type = GRAPH_T)
        g2 = copy(g1)
        @test g1 === g2 # shallow copies are identical for immutable objects

        g2 = copy(g1, deep = true)
        @test g1 == g2
        @test g1 !== g2
    end
end

@testitem "show" begin
    # no throw when global data is not an array
    g = rand_graph(10, 20, gdata = "ciao")
    @test sprint(show, g) == "GNNGraph(10, 20) with u: string(4) data"
    @test sprint(show, MIME("text/plain"), g) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  gdata:\n    u = 4-codeunit String"
end

## Cannot test this because DataStore is not an ordered collection
## Uncomment when/if it will be based on OrderedDict
# @testset "show" begin
#     @test sprint(show, rand_graph(10, 20)) == "GNNGraph(10, 20) with no data"
#     @test sprint(show, rand_graph(10, 20, ndata=rand(5, 10))) == "GNNGraph(10, 20) with x: 5×10 data"
#     @test sprint(show, rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20), gdata=(q=rand(1, 1), p=rand(3, 1)))) == "GNNGraph(10, 20) with (a: 5×10, b: 3×10), e: 2×20, (q: 1×1, p: 3×1) data"
#     @test sprint(show, rand_graph(10, 20, ndata=(a=rand(5, 10),))) == "GNNGraph(10, 20) with a: 5×10 data"
#     @test sprint(show, rand_graph(10, 20, ndata=rand(5, 10), edata=rand(2, 20))) == "GNNGraph(10, 20) with x: 5×10, e: 2×20 data"
#     @test sprint(show, rand_graph(10, 20, ndata=rand(5, 10), gdata=rand(1, 1))) == "GNNGraph(10, 20) with x: 5×10, u: 1×1 data"
#     @test sprint(show, rand_graph(10, 20, ndata=rand(5, 10), edata=(e=rand(2, 20), f=rand(2, 20), h=rand(3, 20)), gdata=rand(1, 1))) == "GNNGraph(10, 20) with x: 5×10, (e: 2×20, f: 2×20, h: 3×20), u: 1×1 data"
#     @test sprint(show, rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20))) == "GNNGraph(10, 20) with (a: 5×10, b: 3×10), e: 2×20 data"
#     @test sprint(show, rand_graph(10, 20, ndata=(a=rand(5,5, 10), b=rand(3,2, 10)), edata=rand(2, 20))) == "GNNGraph(10, 20) with (a: 5×5×10, b: 3×2×10), e: 2×20 data"
# end

# @testset "show plain/text compact true" begin
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20); context=:compact => true) == "GNNGraph(10, 20) with no data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10)); context=:compact => true) == "GNNGraph(10, 20) with x: 5×10 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20), gdata=(q=rand(1, 1), p=rand(3, 1))); context=:compact => true) == "GNNGraph(10, 20) with (a: 5×10, b: 3×10), e: 2×20, (q: 1×1, p: 3×1) data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10),)); context=:compact => true) == "GNNGraph(10, 20) with a: 5×10 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), edata=rand(2, 20)); context=:compact => true) == "GNNGraph(10, 20) with x: 5×10, e: 2×20 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), gdata=rand(1, 1)); context=:compact => true) == "GNNGraph(10, 20) with x: 5×10, u: 1×1 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), edata=(e=rand(2, 20), f=rand(2, 20), h=rand(3, 20)), gdata=rand(1, 1)); context=:compact => true) == "GNNGraph(10, 20) with x: 5×10, (e: 2×20, f: 2×20, h: 3×20), u: 1×1 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20)); context=:compact => true) == "GNNGraph(10, 20) with (a: 5×10, b: 3×10), e: 2×20 data"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5,5, 10), b=rand(3,2, 10)), edata=rand(2, 20)); context=:compact => true) == "GNNGraph(10, 20) with (a: 5×5×10, b: 3×2×10), e: 2×20 data"
# end

# @testset "show plain/text compact false" begin
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\tx = 5×10 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20), gdata=(q=rand(1, 1), p=rand(3, 1))); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\ta = 5×10 Matrix{Float64}\n\tb = 3×10 Matrix{Float64}\n  edata:\n\te = 2×20 Matrix{Float64}\n  gdata:\n\tq = 1×1 Matrix{Float64}\n\tp = 3×1 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10),)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\ta = 5×10 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), edata=rand(2, 20)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\tx = 5×10 Matrix{Float64}\n  edata:\n\te = 2×20 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), gdata=rand(1, 1)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\tx = 5×10 Matrix{Float64}\n  gdata:\n\tu = 1×1 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=rand(5, 10), edata=(e=rand(2, 20), f=rand(2, 20), h=rand(3, 20)), gdata=rand(1, 1)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\tx = 5×10 Matrix{Float64}\n  edata:\n\te = 2×20 Matrix{Float64}\n\tf = 2×20 Matrix{Float64}\n\th = 3×20 Matrix{Float64}\n  gdata:\n\tu = 1×1 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 10), b=rand(3, 10)), edata=rand(2, 20)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\ta = 5×10 Matrix{Float64}\n\tb = 3×10 Matrix{Float64}\n  edata:\n\te = 2×20 Matrix{Float64}"
#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20, ndata=(a=rand(5, 5, 10), b=rand(3, 2, 10)), edata=rand(2, 20)); context=:compact => false) == "GNNGraph:\n  num_nodes: 10\n  num_edges: 20\n  ndata:\n\ta = 5×5×10 Array{Float64, 3}\n\tb = 3×2×10 Array{Float64, 3}\n  edata:\n\te = 2×20 Matrix{Float64}"
# end
