@testitem "is_bidirected" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 20, bidirected = true, graph_type = GRAPH_T)
        @test is_bidirected(g)

        g = rand_graph(10, 20, bidirected = false, graph_type = GRAPH_T)
        @test !is_bidirected(g)
    end
end

@testitem "has_multi_edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        if GRAPH_T == :coo
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            g = GNNGraph(s, t, graph_type = GRAPH_T)
            @test has_multi_edges(g)

            s = [1, 2, 2, 3]
            t = [2, 1, 2, 4]
            g = GNNGraph(s, t, graph_type = GRAPH_T)
            @test !has_multi_edges(g)
        end 
    end
end

@testitem "edges" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(4, 10, graph_type = GRAPH_T)
        @test edgetype(g) <: Graphs.Edge
        for e in edges(g)
            @test e isa Graphs.Edge
        end
    end
end

@testitem "has_isolated_nodes" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        s = [1, 2, 3]
        t = [2, 3, 2]
        g = GNNGraph(s, t, graph_type = GRAPH_T)
        @test has_isolated_nodes(g) == false
        @test has_isolated_nodes(g, dir = :in) == true
    end
end

@testitem "has_self_loops" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        s = [1, 1, 2, 3]
        t = [2, 2, 2, 4]
        g = GNNGraph(s, t, graph_type = GRAPH_T)
        @test has_self_loops(g)

        s = [1, 1, 2, 3]
        t = [2, 2, 3, 4]
        g = GNNGraph(s, t, graph_type = GRAPH_T)
        @test !has_self_loops(g)
    end
end

@testitem "degree" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using SparseArrays: AbstractSparseMatrix

    for GRAPH_T in GRAPH_TYPES
        @testset "unweighted" begin
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            g = GNNGraph(s, t, graph_type = GRAPH_T)

            @test degree(g) isa Vector{Int}
            @test degree(g) == degree(g; dir = :out) == [2, 1, 1, 0] # default is outdegree
            @test degree(g; dir = :in) == [0, 3, 0, 1]
            @test degree(g; dir = :both) == [2, 4, 1, 1]
            @test eltype(degree(g, Float32)) == Float32
        end

        @testset "weighted" begin
            # weighted degree
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            eweight = Float32[0.1, 2.1, 1.2, 1]
            g = GNNGraph((s, t, eweight), graph_type = GRAPH_T)
            @test degree(g) ≈ [2.2, 1.2, 1.0, 0.0]
            d = degree(g, edge_weight = false)
            if GRAPH_T == :coo
                @test d == [2, 1, 1, 0]
            else
                # Adjacency matrix representation cannot disambiguate multiple edges
                # and edge weights
                @test d == [1, 1, 1, 0]
            end
            @test eltype(d) <: Integer
            @test degree(g, edge_weight = 2 * eweight) ≈ [4.4, 2.4, 2.0, 0.0] broken = (GRAPH_T != :coo)
            
            @testset "gradient" begin
                gw = gradient(eweight) do w
                    g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                    sum(degree(g, edge_weight = false))
                end[1]

                @test gw === nothing

                gw = gradient(eweight) do w
                    g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                    sum(degree(g, edge_weight = true))
                end[1]

                @test gw isa AbstractVector{Float32}
                @test gw isa Vector{Float32} broken = (GRAPH_T == :sparse)
                @test gw ≈ ones(Float32, length(gw))            

                gw = gradient(eweight) do w
                    g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                    sum(degree(g, dir=:both, edge_weight=true))
                end[1]

                @test gw isa AbstractVector{Float32}
                @test gw isa Vector{Float32} broken = (GRAPH_T == :sparse)
                @test gw ≈ 2 * ones(Float32, length(gw))

                grad = gradient(g) do g
                    sum(degree(g, edge_weight=false))
                end[1]
                @test grad === nothing
                
                grad = gradient(g) do g
                    sum(degree(g, edge_weight=true))
                end[1]
                
                if GRAPH_T == :coo
                    @test grad.graph[3] isa Vector{Float32}
                    @test grad.graph[3] ≈ ones(Float32, length(gw))
                else
                    if GRAPH_T == :sparse
                        @test grad.graph isa AbstractSparseMatrix{Float32}
                    end
                    @test grad.graph isa AbstractMatrix{Float32}
                    
                    @test grad.graph ≈ [0.0  1.0  0.0  0.0
                                        0.0  1.0  0.0  0.0
                                        0.0  0.0  0.0  1.0
                                        0.0  0.0  0.0  0.0]
                end

                @testset "directed, degree dir=$dir" for dir in [:in, :out, :both]
                    g = rand_graph(10, 30, bidirected=false)
                    w = rand(Float32, 30)
                    s, t = edge_index(g)
                    
                    grad = gradient(w) do w
                        g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                        sum(tanh.(degree(g; dir, edge_weight=true)))
                    end[1]

                    ngrad = ngradient(w) do w
                        g = GNNGraph((s, t, w), graph_type = GRAPH_T)
                        sum(tanh.(degree(g; dir, edge_weight=true)))
                    end[1]

                    @test grad ≈ ngrad
                end
            end
        end
    end
end


@testitem "degree GPU" setup=[GraphsTestModule] tags=[:gpu] begin
    using .GraphsTestModule
    dev = gpu_device(force=true)
    for GRAPH_T in GRAPH_TYPES
        dev isa MetalDevice && continue # TODO: remove when gather/scatter is implemented for Metal
        @testset "unweighted" begin
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            g = GNNGraph(s, t, graph_type = GRAPH_T)

            g_gpu = g |> dev
            d = degree(g)
            d_gpu = degree(g_gpu)
            @test d_gpu isa AbstractVector{Int}
            @test get_device(d_gpu) isa AbstractGPUDevice
            @test Array(d_gpu) == d
        end

        @testset "weighted" begin
            # weighted degree
            s = [1, 1, 2, 3]
            t = [2, 2, 2, 4]
            eweight = Float32[0.1, 2.1, 1.2, 1]
            g = GNNGraph((s, t, eweight), graph_type = GRAPH_T)  
            g_gpu = g |> dev
            d = degree(g)
            d_gpu = degree(g_gpu)
            @test d_gpu isa AbstractArray{Float32}
            @test get_device(d_gpu) isa AbstractGPUDevice
            @test Array(d_gpu) ≈ d
        end
    end
end

@testitem "heterognn, degree" begin
    g = GNNHeteroGraph((:A, :to, :B) => ([1,1,2,3], [7,13,5,7]))
    @test degree(g, (:A, :to, :B), dir = :out) == [2, 1, 1]
    @test degree(g, (:A, :to, :B), dir = :in) == [0,  0,  0,  0,  1,  0,  2,  0,  0,  0,  0,  0,  1]
    @test degree(g, (:A, :to, :B)) == [2, 1, 1]
end

@testitem "laplacian_matrix" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 30, graph_type = GRAPH_T)
        A = adjacency_matrix(g)
        D = Diagonal(vec(sum(A, dims = 2)))
        L = laplacian_matrix(g)
        @test eltype(L) == eltype(g)
        @test L ≈ D - A
    end
end

@testitem "laplacian_lambda_max" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    s = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    t = [2, 3, 4, 5, 1, 5, 1, 2, 3, 4]
    g = GNNGraph(s, t)
    @test laplacian_lambda_max(g) ≈ Float32(1.809017)
    data1 = [g for i in 1:5]
    gall1 = MLUtils.batch(data1)
    @test laplacian_lambda_max(gall1) ≈ [Float32(1.809017) for i in 1:5]
    data2 = [rand_graph(10, 20) for i in 1:3]
    gall2 = MLUtils.batch(data2)
    @test length(laplacian_lambda_max(gall2, add_self_loops=true)) == 3
end

@testitem "adjacency_matrix" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        a = sprand(5, 5, 0.5)
        abin = map(x -> x > 0 ? 1 : 0, a)

        g = GNNGraph(a, graph_type = GRAPH_T)
        A = adjacency_matrix(g, Float32)
        @test A ≈ a
        @test eltype(A) == Float32

        Abin = adjacency_matrix(g, Float32, weighted = false)
        @test Abin ≈ abin
        @test eltype(Abin) == Float32

        @testset "gradient" begin
            s = [1, 2, 3]
            t = [2, 3, 1]
            w = [0.1, 0.1, 0.2]
            gw = gradient(w) do w
                g = GNNGraph(s, t, w, graph_type = GRAPH_T)
                A = adjacency_matrix(g, weighted = false)
                sum(A)
            end[1]
            @test gw === nothing

            gw = gradient(w) do w
                g = GNNGraph(s, t, w, graph_type = GRAPH_T)
                A = adjacency_matrix(g, weighted = true)
                sum(A)
            end[1]

            @test gw == [1, 1, 1]
        end
        
        @testset  "fmt" begin
            Asparse = adjacency_matrix(g, weighted = false, fmt = :sparse)
            @test Asparse isa AbstractSparseMatrix 
            Adense = adjacency_matrix(g, weighted = false, fmt = :dense)
            @test Adense isa AbstractMatrix
            @test !(Adense isa AbstractSparseMatrix)
            @test collect(Asparse) == Adense
        end
    end

    @testset "khop_adj" begin
        s = [1, 2, 3]
        t = [2, 3, 1]
        w = [0.1, 0.1, 0.2]
        g = GNNGraph(s, t, w)
        @test khop_adj(g, 2) == adjacency_matrix(g) * adjacency_matrix(g)
        @test khop_adj(g, 2, Int8; weighted = false) == sparse([0 0 1; 1 0 0; 0 1 0])
        @test khop_adj(g, 2, Int8; dir = in, weighted = false) ==
                sparse([0 0 1; 1 0 0; 0 1 0]')
        @test khop_adj(g, 1) == adjacency_matrix(g)
        @test eltype(khop_adj(g, 4)) == Float64
        @test eltype(khop_adj(g, 10, Float32)) == Float32
    end
end

@testitem "HeteroGraph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    @testset "graph_indicator" begin
        gs = [rand_heterograph(Dict(:user => 10, :movie => 20, :actor => 30), 
                            Dict((:user,:like,:movie) => 10,
                                    (:actor,:rate,:movie)=>20)) for _ in 1:3]
        g = MLUtils.batch(gs)
        @test graph_indicator(g) == Dict(:user => [repeat([1], 10); repeat([2], 10); repeat([3], 10)],
                                    :movie => [repeat([1], 20); repeat([2], 20); repeat([3], 20)],
                                    :actor => [repeat([1], 30); repeat([2], 30); repeat([3], 30)])
        @test graph_indicator(g, :movie) == [repeat([1], 20); repeat([2], 20); repeat([3], 20)]
    end 
end

@testitem "get_graph_type" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        g = rand_graph(10, 20, graph_type = GRAPH_T)
        @test get_graph_type(g) == GRAPH_T

        gsparse = GNNGraph(g, graph_type=:sparse)
        @test get_graph_type(gsparse) == :sparse
        @test gsparse.graph isa SparseMatrixCSC

        gcoo = GNNGraph(g, graph_type=:coo)
        @test get_graph_type(gcoo) == :coo
        @test gcoo.graph[1:2] isa Tuple{Vector{Int}, Vector{Int}}


        gdense = GNNGraph(g, graph_type=:dense)
        @test get_graph_type(gdense) == :dense
        @test gdense.graph isa Matrix{Int}
    end
end
