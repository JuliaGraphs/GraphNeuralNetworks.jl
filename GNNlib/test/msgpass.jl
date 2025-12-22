@testitem "msgpass" setup=[TestModuleGNNlib] begin
    using .TestModuleGNNlib
    #TODO test all graph types
    g = TEST_GRAPHS[1]
    out_channel = size(g.x, 1)
    num_V = g.num_nodes
    num_E = g.num_edges
    g = GNNGraph(g, edata = rand(Float32, size(g.x, 1), g.num_edges))
    
    @testset "propagate" begin
        function message(xi, xj, e)
            @test xi === nothing
            @test e === nothing
            ones(Float32, out_channel, size(xj, 2))
        end

        m = propagate(message, g, +, xj = g.x)

        @test size(m) == (out_channel, num_V)

        @testset "isolated nodes" begin
            x1 = rand(1, 6)
            g1 = GNNGraph(collect(1:5), collect(1:5), num_nodes = 6)
            y1 = propagate((xi, xj, e) -> xj, g1, +, xj = x1)
            @test size(y1) == (1, 6)
        end
    end

    @testset "apply_edges" begin
        m = apply_edges(g, e = g.e) do xi, xj, e
            @test xi === nothing
            @test xj === nothing
            ones(out_channel, size(e, 2))
        end

        @test m == ones(out_channel, num_E)

        # With NamedTuple input
        m = apply_edges(g, xj = (; a = g.x, b = 2g.x), e = g.e) do xi, xj, e
            @test xi === nothing
            @test xj.b == 2 * xj.a
            @test size(xj.a, 2) == size(xj.b, 2) == size(e, 2)
            ones(out_channel, size(e, 2))
        end

        # NamedTuple output
        m = apply_edges(g, e = g.e) do xi, xj, e
            @test xi === nothing
            @test xj === nothing
            (; a = ones(out_channel, size(e, 2)))
        end

        @test m.a == ones(out_channel, num_E)

        @testset "sizecheck" begin
            x = rand(3, g.num_nodes - 1)
            @test_throws AssertionError apply_edges(copy_xj, g, xj = x)
            @test_throws AssertionError apply_edges(copy_xj, g, xi = x)

            x = (a = rand(3, g.num_nodes), b = rand(3, g.num_nodes + 1))
            @test_throws AssertionError apply_edges(copy_xj, g, xj = x)
            @test_throws AssertionError apply_edges(copy_xj, g, xi = x)

            e = rand(3, g.num_edges - 1)
            @test_throws AssertionError apply_edges(copy_xj, g, e = e)
        end
    end

    @testset "copy_xj" begin
        n = 128
        A = sprand(n, n, 0.1)
        Adj = map(x -> x > 0 ? 1 : 0, A)
        X = rand(10, n)

        g = GNNGraph(A, ndata = X, graph_type = :coo)

        function spmm_copyxj_fused(g)
            propagate(copy_xj,
                        g, +; xj = g.ndata.x)
        end

        function spmm_copyxj_unfused(g)
            propagate((xi, xj, e) -> xj,
                        g, +; xj = g.ndata.x)
        end

        @test spmm_copyxj_unfused(g) ≈ X * Adj
        @test spmm_copyxj_fused(g) ≈ X * Adj
    end

    @testset "e_mul_xj and w_mul_xj for weighted conv" begin
        n = 128
        A = sprand(n, n, 0.1)
        Adj = map(x -> x > 0 ? 1 : 0, A)
        X = rand(10, n)

        g = GNNGraph(A, ndata = X, edata = A.nzval, graph_type = :coo)

        function spmm_unfused(g)
            propagate((xi, xj, e) -> reshape(e, 1, :) .* xj,
                        g, +; xj = g.ndata.x, e = g.edata.e)
        end
        function spmm_fused(g)
            propagate(e_mul_xj,
                        g, +; xj = g.ndata.x, e = g.edata.e)
        end

        function spmm_fused2(g)
            propagate(w_mul_xj,
                        g, +; xj = g.ndata.x)
        end

        @test spmm_unfused(g) ≈ X * A
        @test spmm_fused(g) ≈ X * A
        @test spmm_fused2(g) ≈ X * A
    end

    @testset "aggregate_neighbors" begin
        @testset "sizecheck" begin
            m = rand(2, g.num_edges - 1)
            @test_throws AssertionError aggregate_neighbors(g, +, m)

            m = (a = rand(2, g.num_edges + 1), b = nothing)
            @test_throws AssertionError aggregate_neighbors(g, +, m)
        end
    end
end

@testitem "propagate" setup=[TestModuleGNNlib] begin
    using .TestModuleGNNlib

    @testset "copy_xj +" begin
        for g in TEST_GRAPHS
            f(g, x) = propagate(copy_xj, g, +, xj = x)
            test_gradients(f, g, g.x; test_grad_f=false)
        end
    end

    @testset "copy_xj mean" begin
        for g in TEST_GRAPHS
            f(g, x) = propagate(copy_xj, g, mean, xj = x)
            test_gradients(f, g, g.x; test_grad_f=false)
        end
    end

    @testset "e_mul_xj +" begin
        for g in TEST_GRAPHS
            e = rand(Float32, size(g.x, 1), g.num_edges)
            f(g, x, e) = propagate(e_mul_xj, g, +; xj = x, e)
            test_gradients(f, g, g.x, e; test_grad_f=false)
        end
    end

    @testset "w_mul_xj +" begin
        for g in TEST_GRAPHS
            w = rand(Float32, g.num_edges)
            function f(g, x, w)
                g = set_edge_weight(g, w)
                return propagate(w_mul_xj, g, +, xj = x)
            end
            test_gradients(f, g, g.x, w; test_grad_f=false)
        end
    end
end

@testitem "propagate GPU" setup=[TestModuleGNNlib] tags=[:gpu] begin
    using .TestModuleGNNlib

    @testset "copy_xj +" begin
        for g in TEST_GRAPHS
            # broken = get_graph_type(g) == :sparse
            broken = false
            f(g, x) = propagate(copy_xj, g, +, xj = x)
            @test test_gradients(
                f, g, g.x; test_gpu=true, test_grad_f=false, compare_finite_diff=false
            ) broken=broken
        end
    end

    @testset "copy_xj mean" begin
        for g in TEST_GRAPHS
            # broken = get_graph_type(g) == :sparse
            broken = false
            f(g, x) = propagate(copy_xj, g, mean, xj = x)
            @test test_gradients(
                f, g, g.x; test_gpu=true, test_grad_f=false, compare_finite_diff=false
            ) broken=broken
        end
    end

    @testset "e_mul_xj +" begin
        for g in TEST_GRAPHS
            # broken = get_graph_type(g) == :sparse
            broken = false
            e = rand(Float32, size(g.x, 1), g.num_edges)
            f(g, x, e) = propagate(e_mul_xj, g, +; xj = x, e)
            @test test_gradients(
                f, g, g.x, e; test_gpu=true, test_grad_f=false, compare_finite_diff=false
            ) broken=broken
        end
    end

    @testset "w_mul_xj +" begin
        for g in TEST_GRAPHS
            w = rand(Float32, g.num_edges)
            function f(g, x, w)
                g = set_edge_weight(g, w)
                return propagate(w_mul_xj, g, +, xj = x)
            end
            # @show get_graph_type(g) has_isolated_nodes(g)
            # broken = get_graph_type(g) == :sparse
            broken = true
            @test test_gradients(
                f, g, g.x, w; test_gpu=true, test_grad_f=false, compare_finite_diff=false
            ) broken=broken
        end
    end
end
