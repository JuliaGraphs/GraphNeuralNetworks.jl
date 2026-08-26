@testitem "Enzyme differentiates through adjacency-matrix → COO conversion" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using Enzyme: Enzyme, Const, Reverse, Duplicated, Active
    import NNlib

    # Differentiating the conversion used by gcn_conv & co. on :dense/:sparse graphs
    # used to crash Enzyme (EnzymeAD/Enzyme.jl#3409, fixed in Enzyme 0.13.197).
    rng = MersenneTwister(17)
    loss(x, g) = sum(abs2, NNlib.gather(x, edge_index(GNNGraphs._to_coo_graph(g))[1]))
    for graph_type in (:dense, :sparse)
        g = rand_graph(rng, 6, 10; graph_type)
        x = randn(rng, Float32, 3, g.num_nodes)
        grad = Enzyme.gradient(Reverse, loss, x, Const(g))[1]
        gfd = ngradient(x -> loss(x, g), x)[1]
        @test grad ≈ gfd rtol = 1e-4
    end

    # Edge weights stay on the differentiable path through the conversion.
    loss_w(g) = sum(abs2, get_edge_weight(GNNGraphs._to_coo_graph(g)))
    S = sparse([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], rand(rng, 5), 6, 6)
    g = GNNGraph(S, graph_type = :sparse)
    dg = Enzyme.make_zero(g)
    Enzyme.autodiff(Reverse, loss_w, Active, Duplicated(g, dg))
    @test nonzeros(dg.graph) ≈ 2 .* nonzeros(S)
end

@testitem "Enzyme treats scaled_laplacian as constant" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    using Enzyme: Enzyme, Const, Reverse

    # Without GNNGraphsEnzymeCoreExt Enzyme differentiates the Krylov eigensolve
    # inside `scaled_laplacian` and fails; ChainRules already treats it as constant.
    # Bidirected cycle: symmetric and without isolated nodes.
    src = [1, 2, 3, 4, 5, 6, 7, 8]
    dst = [2, 3, 4, 5, 6, 7, 8, 1]
    g = GNNGraph(vcat(src, dst), vcat(dst, src))
    x = randn(MersenneTwister(17), Float32, 3, g.num_nodes)
    loss(x, g) = sum(abs2, x * scaled_laplacian(g, Float32))

    grad = Enzyme.gradient(Reverse, loss, x, Const(g))[1]
    gfd = ngradient(x -> loss(x, g), x)[1]
    @test grad ≈ gfd rtol = 1e-4
end
