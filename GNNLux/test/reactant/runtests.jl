# Optional integration tests; the regular GNNLux test project does not depend on
# Reactant. From the repository root, instantiate and run with:
#   julia --project=GNNLux/test/reactant -e 'using Pkg; Pkg.instantiate()'
#   julia --project=GNNLux/test/reactant GNNLux/test/reactant/runtests.jl

using GNNGraphs
using GNNLux
using Lux
using Random
using Reactant
using Test

degree_for_features(g, x) = GNNGraphs.degree(g, eltype(x); dir = :in, edge_weight = false)

function test_graph()
    return GNNGraph(
        ([1, 2, 3, 4, 5, 6], [2, 3, 1, 5, 6, 4]);
        num_nodes = 6,
    )
end

@testset "Reactant degree uses feature numeric type" begin
    g = test_graph()
    x = randn(Xoshiro(42), Float32, 3, 6)
    x_d = x |> reactant_device()

    compiled_degree = @compile degree_for_features(g, x_d)
    degree_d = compiled_degree(g, x_d)

    @test Array(degree_d) == ones(Float32, 6)
end

@testset "Reactant GCNConv compiles, executes, and matches CPU" begin
    rng = Xoshiro(42)
    model = GCNConv(3 => 2, relu)
    ps, st = Lux.setup(rng, model)
    g = test_graph()
    x = randn(rng, Float32, 3, 6)
    y_cpu, _ = model(g, x, ps, st)

    dev = reactant_device()
    ps_d, st_d = (ps, st) |> dev
    x_d = x |> dev
    compiled = @compile model(g, x_d, ps_d, st_d)
    y_d, _ = compiled(g, x_d, ps_d, st_d)

    @test size(y_d) == (2, 6)
    @test Array(y_d) ≈ y_cpu rtol = 1.0f-4 atol = 1.0f-5
end
