
@testitem "constructor" begin
    @test_throws AssertionError DataStore(10, (:x => rand(10), :y => rand(2, 4)))

    @testset "keyword args" begin
        ds = DataStore(10, x = rand(10), y = rand(2, 10))
        @test size(ds.x) == (10,)
        @test size(ds.y) == (2, 10)

        ds = DataStore(x = rand(10), y = rand(2, 10))
        @test size(ds.x) == (10,)
        @test size(ds.y) == (2, 10)
    end
end

@testitem "getproperty / setproperty!" begin
    x = rand(10)
    ds = DataStore(10, (:x => x, :y => rand(2, 10)))
    @test ds.x == ds[:x] == x
    @test_throws DimensionMismatch ds.z=rand(12)
    ds.z = [1:10;]
    @test ds.z == [1:10;]

    # issue #504, where vector creation failed
    @test fill(DataStore(), 3) isa Vector
end

@testitem "setindex!" begin
    ds = DataStore(10)
    x = rand(10)
    @test (ds[:x] = x) == x # Tests setindex!
    @test ds.x == ds[:x] == x
end

@testitem "map" begin
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))
    ds2 = map(x -> x .+ 1, ds)
    @test ds2.x == ds.x .+ 1
    @test ds2.y == ds.y .+ 1

    @test_throws AssertionError ds2 == map(x -> [x; x], ds)
end

@testitem "getdata / getn" begin
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))
    @test GNNGraphs.getdata(ds) == getfield(ds, :_data)
    @test_throws KeyError ds.data
    @test GNNGraphs.getn(ds) == getfield(ds, :_n)
    @test_throws KeyError ds.n
end

@testitem "cat empty" begin
    ds1 = DataStore(2, (:x => rand(2)))
    ds2 = DataStore(1, (:x => rand(1)))
    dsempty = DataStore(0, (:x => rand(0)))

    ds = GNNGraphs.cat_features(ds1, ds2)
    @test GNNGraphs.getn(ds) == 3
    ds = GNNGraphs.cat_features(ds1, dsempty)
    @test GNNGraphs.getn(ds) == 2

    # issue #280
    g = GNNGraph([1], [2])
    h = add_edges(g, Int[], Int[])  # adds no edges
    @test GNNGraphs.getn(g.edata) == 1
    @test GNNGraphs.getn(h.edata) == 1
end


@testitem "gradient"  setup=[GraphsTestModule] begin
    using .GraphsTestModule
    ds = DataStore(10, (:x => rand(10), :y => rand(2, 10)))

    f1(ds) = sum(ds.x)
    grad = gradient(f1, ds)[1]
    @test grad._data[:x] ≈ ngradient(f1, ds)[1][:x]

    g = rand_graph(5, 2)
    x = rand(2, 5)
    grad = gradient(x -> sum(exp, GNNGraph(g, ndata = x).ndata.x), x)[1]
    @test grad == exp.(x)
end

@testitem "functor" begin
    using Functors
    ds = DataStore(10, (:x => zeros(10), :y => ones(2, 10)))
    p, re = Functors.functor(ds)
    @test p[1] === GNNGraphs.getn(ds)
    @test p[2] === GNNGraphs.getdata(ds)
    @test ds == re(p)

    ds2 = Functors.fmap(ds) do x
        if x isa AbstractArray
            x .+ 1
        else
            x
        end
    end
    @test ds isa DataStore
    @test ds2.x == ds.x .+ 1
end
