#TODO add graph_type = GRAPH_TYPE to all constructor calls

@testitem "Constructor array TemporalSnapshotsGNNGraph" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    for GRAPH_T in GRAPH_TYPES
        snapshots = [rand_graph(10, 20, graph_type=GRAPH_T) for i in 1:5]
        tg = TemporalSnapshotsGNNGraph(snapshots)
        @test tg.num_nodes == [10 for i in 1:5]
        @test tg.num_edges == [20 for i in 1:5]
        @test tg.num_snapshots == 5
        
        snapshots = [rand_graph(i, 2*i, graph_type=GRAPH_T) for i in 10:10:50]
        tg = TemporalSnapshotsGNNGraph(snapshots)
        @test tg.num_nodes == [i for i in 10:10:50]
        @test tg.num_edges == [2*i for i in 10:10:50]
        @test tg.num_snapshots == 5
    end
end


@testitem "==" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg1 = TemporalSnapshotsGNNGraph(snapshots)
    tsg2 = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg1 == tsg2
    tsg3 = TemporalSnapshotsGNNGraph(snapshots[1:3])
    @test tsg1 != tsg3
    @test tsg1 !== tsg3
end

@testitem "getindex" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg[3] == snapshots[3]
    @test tsg[[1,2]] == TemporalSnapshotsGNNGraph([10,10], [20,20], 2, snapshots[1:2], tsg.tgdata)
end

@testitem "setindex!" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    g = rand_graph(20, 40)
    tsg[3] = g
    @test tsg.snapshots[3] === g
    @test tsg.num_nodes == [10, 10, 20, 10, 10]
    @test tsg.num_edges == [20, 20, 40, 20, 20]
    @test_throws MethodError tsg[3:4] = g
end

@testitem "getproperty" begin
    x = rand(Float32, 10)
    snapshots = [rand_graph(10, 20, ndata = x) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg.tgdata == DataStore()
    @test tsg.x == tsg.ndata.x == [x for i in 1:5]
    @test_throws KeyError tsg.ndata.w
    @test_throws ArgumentError tsg.w
end

@testitem "add/remove_snapshot" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    g = rand_graph(10, 20)
    tsg = add_snapshot(tsg, 3, g)
    @test tsg.num_nodes == [10 for i in 1:6]
    @test tsg.num_edges == [20 for i in 1:6]
    @test tsg.snapshots[3] == g
    tsg = remove_snapshot(tsg, 3)
    @test tsg.num_nodes == [10 for i in 1:5]
    @test tsg.num_edges == [20 for i in 1:5]
    @test tsg.snapshots == snapshots
end

@testitem "add/remove_snapshot" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    g = rand_graph(10, 20)
    tsg2 = add_snapshot(tsg, 3, g)
    @test tsg2.num_nodes == [10 for i in 1:6]
    @test tsg2.num_edges == [20 for i in 1:6]
    @test tsg2.snapshots[3] == g
    @test tsg2.num_snapshots == 6
    @test tsg.num_nodes == [10 for i in 1:5]
    @test tsg.num_edges == [20 for i in 1:5]
    @test tsg.snapshots[2] === tsg2.snapshots[2]
    @test tsg.snapshots[3] === tsg2.snapshots[4]
    @test length(tsg.snapshots) == 5
    @test tsg.num_snapshots == 5
    
    tsg21 = add_snapshot(tsg2, 7, g)
    @test tsg21.num_snapshots == 7
    
    tsg3 = remove_snapshot(tsg, 3)
    @test tsg3.num_nodes == [10 for i in 1:4]
    @test tsg3.num_edges == [20 for i in 1:4]
    @test tsg3.snapshots == snapshots[[1,2,4,5]]
end


# @testitem "add/remove_snapshot!" begin
#     snapshots = [rand_graph(10, 20) for i in 1:5]
#     tsg = TemporalSnapshotsGNNGraph(snapshots)
#     g = rand_graph(10, 20)
#     tsg2 = add_snapshot!(tsg, 3, g)
#     @test tsg2.num_nodes == [10 for i in 1:6]
#     @test tsg2.num_edges == [20 for i in 1:6]
#     @test tsg2.snapshots[3] == g
#     @test tsg2.num_snapshots == 6
#     @test tsg2 === tsg
    
#     tsg3 = remove_snapshot!(tsg, 3)
#     @test tsg3.num_nodes == [10 for i in 1:4]
#     @test tsg3.num_edges == [20 for i in 1:4]
#     @test length(tsg3.snapshots) === 4
#     @test tsg3 === tsg
# end

@testitem "show" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5)"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact => true) == "TemporalSnapshotsGNNGraph(5)"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact =>  false) == "TemporalSnapshotsGNNGraph:\n  num_nodes: [10, 10, 10, 10, 10]\n  num_edges: [20, 20, 20, 20, 20]\n  num_snapshots: 5"
    tsg.tgdata.x = rand(Float32, 4)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5)"
end

@testitem "broadcastable" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    f(g) = g isa GNNGraph
    @test f.(tsg) == trues(5)
end

@testitem "iterate" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test [g for g in tsg] isa Vector{<:GNNGraph}
end

@testitem "gpu" setup=[GraphsTestModule] tags=[:gpu] begin
    using .GraphsTestModule
    dev = gpu_device(force=true)
    snapshots = [rand_graph(10, 20; ndata = rand(Float32, 5,10)) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    tsg.tgdata.x = rand(Float32, 5)
    tsg = tsg |> dev
    @test tsg.snapshots[1].ndata.x isa AbstractMatrix{Float32}
    @test get_device(tsg.snapshots[1].ndata.x) isa AbstractGPUDevice
    @test tsg.snapshots[end].ndata.x isa AbstractMatrix{Float32}
    @test get_device(tsg.snapshots[end].ndata.x) isa AbstractGPUDevice
    @test tsg.tgdata.x isa AbstractVector{Float32}
    @test get_device(tsg.tgdata.x) isa AbstractGPUDevice
    # num_nodes and num_edges are not copied to the device
    @test tsg.num_nodes isa Vector{Int}
    @test tsg.num_edges isa Vector{Int}
end
