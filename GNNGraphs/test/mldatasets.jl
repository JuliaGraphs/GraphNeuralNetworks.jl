@testitem "mldataset2gnngraph" begin
    using MLDatasets: Cora
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets
    dataset = Cora()
    classes = dataset.metadata["classes"]
    gml = dataset[1]
    g = mldataset2gnngraph(dataset)
    @test g isa GNNGraph
    @test g.num_nodes == gml.num_nodes
    @test g.num_edges == gml.num_edges
    @test edge_index(g) === gml.edge_index
end
