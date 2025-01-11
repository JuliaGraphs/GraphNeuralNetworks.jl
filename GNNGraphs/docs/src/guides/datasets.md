# Datasets

GNNGraphs.jl doesn't come with its own datasets, but leverages those available in the Julia (and non-Julia) ecosystem. 

## MLDatasets.jl

Some of the [examples in the GraphNeuralNetworks.jl repository](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/tree/master/examples) make use of the [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl) package. There you will find common graph datasets such as Cora, PubMed, Citeseer, TUDataset and [many others](https://juliaml.github.io/MLDatasets.jl/dev/datasets/graphs/).
For graphs with static structures and temporal features, datasets such as METRLA, PEMSBAY, ChickenPox, and WindMillEnergy are available. For graphs featuring both temporal structures and temporal features, the TemporalBrains dataset is suitable.

GraphNeuralNetworks.jl provides the [`mldataset2gnngraph`](@ref) method for interfacing with MLDatasets.jl.

## PyGDatasets.jl

The package [PyGDatasets.jl](https://github.com/CarloLucibello/PyGDatasets.jl) makes available to Julia users the datasets from the [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) library. 

PyGDatasets' datasets are compatible with GNNGraphs, so no additional conversion is needed. 
```julia
julia> using PyGDatasets

julia> dataset = load_dataset("TUDataset", name="MUTAG")
TUDataset(MUTAG) - InMemoryGNNDataset
  num_graphs: 188
  node_features: [:x]
  edge_features: [:edge_attr]
  graph_features: [:y]
  root: /Users/carlo/.julia/scratchspaces/44f67abd-f36e-4be4-bfe5-65f468a62b3d/datasets/TUDataset

julia> g = dataset[1]
GNNGraph:
  num_nodes: 17
  num_edges: 38
  ndata:
    x = 7×17 Matrix{Float32}
  edata:
    edge_attr = 4×38 Matrix{Float32}
  gdata:
    y = 1-element Vector{Int64}

julia> using MLUtils: DataLoader

julia> data_loader = DataLoader(dataset, batch_size=32);
```

PyGDatasets is based on [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). It carries over some heavy dependencies such as python, pytorch and pytorch geometric.
