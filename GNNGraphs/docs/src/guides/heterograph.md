```@meta
CurrentModule = GNNGraphs
```

# Heterogeneous Graphs

Heterogeneous graphs (also called heterographs), are graphs where each node has a type,
that we denote with symbols such as `:user` and `:movie`.
Relations such as `:rate` or `:like` can connect nodes of different types. We call a triplet `(source_node_type, relation_type, target_node_type)` the type of a edge, e.g. `(:user, :rate, :movie)`.

Different node/edge types can store different groups of features
and this makes heterographs a very flexible modeling tools 
and data containers. In GNNGraphs.jl heterographs are implemented in 
the type [`GNNHeteroGraph`](@ref).


## Creating a Heterograph

A heterograph can be created empty or by passing pairs `edge_type => data` to the constructor.
```jldoctest hetero
julia> using GNNGraphs

julia> g = GNNHeteroGraph()
GNNHeteroGraph:
  num_nodes: Dict()
  num_edges: Dict()
  
julia> g = GNNHeteroGraph((:user, :like, :actor) => ([1,2,2,3], [1,3,2,9]),
                          (:user, :rate, :movie) => ([1,1,2,3], [7,13,5,7]))
GNNHeteroGraph:
  num_nodes: Dict(:actor => 9, :movie => 13, :user => 3)
  num_edges: Dict((:user, :like, :actor) => 4, (:user, :rate, :movie) => 4)

julia> g = GNNHeteroGraph((:user, :rate, :movie) => ([1,1,2,3], [7,13,5,7]))
GNNHeteroGraph:
  num_nodes: Dict(:movie => 13, :user => 3)
  num_edges: Dict((:user, :rate, :movie) => 4)
```
New relations, possibly with new node types, can be added with the function [`add_edges`](@ref).
```jldoctest hetero
julia> g = add_edges(g, (:user, :like, :actor) => ([1,2,3,3,3], [3,5,1,9,4]))
GNNHeteroGraph:
  num_nodes: Dict(:actor => 9, :movie => 13, :user => 3)
  num_edges: Dict((:user, :like, :actor) => 5, (:user, :rate, :movie) => 4)
```
See [`rand_heterograph`](@ref), [`rand_bipartite_heterograph`](@ref)
for generating random heterographs. 

```jldoctest hetero
julia> g = rand_bipartite_heterograph((10, 15), 20)
GNNHeteroGraph:
  num_nodes: Dict(:A => 10, :B => 15)
  num_edges: Dict((:A, :to, :B) => 20, (:B, :to, :A) => 20)
```

## Basic Queries

Basic queries are similar to those for homogeneous graphs:
```jldoctest hetero
julia> g = GNNHeteroGraph((:user, :rate, :movie) => ([1,1,2,3], [7,13,5,7]))
GNNHeteroGraph:
  num_nodes: Dict(:movie => 13, :user => 3)
  num_edges: Dict((:user, :rate, :movie) => 4)

julia> g.num_nodes
Dict{Symbol, Int64} with 2 entries:
  :user  => 3
  :movie => 13

julia> g.num_edges
Dict{Tuple{Symbol, Symbol, Symbol}, Int64} with 1 entry:
  (:user, :rate, :movie) => 4

julia> edge_index(g, (:user, :rate, :movie)) # source and target node for a given relation
([1, 1, 2, 3], [7, 13, 5, 7])

julia> g.ntypes  # node types
2-element Vector{Symbol}:
 :user
 :movie

julia> g.etypes  # edge types
1-element Vector{Tuple{Symbol, Symbol, Symbol}}:
 (:user, :rate, :movie)
```

## Data Features

Node, edge, and graph features can be added at construction time or later using:
```jldoctest hetero
julia> g[:user].x = rand(Float32, 64, 3); # equivalent to g.ndata[:user][:x] = ...

julia> g[:movie].z = rand(Float32, 64, 13);

julia> g[:user, :rate, :movie].e = rand(Float32, 64, 4); # equivalent to g.edata[(:user, :rate, :movie)][:e] = ...

julia> g
GNNHeteroGraph:
  num_nodes: Dict(:movie => 13, :user => 3)
  num_edges: Dict((:user, :rate, :movie) => 4)
  ndata:
    :movie  =>  DataStore(z = [64×13 Matrix{Float32}])
    :user  =>  DataStore(x = [64×3 Matrix{Float32}])
  edata:
    (:user, :rate, :movie)  =>  DataStore(e = [64×4 Matrix{Float32}])
```

## Batching
Similarly to graphs, also heterographs can be batched together.
```jldoctest hetero
julia> using MLUtils

julia> gs = [rand_bipartite_heterograph((5, 10), 20) for _ in 1:32];

julia> MLUtils.batch(gs)
GNNHeteroGraph:
  num_nodes: Dict(:A => 160, :B => 320)
  num_edges: Dict((:A, :to, :B) => 640, (:B, :to, :A) => 640)
  num_graphs: 32
```
Batching is automatically performed by the [`DataLoader`](https://fluxml.ai/Flux.jl/stable/data/mlutils/#MLUtils.DataLoader) iterator
when the `collate` option is set to `true`.

```julia
using MLUtils: DataLoader

data = [rand_bipartite_heterograph((5, 10), 20, 
            ndata=Dict(:A=>rand(Float32, 3, 5))) 
        for _ in 1:320];

train_loader = DataLoader(data, batchsize=16, shuffle=true, collate=true)

for g in train_loader
    @assert g.num_graphs == 16
    @assert g.num_nodes[:A] == 80
    @assert size(g.ndata[:A].x) == (3, 80)    
    # ...
end
```

## Graph convolutions on heterographs

See `HeteroGraphConv` for how to perform convolutions on heterogeneous graphs.
