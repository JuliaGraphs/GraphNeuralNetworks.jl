```@meta
CurrentModule = GNNGraphs
CollapsedDocStrings = true
```

# GNNGraph

Documentation page for the graph type `GNNGraph` provided by GNNGraphs.jl and related methods. 

Besides the methods documented here, one can rely on the large set of functionalities
given by [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) thanks to the fact
that `GNNGraph` inherits from `Graphs.AbstractGraph`.


## GNNGraph type

```@docs
GNNGraph
Base.copy
```

## DataStore

```@autodocs; canonical = true
Modules = [GNNGraphs]
Pages   = ["datastore.jl"]
Private = false
```

## Query

```@autodocs; canonical = true
Modules = [GNNGraphs]
Pages   = ["src/query.jl"]
Private = false
```

```@docs; canonical = true
Graphs.neighbors(::GNNGraph, ::Integer)
```

## Transform

```@autodocs; canonical = true
Modules = [GNNGraphs]
Pages   = ["src/transform.jl"]
Private = false
```

## Utils

```@docs; canonical = true
GNNGraphs.sort_edge_index
GNNGraphs.color_refinement
``` 

## Generate

```@autodocs; canonical = true
Modules = [GNNGraphs]
Pages   = ["src/generate.jl"]
Private = false
Filter = t -> typeof(t) <: Function && t!=rand_temporal_radius_graph && t!=rand_temporal_hyperbolic_graph
```

## Operators

```@autodocs; canonical = true
Modules = [GNNGraphs]
Pages   = ["src/operators.jl"]
Private = false
```

```@docs; canonical = true
Base.intersect
```

## Sampling 

```@docs
NeighborLoader
sample_neighbors
Graphs.induced_subgraph(::GNNGraph, ::Vector{Int})
```
