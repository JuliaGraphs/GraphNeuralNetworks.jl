```@meta
CurrentModule = GNNGraphs
```

# Temporal Graphs

Temporal graphs are graphs with time-varying topologies and features. In GNNGraphs.jl, they are represented by the [`TemporalSnapshotsGNNGraph`](@ref) type.

## Creating a TemporalSnapshotsGNNGraph

A temporal graph can be created by passing a list of snapshots to the constructor. Each snapshot is a [`GNNGraph`](@ref). 

```jldoctest temporal
julia> using GNNGraphs

julia> snapshots = [rand_graph(10, 20) for i in 1:5];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5
```

A new temporal graph can be created by adding or removing snapshots to an existing temporal graph. 

```jldoctest temporal
julia> new_tg = add_snapshot(tg, 3, rand_graph(10, 16)) # add a new snapshot at time 3
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10, 10]
  num_edges: [20, 20, 16, 20, 20, 20]
  num_snapshots: 6
```
```jldoctest temporal
julia> snapshots = [rand_graph(10,20), rand_graph(10,14), rand_graph(10,22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> new_tg = remove_snapshot(tg, 2) # remove snapshot at time 2
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10]
  num_edges: [20, 22]
  num_snapshots: 2
```

See [`rand_temporal_radius_graph`](@ref) and [`rand_temporal_hyperbolic_graph`](@ref) for generating random temporal graphs. 

```julia
julia> tg = rand_temporal_radius_graph(10, 3, 0.1, 0.5)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [32, 30, 34]
  num_snapshots: 3
``` 

## Indexing

Snapshots in a temporal graph can be accessed using indexing:

```jldoctest temporal
julia> snapshots = [rand_graph(10, 20), rand_graph(10, 14), rand_graph(10, 22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> tg[1] # first snapshot
GNNGraph:
  num_nodes: 10
  num_edges: 20

julia> tg[2:3] # snapshots 2 and 3
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10]
  num_edges: [14, 22]
  num_snapshots: 2
```

A snapshot can be modified by assigning a new snapshot to the temporal graph:

```jldoctest temporal
julia> tg[1] = rand_graph(10, 16) # replace first snapshot
GNNGraph:
  num_nodes: 10
  num_edges: 16
```

## Iteration and Broadcasting

Iteration and broadcasting over a temporal graph is similar to that of a vector of snapshots:

```jldoctest temporal
julia> snapshots = [rand_graph(10, 20), rand_graph(10, 14), rand_graph(10, 22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots);

julia> [g for g in tg] # iterate over snapshots
3-element Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}:
 GNNGraph(10, 20) with no data
 GNNGraph(10, 14) with no data
 GNNGraph(10, 22) with no data

julia> f(g) = g isa GNNGraph;

julia> f.(tg) # broadcast over snapshots
3-element BitVector:
 1
 1
 1
```

## Basic Queries

Basic queries are similar to those for [`GNNGraph`](@ref)s:
```jldoctest temporal
julia> snapshots = [rand_graph(10,20), rand_graph(12,14), rand_graph(14,22)];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 12, 14]
  num_edges: [20, 14, 22]
  num_snapshots: 3

julia> tg.num_nodes         # number of nodes in each snapshot
3-element Vector{Int64}:
 10
 12
 14

julia> tg.num_edges         # number of edges in each snapshot
3-element Vector{Int64}:
 20
 14
 22

julia> tg.num_snapshots     # number of snapshots
3

julia> tg.snapshots         # list of snapshots
3-element Vector{GNNGraph{Tuple{Vector{Int64}, Vector{Int64}, Nothing}}}:
 GNNGraph(10, 20) with no data
 GNNGraph(12, 14) with no data
 GNNGraph(14, 22) with no data

julia> tg.snapshots[1]      # first snapshot, same as tg[1]
GNNGraph:
  num_nodes: 10
  num_edges: 20
```

## Data Features
A temporal graph can store global feature for the entire time series in the `tgdata` field.
Also, each snapshot can store node, edge, and graph features in the `ndata`, `edata`, and `gdata` fields, respectively. 

```jldoctest temporal
julia> snapshots = [rand_graph(10, 20; ndata = rand(Float32, 3, 10)), 
                    rand_graph(10, 14; ndata = rand(Float32, 4, 10)), 
                    rand_graph(10, 22; ndata = rand(Float32, 5, 10))]; # node features at construction time

julia> tg = TemporalSnapshotsGNNGraph(snapshots);

julia> tg.tgdata.y = rand(Float32, 3, 1); # add global features after construction

julia> tg
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10]
  num_edges: [20, 14, 22]
  num_snapshots: 3
  tgdata:
    y = 3×1 Matrix{Float32}

julia> tg.ndata # vector of DataStore containing node features for each snapshot
3-element Vector{DataStore}:
 DataStore(10) with 1 element:
  x = 3×10 Matrix{Float32}
 DataStore(10) with 1 element:
  x = 4×10 Matrix{Float32}
 DataStore(10) with 1 element:
  x = 5×10 Matrix{Float32}

julia> [ds.x for ds in tg.ndata]; # vector containing the x feature of each snapshot

julia> [g.x for g in tg.snapshots]; # same vector as above, now accessing 
                                   # the x feature directly from the snapshots
```
