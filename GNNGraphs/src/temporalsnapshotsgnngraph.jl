"""
    TemporalSnapshotsGNNGraph(snapshots)

A type representing a time-varying graph as a sequence of snapshots,
each snapshot being a [`GNNGraph`](@ref).

The argument `snapshots` is a collection of `GNNGraph`s with arbitrary 
number of nodes and edges each. 

Calling `tg` the temporal graph, `tg[t]` returns the `t`-th snapshot.

The snapshots can contain node/edge/graph features, while global features for the
whole temporal sequence can be stored in `tg.tgdata`.

See [`add_snapshot`](@ref) and [`remove_snapshot`](@ref) for adding and removing snapshots.

# Examples

```jldoctest
julia> snapshots = [rand_graph(i , 2*i) for i in 10:10:50];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 20, 30, 40, 50]
  num_edges: [20, 40, 60, 80, 100]
  num_snapshots: 5

julia> tg.num_snapshots
5

julia> tg.num_nodes
5-element Vector{Int64}:
 10
 20
 30
 40
 50

julia> tg[1]
GNNGraph:
  num_nodes: 10
  num_edges: 20

julia> tg[2:3]
TemporalSnapshotsGNNGraph:
  num_nodes: [20, 30]
  num_edges: [40, 60]
  num_snapshots: 2

julia> tg[1] = rand_graph(10, 16)
GNNGraph:
  num_nodes: 10
  num_edges: 16
```
"""
struct TemporalSnapshotsGNNGraph{G<:GNNGraph, D<:DataStore}
    num_nodes::Vector{Int}   
    num_edges::Vector{Int}
    num_snapshots::Int
    snapshots::Vector{G}
    tgdata::D   
end

# do not move to gpu num_nodes and num_edges
@functor TemporalSnapshotsGNNGraph (snapshots, tgdata)

function TemporalSnapshotsGNNGraph(snapshots)
    snapshots = collect(snapshots)
    return TemporalSnapshotsGNNGraph(
        [s.num_nodes for s in snapshots],
        [s.num_edges for s in snapshots],
        length(snapshots),
        collect(snapshots),
        DataStore()
    )
end

function Base.:(==)(tsg1::TemporalSnapshotsGNNGraph, tsg2::TemporalSnapshotsGNNGraph)
    tsg1 === tsg2 && return true
    for k in fieldnames(typeof(tsg1))
        getfield(tsg1, k) != getfield(tsg2, k) && return false
    end
    return true
end

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::Int)
    return tg.snapshots[t]
end

function Base.getindex(tg::TemporalSnapshotsGNNGraph, t::AbstractVector)
    return TemporalSnapshotsGNNGraph(tg.num_nodes[t], tg.num_edges[t], 
                length(t), tg.snapshots[t], tg.tgdata)
end

function Base.length(tg::TemporalSnapshotsGNNGraph)
    return tg.num_snapshots
end 

# Allow broadcasting over the temporal snapshots
Base.broadcastable(tg::TemporalSnapshotsGNNGraph) = tg.snapshots

Base.iterate(tg::TemporalSnapshotsGNNGraph) = Base.iterate(tg.snapshots)
Base.iterate(tg::TemporalSnapshotsGNNGraph, i) = Base.iterate(tg.snapshots, i)

function Base.setindex!(tg::TemporalSnapshotsGNNGraph, g::GNNGraph, t::Int)
    tg.snapshots[t] = g
    tg.num_nodes[t] = g.num_nodes
    tg.num_edges[t] = g.num_edges
    return tg
end

"""
    add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)

Return a `TemporalSnapshotsGNNGraph` created starting from `tg` by adding the snapshot `g` at time index `t`.

# Examples

```jldoctest
julia> snapshots = [rand_graph(10, 20) for i in 1:5];

julia> tg = TemporalSnapshotsGNNGraph(snapshots)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [20, 20, 20, 20, 20]
  num_snapshots: 5

julia> new_tg = add_snapshot(tg, 3, rand_graph(10, 16)) # add a new snapshot at time 3
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10, 10]
  num_edges: [20, 20, 16, 20, 20, 20]
  num_snapshots: 6
```
"""
function add_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)
    if tg.num_snapshots > 0
        @assert g.num_nodes == first(tg.num_nodes) "number of nodes must match"
    end
    @assert t <= tg.num_snapshots + 1 "cannot add snapshot at time $t, the temporal graph has only $(tg.num_snapshots) snapshots"
    num_nodes = tg.num_nodes |> copy
    num_edges = tg.num_edges |> copy
    snapshots = tg.snapshots |> copy
    num_snapshots = tg.num_snapshots + 1
    insert!(num_nodes, t, g.num_nodes)
    insert!(num_edges, t, g.num_edges)
    insert!(snapshots, t, g)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

# """
#     add_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)

# Add to `tg` the snapshot `g` at time index `t`.

# See also [`add_snapshot`](@ref) for a non-mutating version.
# """
# function add_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int, g::GNNGraph)
#     if t > tg.num_snapshots + 1
#         error("cannot add snapshot at time $t, the temporal graph has only $(tg.num_snapshots) snapshots")
#     end
#     if tg.num_snapshots > 0
#         @assert g.num_nodes == first(tg.num_nodes) "number of nodes must match"
#     end
#     insert!(tg.num_nodes, t, g.num_nodes)
#     insert!(tg.num_edges, t, g.num_edges)
#     insert!(tg.snapshots, t, g)
#     return tg 
# end

"""
    remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)

Return a [`TemporalSnapshotsGNNGraph`](@ref) created starting from `tg` by removing the snapshot at time index `t`.

# Examples

```jldoctest
julia> using GNNGraphs

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
"""
function remove_snapshot(tg::TemporalSnapshotsGNNGraph, t::Int)
    num_nodes = tg.num_nodes |> copy
    num_edges = tg.num_edges |> copy
    snapshots = tg.snapshots |> copy
    num_snapshots = tg.num_snapshots - 1
    deleteat!(num_nodes, t)
    deleteat!(num_edges, t)
    deleteat!(snapshots, t)
    return TemporalSnapshotsGNNGraph(num_nodes, num_edges, num_snapshots, snapshots, tg.tgdata) 
end

# """
#     remove_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int)

# Remove the snapshot at time index `t` from `tg` and return `tg`.

# See [`remove_snapshot`](@ref) for a non-mutating version.
# """
# function remove_snapshot!(tg::TemporalSnapshotsGNNGraph, t::Int)
#     @assert t <= tg.num_snapshots "snapshot index $t out of bounds"
#     tg.num_snapshots -= 1
#     deleteat!(tg.num_nodes, t)
#     deleteat!(tg.num_edges, t)
#     deleteat!(tg.snapshots, t)
#     return tg
# end

function Base.getproperty(tg::TemporalSnapshotsGNNGraph, prop::Symbol)
    if prop ∈ fieldnames(TemporalSnapshotsGNNGraph)
        return getfield(tg, prop)
    else
        return [getproperty(s, prop) for s in tg.snapshots]
    end
end

function Base.show(io::IO, tsg::TemporalSnapshotsGNNGraph)
    print(io, "TemporalSnapshotsGNNGraph($(tsg.num_snapshots))")
end

function Base.show(io::IO, ::MIME"text/plain", tsg::TemporalSnapshotsGNNGraph)
    if get(io, :compact, false)
        print(io, "TemporalSnapshotsGNNGraph($(tsg.num_snapshots))")
    else
        print(io,
              "TemporalSnapshotsGNNGraph:\n  num_nodes: $(tsg.num_nodes)\n  num_edges: $(tsg.num_edges)\n  num_snapshots: $(tsg.num_snapshots)")
        if !isempty(tsg.tgdata)
            print(io, "\n  tgdata:")
            for k in keys(tsg.tgdata)
                print(io, "\n    $k = $(shortsummary(tsg.tgdata[k]))")
            end
        end
    end
end
