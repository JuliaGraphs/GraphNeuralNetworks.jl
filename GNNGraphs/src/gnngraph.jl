#===================================
Define GNNGraph type as a subtype of Graphs.AbstractGraph.
For the core methods to be implemented by any AbstractGraph, see
https://juliagraphs.org/Graphs.jl/latest/types/#AbstractGraph-Type
https://juliagraphs.org/Graphs.jl/latest/developing/#Developing-Alternate-Graph-Types
=============================================#

"""
    GNNGraph(data; [graph_type, ndata, edata, gdata, num_nodes, graph_indicator, dir])
    GNNGraph(g::GNNGraph; [ndata, edata, gdata])

A type representing a graph structure that also stores
feature arrays associated to nodes, edges, and the graph itself.

The feature arrays are stored in the fields `ndata`, `edata`, and `gdata`
as [`DataStore`](@ref) objects offering a convenient dictionary-like 
and namedtuple-like interface. The features can be passed at construction
time or added later.

A `GNNGraph` can be constructed out of different `data` objects
expressing the connections inside the graph. The internal representation type
is determined by `graph_type`.

When constructed from another `GNNGraph`, the internal graph representation
is preserved and shared. The node/edge/graph features are retained
as well, unless explicitely set by the keyword arguments
`ndata`, `edata`, and `gdata`.

A `GNNGraph` can also represent multiple graphs batched togheter
(see [`MLUtils.batch`](@ref) or [`SparseArrays.blockdiag`](@ref)).
The field `g.graph_indicator` contains the graph membership
of each node.

`GNNGraph`s are always directed graphs, therefore each edge is defined
by a source node and a target node (see [`edge_index`](@ref)).
Self loops (edges connecting a node to itself) and multiple edges
(more than one edge between the same pair of nodes) are supported.

A `GNNGraph` is a Graphs.jl's `AbstractGraph`, therefore it supports most
functionality from that library.

# Arguments

- `data`: Some data representing the graph topology. Possible type are
    - An adjacency matrix
    - An adjacency list.
    - A tuple containing the source and target vectors (COO representation)
    - A Graphs.jl' graph.
- `graph_type`: A keyword argument that specifies
                the underlying representation used by the GNNGraph.
                Currently supported values are
    - `:coo`. Graph represented as a tuple `(source, target)`, such that the `k`-th edge
              connects the node `source[k]` to node `target[k]`.
              Optionally, also edge weights can be given: `(source, target, weights)`.
    - `:sparse`. A sparse adjacency matrix representation.
    - `:dense`. A dense adjacency matrix representation.
    Defaults to `:coo`, currently the most supported type.
- `dir`: The assumed edge direction when given adjacency matrix or adjacency list input data `g`.
        Possible values are `:out` and `:in`. Default `:out`.
- `num_nodes`: The number of nodes. If not specified, inferred from `g`. Default `nothing`.
- `graph_indicator`: For batched graphs, a vector containing the graph assignment of each node. Default `nothing`.
- `ndata`: Node features. An array or named tuple of arrays whose last dimension has size `num_nodes`.
- `edata`: Edge features. An array or named tuple of arrays whose last dimension has size `num_edges`.
- `gdata`: Graph features. An array or named tuple of arrays whose last dimension has size `num_graphs`.

# Examples

```julia
using GNNGraphs

# Construct from adjacency list representation
data = [[2,3], [1,4,5], [1], [2,5], [2,4]]
g = GNNGraph(data)

# Number of nodes, edges, and batched graphs
g.num_nodes  # 5
g.num_edges  # 10
g.num_graphs # 1

# Same graph in COO representation
s = [1,1,2,2,2,3,4,4,5,5]
t = [2,3,1,4,5,3,2,5,2,4]
g = GNNGraph(s, t)

# From a Graphs' graph
g = GNNGraph(erdos_renyi(100, 20))

# Add 2 node feature arrays at creation time
g = GNNGraph(g, ndata = (x=rand(100, g.num_nodes), y=rand(g.num_nodes)))

# Add 1 edge feature array, after the graph creation
g.edata.z = rand(16, g.num_edges)

# Add node features and edge features with default names `x` and `e`
g = GNNGraph(g, ndata = rand(100, g.num_nodes), edata = rand(16, g.num_edges))

g.ndata.x # or just g.x
g.edata.e # or just g.e

# Collect edges' source and target nodes.
# Both source and target are vectors of length num_edges
source, target = edge_index(g)
```
A `GNNGraph` can be sent to the GPU, for example by using Flux.jl's `gpu` function
or MLDataDevices.jl's utilities. 
```
"""
struct GNNGraph{T <: Union{COO_T, ADJMAT_T}} <: AbstractGNNGraph{T}
    graph::T
    num_nodes::Int
    num_edges::Int
    num_graphs::Int
    graph_indicator::Union{Nothing, AVecI}       # vector of ints or nothing
    ndata::DataStore
    edata::DataStore
    gdata::DataStore
end

function GNNGraph(data::D;
                  num_nodes = nothing,
                  graph_indicator = nothing,
                  graph_type = :coo,
                  dir = :out,
                  ndata = nothing,
                  edata = nothing,
                  gdata = nothing) where {D <: Union{COO_T, ADJMAT_T, ADJLIST_T}}
    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]

    if graph_type == :coo
        graph, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        graph, num_nodes, num_edges = to_dense(data; num_nodes, dir)
    elseif graph_type == :sparse
        graph, num_nodes, num_edges = to_sparse(data; num_nodes, dir)
    end

    num_graphs = !isnothing(graph_indicator) ? maximum(graph_indicator) : 1

    ndata = normalize_graphdata(ndata, default_name = :x, n = num_nodes)
    edata = normalize_graphdata(edata, default_name = :e, n = num_edges,
                                duplicate_if_needed = true)

    # don't force the shape of the data when there is only one graph
    gdata = normalize_graphdata(gdata, default_name = :u,
                                n = num_graphs > 1 ? num_graphs : -1)

    GNNGraph(graph,
             num_nodes, num_edges, num_graphs,
             graph_indicator,
             ndata, edata, gdata)
end

GNNGraph(; kws...) = GNNGraph(0; kws...)

function (::Type{<:GNNGraph})(num_nodes::T; kws...) where {T <: Integer}
    s, t = T[], T[]
    return GNNGraph(s, t; num_nodes, kws...)
end

Base.zero(::Type{G}) where {G <: GNNGraph} = G(0)

# COO convenience constructors
function GNNGraph(s::AbstractVector, t::AbstractVector, v = nothing; kws...)
    GNNGraph((s, t, v); kws...)
end
GNNGraph((s, t)::NTuple{2}; kws...) = GNNGraph((s, t, nothing); kws...)

# GNNGraph(g::AbstractGraph; kws...) = GNNGraph(adjacency_matrix(g, dir=:out); kws...)

function GNNGraph(g::AbstractGraph; edge_weight = nothing, kws...)
    s = Graphs.src.(Graphs.edges(g))
    t = Graphs.dst.(Graphs.edges(g))
    w = edge_weight
    if !Graphs.is_directed(g)
        # add reverse edges since GNNGraph is directed
        s, t = [s; t], [t; s]
        if !isnothing(w)
            @assert length(w) == Graphs.ne(g) "edge_weight must have length equal to the number of undirected edges"
            w = [w; w]
        end
    end
    num_nodes::Int = Graphs.nv(g)
    GNNGraph((s, t, w); num_nodes = num_nodes, kws...)
end

function GNNGraph(g::GNNGraph; ndata = g.ndata, edata = g.edata, gdata = g.gdata,
                  graph_type = nothing)
    ndata = normalize_graphdata(ndata, default_name = :x, n = g.num_nodes)
    edata = normalize_graphdata(edata, default_name = :e, n = g.num_edges,
                                duplicate_if_needed = true)
    gdata = normalize_graphdata(gdata, default_name = :u, n = g.num_graphs)

    if !isnothing(graph_type)
        if graph_type == :coo
            graph, num_nodes, num_edges = to_coo(g.graph; g.num_nodes)
        elseif graph_type == :dense
            graph, num_nodes, num_edges = to_dense(g.graph; g.num_nodes)
        elseif graph_type == :sparse
            graph, num_nodes, num_edges = to_sparse(g.graph; g.num_nodes)
        end
        @assert num_nodes == g.num_nodes
        @assert num_edges == g.num_edges
    else
        graph = g.graph
    end
    return GNNGraph(graph,
                g.num_nodes, g.num_edges, g.num_graphs,
                g.graph_indicator,
                ndata, edata, gdata)
end

"""
    copy(g::GNNGraph; deep=false)

Create a copy of `g`. If `deep` is `true`, then copy will be a deep copy (equivalent to `deepcopy(g)`),
otherwise it will be a shallow copy with the same underlying graph data.
"""
function Base.copy(g::GNNGraph; deep = false)
    if deep
        GNNGraph(deepcopy(g.graph),
                 g.num_nodes, g.num_edges, g.num_graphs,
                 deepcopy(g.graph_indicator),
                 deepcopy(g.ndata), deepcopy(g.edata), deepcopy(g.gdata))
    else
        GNNGraph(g.graph,
                 g.num_nodes, g.num_edges, g.num_graphs,
                 g.graph_indicator,
                 g.ndata, g.edata, g.gdata)
    end
end

feature_summary(x::AbstractArray) = "$(dims2string(size(x)))"
feature_summary(x::AbstractString) = "string($(length(x)))"
feature_summary(x::T) where T = "$T"

function print_feature(io::IO, feature)
    if !isempty(feature)
        if length(keys(feature)) == 1
            k = first(keys(feature))
            v = first(values(feature))
            print(io, "$(k): $(feature_summary(v))")
        else
            print(io, "(")
            for (i, (k, v)) in enumerate(pairs(feature))
                print(io, "$k: $(feature_summary(v))")
                if i == length(feature)
                    print(io, ")")
                else
                    print(io, ", ")
                end
            end
        end
    end
end

function print_all_features(io::IO, feat1, feat2, feat3)
    n1 = length(feat1)
    n2 = length(feat2)
    n3 = length(feat3)
    if n1 == 0 && n2 == 0 && n3 == 0
        print(io, "no")
    elseif n1 != 0 && (n2 != 0 || n3 != 0)
        print_feature(io, feat1)
        print(io, ", ")
    elseif n2 == 0 && n3 == 0
        print_feature(io, feat1)
    end
    if n2 != 0 && n3 != 0
        print_feature(io, feat2)
        print(io, ", ")
    elseif n2 != 0 && n3 == 0
        print_feature(io, feat2)
    end
    print_feature(io, feat3)
end

function Base.show(io::IO, g::GNNGraph)
    print(io, "GNNGraph($(g.num_nodes), $(g.num_edges)) with ")
    print_all_features(io, g.ndata, g.edata, g.gdata)
    print(io, " data")
end

function Base.show(io::IO, ::MIME"text/plain", g::GNNGraph)
    if get(io, :compact, false)
        print(io, "GNNGraph($(g.num_nodes), $(g.num_edges)) with ")
        print_all_features(io, g.ndata, g.edata, g.gdata)
        print(io, " data")
    else
        print(io,
              "GNNGraph:\n  num_nodes: $(g.num_nodes)\n  num_edges: $(g.num_edges)")
        g.num_graphs > 1 && print(io, "\n  num_graphs: $(g.num_graphs)")
        if !isempty(g.ndata)
            print(io, "\n  ndata:")
            for k in keys(g.ndata)
                print(io, "\n    $k = $(shortsummary(g.ndata[k]))")
            end
        end
        if !isempty(g.edata)
            print(io, "\n  edata:")
            for k in keys(g.edata)
                print(io, "\n    $k = $(shortsummary(g.edata[k]))")
            end
        end
        if !isempty(g.gdata)
            print(io, "\n  gdata:")
            for k in keys(g.gdata)
                print(io, "\n    $k = $(shortsummary(g.gdata[k]))")
            end
        end
    end
end

MLUtils.numobs(g::GNNGraph) = g.num_graphs
MLUtils.getobs(g::GNNGraph, i) = getgraph(g, i)

#########################

function Base.:(==)(g1::GNNGraph, g2::GNNGraph)
    g1 === g2 && return true
    for k in fieldnames(typeof(g1))
        k === :graph_indicator && continue
        getfield(g1, k) != getfield(g2, k) && return false
    end
    return true
end

function Base.hash(g::T, h::UInt) where {T <: GNNGraph}
    fs = (getfield(g, k) for k in fieldnames(T) if k !== :graph_indicator)
    return foldl((h, f) -> hash(f, h), fs, init = hash(T, h))
end

function Base.getproperty(g::GNNGraph, s::Symbol)
    if s in fieldnames(GNNGraph)
        return getfield(g, s)
    end
    if (s in keys(g.ndata)) + (s in keys(g.edata)) + (s in keys(g.gdata)) > 1
        throw(ArgumentError("Ambiguous property name $s"))
    end
    if s in keys(g.ndata)
        return g.ndata[s]
    elseif s in keys(g.edata)
        return g.edata[s]
    elseif s in keys(g.gdata)
        return g.gdata[s]
    else
        throw(ArgumentError("$(s) is not a field of GNNGraph"))
    end
end
