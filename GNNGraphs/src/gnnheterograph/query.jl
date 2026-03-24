"""
    edge_index(g::GNNHeteroGraph, [edge_t])

Return a tuple containing two vectors, respectively storing the source and target nodes
for each edges in `g` of type `edge_t = (src_t, rel_t, trg_t)`.

If `edge_t` is not provided, it will error if `g` has more than one edge type.
"""
edge_index(g::GNNHeteroGraph{<:COO_T}, edge_t::EType) = g.graph[edge_t][1:2]
edge_index(g::GNNHeteroGraph{<:COO_T}) = only(g.graph)[2][1:2]

get_edge_weight(g::GNNHeteroGraph{<:COO_T}, edge_t::EType) = g.graph[edge_t][3]

"""
    has_edge(g::GNNHeteroGraph, edge_t, i, j)

Return `true` if there is an edge of type `edge_t` from node `i` to node `j` in `g`.

# Examples

```jldoctest
julia> g = rand_bipartite_heterograph((2, 2), (4, 0), bidirected=false)
GNNHeteroGraph:
  num_nodes: Dict(:A => 2, :B => 2)
  num_edges: Dict((:A, :to, :B) => 4, (:B, :to, :A) => 0)

julia> has_edge(g, (:A,:to,:B), 1, 1)
true

julia> has_edge(g, (:B,:to,:A), 1, 1)
false
```
"""
function Graphs.has_edge(g::GNNHeteroGraph, edge_t::EType, i::Integer, j::Integer)
    s, t = edge_index(g, edge_t)
    return any((s .== i) .& (t .== j))
end


"""
    degree(g::GNNHeteroGraph, edge_type::EType; dir = :in) 

Return a vector containing the degrees of the nodes in `g` GNNHeteroGraph
given `edge_type`.

# Arguments

- `g`: A graph.
- `edge_type`: A tuple of symbols `(source_t, edge_t, target_t)` representing the edge type.
- `T`: Element type of the returned vector. If `nothing`, is
       chosen based on the graph type. Default `nothing`.
- `dir`: For `dir = :out` the degree of a node is counted based on the outgoing edges.
         For `dir = :in`, the ingoing edges are used. If `dir = :both` we have the sum of the two.
         Default `dir = :out`.

"""
function Graphs.degree(g::GNNHeteroGraph, edge::EType, 
                       T::TT = nothing; dir = :out) where {
                                                         TT <: Union{Nothing, Type{<:Number}}}  

    s, t = edge_index(g, edge)

    T = isnothing(T) ? eltype(s) : T

    n_type = dir == :in ? g.ntypes[2] : g.ntypes[1]

    return _degree((s, t), T, dir, nothing, g.num_nodes[n_type])
end

"""
    graph_indicator(g::GNNHeteroGraph, [node_t])

Return a Dict of vectors containing the graph membership
(an integer from `1` to `g.num_graphs`) of each node in the graph for each node type.
If `node_t` is provided, return the graph membership of each node of type `node_t` instead.

See also [`batch`](@ref).
"""
function graph_indicator(g::GNNHeteroGraph)
    return g.graph_indicator
end

function graph_indicator(g::GNNHeteroGraph, node_t::Symbol)
    @assert node_t ∈ g.ntypes
    if isnothing(g.graph_indicator)
        gi = ones_like(edge_index(g, first(g.etypes))[1], Int, g.num_nodes[node_t])
    else
        gi = g.graph_indicator[node_t]
    end
    return gi
end

edge_features(g::GNNHeteroGraph, edge_t::EType) = begin
    ds == g.edata[edge_t]
    isempty(ds) ? nothing : first(values(ds))
end

edge_features(g::GNNHeteroGraph) = begin
    ds = only(values(g.edata))
    isempty(ds) ? nothing : first(values(ds))
end