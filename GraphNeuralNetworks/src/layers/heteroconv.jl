@doc raw"""
    HeteroGraphConv(itr; aggr = +)
    HeteroGraphConv(pairs...; aggr = +)

A convolutional layer for heterogeneous graphs.

The `itr` argument is an iterator of `pairs` of the form `edge_t => layer`, where `edge_t` is a
3-tuple of the form `(src_node_type, edge_type, dst_node_type)`, and `layer` is a 
convolutional layers for homogeneous graphs. 

Each convolution is applied to the corresponding relation. 
Since a node type can be involved in multiple relations, the single convolution outputs 
have to be aggregated using the `aggr` function. The default is to sum the outputs.

# Forward Arguments

* `g::GNNHeteroGraph`: The input graph.
* `x::Union{NamedTuple,Dict}`: The input node features. The keys are node types and the
  values are node feature tensors.

# Examples 

```jldoctest
julia> using GraphNeuralNetworks, Flux

julia> g = rand_bipartite_heterograph((10, 15), 80)
GNNHeteroGraph:
  num_nodes: Dict(:A => 10, :B => 15)
  num_edges: Dict((:A, :to, :B) => 80, (:B, :to, :A) => 80)

julia> x = (A = rand(Float32, 64, 10), B = rand(Float32, 64, 15));

julia> layer = HeteroGraphConv((:A, :to, :B) => GraphConv(64 => 32, relu),
                               (:B, :to, :A) => GraphConv(64 => 32, relu));

julia> y = layer(g, x); # output is a named tuple

julia> size(y.A) == (32, 10) && size(y.B) == (32, 15)
true
```
"""
struct HeteroGraphConv
    etypes::Vector{EType}
    layers::Vector{<:GNNLayer}
    aggr::Function
end

Flux.@layer HeteroGraphConv

HeteroGraphConv(itr::Dict; aggr = +) = HeteroGraphConv(pairs(itr); aggr)
HeteroGraphConv(itr::Pair...; aggr = +) = HeteroGraphConv(itr; aggr)

function HeteroGraphConv(itr; aggr = +)
    etypes = [k[1] for k in itr]
    layers = [k[2] for k in itr]
    return HeteroGraphConv(etypes, layers, aggr)
end

function (hgc::HeteroGraphConv)(g::GNNHeteroGraph, x::Union{NamedTuple,Dict})
    function forw(l, et)
        sg = edge_type_subgraph(g, et)
        node1_t, _, node2_t = et
        return l(sg, (x[node1_t], x[node2_t]))
    end
    outs = [forw(l, et) for (l, et) in zip(hgc.layers, hgc.etypes)]
    dst_ntypes = [et[3] for et in hgc.etypes]
    return _reduceby_node_t(hgc.aggr, outs, dst_ntypes)
end

function _reduceby_node_t(aggr, outs, ntypes)
    function _reduce(node_t)
        idxs = findall(x -> x == node_t, ntypes)
        if length(idxs) == 0
            return nothing
        elseif length(idxs) == 1
            return outs[idxs[1]]
        else
            return foldl(aggr, outs[i] for i in idxs)
        end
    end
    # workaround to provide the aggregation once per unique node type,
    # gradient is not needed
    unique_ntypes = ChainRulesCore.ignore_derivatives() do
        unique(ntypes)
    end
    vals = [_reduce(node_t) for node_t in unique_ntypes]
    return NamedTuple{tuple(unique_ntypes...)}(vals)
end

function Base.show(io::IO, hgc::HeteroGraphConv)
    if get(io, :compact, false)
        print(io, "HeteroGraphConv(aggr=$(hgc.aggr))")
    else
        println(io, "HeteroGraphConv(aggr=$(hgc.aggr)):")
        for (i, (et,layer)) in enumerate(zip(hgc.etypes, hgc.layers))
            print(io, "  $(et => layer)")
            if i < length(hgc.etypes)
                print(io, "\n")
            end
        end
    end
end
