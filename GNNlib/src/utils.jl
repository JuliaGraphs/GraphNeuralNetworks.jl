ofeltype(x, y) = convert(float(eltype(x)), y)

"""
    reduce_nodes(aggr, g, x)

For a batched graph `g`, return the graph-wise aggregation of the node
features `x`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.
The returned array will have last dimension `g.num_graphs`.

See also: [`reduce_edges`](@ref).
"""
function reduce_nodes(aggr, g::GNNGraph, x)
    @assert size(x)[end] == g.num_nodes
    indexes = graph_indicator(g)
    return NNlib.scatter(aggr, x, indexes)
end

"""
    reduce_nodes(aggr, indicator::AbstractVector, x)

Return the graph-wise aggregation of the node features `x` given the
graph indicator `indicator`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.

See also [`graph_indicator`](@ref).
"""
function reduce_nodes(aggr, indicator::AbstractVector, x)
    return NNlib.scatter(aggr, x, indicator)
end

"""
    reduce_edges(aggr, g, e)

For a batched graph `g`, return the graph-wise aggregation of the edge
features `e`. The aggregation operator `aggr` can be `+`, `mean`, `max`, or `min`.
The returned array will have last dimension `g.num_graphs`.
"""
function reduce_edges(aggr, g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    s, t = edge_index(g)
    indexes = graph_indicator(g)[s]
    return NNlib.scatter(aggr, e, indexes)
end

"""
    softmax_nodes(g, x)

Graph-wise softmax of the node features `x`.
"""
function softmax_nodes(g::GNNGraph, x)
    @assert size(x)[end] == g.num_nodes
    gi = graph_indicator(g)
    max_ = gather(scatter(max, x, gi), gi)
    num = exp.(x .- max_)
    den = reduce_nodes(+, g, num)
    den = gather(den, gi)
    return num ./ den
end

"""
    softmax_edges(g, e)

Graph-wise softmax of the edge features `e`.
"""
function softmax_edges(g::GNNGraph, e)
    @assert size(e)[end] == g.num_edges
    gi = graph_indicator(g, edges = true)
    max_ = gather(scatter(max, e, gi), gi)
    num = exp.(e .- max_)
    den = reduce_edges(+, g, num)
    den = gather(den, gi)
    return num ./ (den .+ eps(eltype(e)))
end

@doc raw"""
    softmax_edge_neighbors(g, e)

Softmax over each node's neighborhood of the edge features `e`.

```math
\mathbf{e}'_{j\to i} = \frac{e^{\mathbf{e}_{j\to i}}}
                    {\sum_{j'\in N(i)} e^{\mathbf{e}_{j'\to i}}}.
```
"""
function softmax_edge_neighbors(g::AbstractGNNGraph, e)
    if g isa GNNHeteroGraph
        for (key, value) in g.num_edges
            @assert size(e)[end] == value
        end
    else
        @assert size(e)[end] == g.num_edges
    end
    s, t = edge_index(g)
    if isempty(t)
        return zeros_like(e, eltype(e), (size(e)[1:end-1]..., 0))
    end
    max_ = gather(scatter(max, e, t), t)
    num = exp.(e .- max_)
    den = gather(scatter(+, num, t), t)
    return num ./ den
end

"""
    broadcast_nodes(g, x)

Graph-wise broadcast array `x` of size `(*, g.num_graphs)` 
to size `(*, g.num_nodes)`.
"""
function broadcast_nodes(g::GNNGraph, x)
    @assert size(x)[end] == g.num_graphs
    gi = graph_indicator(g)
    return gather(x, gi)
end

"""
    broadcast_edges(g, x)

Graph-wise broadcast array `x` of size `(*, g.num_graphs)` 
to size `(*, g.num_edges)`.
"""
function broadcast_edges(g::GNNGraph, x)
    @assert size(x)[end] == g.num_graphs
    gi = graph_indicator(g, edges = true)
    return gather(x, gi)
end

expand_srcdst(g::AbstractGNNGraph, x) = throw(ArgumentError("Invalid input type, expected matrix or tuple of matrices."))
expand_srcdst(g::AbstractGNNGraph, x::AbstractMatrix) = (x, x)
expand_srcdst(g::AbstractGNNGraph, x::Tuple{<:AbstractMatrix, <:AbstractMatrix}) = x

# Replacement for Base.Fix1 to allow for multiple arguments
struct Fix1{F,X}
    f::F
    x::X
end

(f::Fix1)(y...) = f.f(f.x, y...)
