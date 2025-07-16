"""
    propagate(fmsg, g, aggr; [xi, xj, e])
    propagate(fmsg, g, aggr xi, xj, e=nothing) 

Performs message passing on graph `g`. Takes care of materializing the node features on each edge, 
applying the message function `fmsg`, and returning an aggregated message ``\\bar{\\mathbf{m}}`` 
(depending on the return value of `fmsg`, an array or a named tuple of 
arrays with last dimension's size `g.num_nodes`).

It can be decomposed in two steps:

```julia
m = apply_edges(fmsg, g, xi, xj, e)
m̄ = aggregate_neighbors(g, aggr, m)
```

GNN layers typically call `propagate` in their forward pass,
providing as input `f` a closure.  

# Arguments

- `g`: A `GNNGraph`.
- `xi`: An array or a named tuple containing arrays whose last dimension's size 
        is `g.num_nodes`. It will be appropriately materialized on the
        target node of each edge (see also [`edge_index`](@ref GNNGraphs.edge_index)).
- `xj`: As `xj`, but to be materialized on edges' sources. 
- `e`: An array or a named tuple containing arrays whose last dimension's size is `g.num_edges`.
- `fmsg`: A generic function that will be passed over to [`apply_edges`](@ref). 
      Has to take as inputs the edge-materialized `xi`, `xj`, and `e` 
      (arrays or named tuples of arrays whose last dimension' size is the size of 
      a batch of edges). Its output has to be an array or a named tuple of arrays
      with the same batch size. If also `layer` is passed to propagate,
      the signature of `fmsg` has to be `fmsg(layer, xi, xj, e)` 
      instead of `fmsg(xi, xj, e)`.
- `aggr`: Neighborhood aggregation operator. Use `+`, `mean`, `max`, or `min`. 

# Examples

```julia
using GraphNeuralNetworks, Flux

struct GNNConv <: GNNLayer
    W
    b
    σ
end

Flux.@layer GNNConv

function GNNConv(ch::Pair{Int,Int}, σ=identity)
    in, out = ch
    W = Flux.glorot_uniform(out, in)
    b = zeros(Float32, out)
    GNNConv(W, b, σ)
end

function (l::GNNConv)(g::GNNGraph, x::AbstractMatrix)
    message(xi, xj, e) = l.W * xj
    m̄ = propagate(message, g, +, xj=x)
    return l.σ.(m̄ .+ l.bias)
end

l = GNNConv(10 => 20)
l(g, x)
```

See also [`apply_edges`](@ref) and [`aggregate_neighbors`](@ref).
"""
function propagate end

function propagate(f, g::AbstractGNNGraph, aggr; xi = nothing, xj = nothing, e = nothing)
    propagate(f, g, aggr, xi, xj, e)
end

function propagate(f, g::AbstractGNNGraph, aggr, xi, xj, e = nothing)
    m = apply_edges(f, g, xi, xj, e)
    m̄ = aggregate_neighbors(g, aggr, m)
    return m̄
end

## APPLY EDGES

"""
    apply_edges(fmsg, g; [xi, xj, e])
    apply_edges(fmsg, g, xi, xj, e=nothing)

Returns the message from node `j` to node `i` applying
the message function `fmsg` on the edges in graph `g`.
In the message-passing scheme, the incoming messages 
from the neighborhood of `i` will later be aggregated
in order to update the features of node `i` (see [`aggregate_neighbors`](@ref)).

The function `fmsg` operates on batches of edges, therefore
`xi`, `xj`, and `e` are tensors whose last dimension
is the batch size, or can be named tuples of 
such tensors.
    
# Arguments

- `g`: An `AbstractGNNGraph`.
- `xi`: An array or a named tuple containing arrays whose last dimension's size 
        is `g.num_nodes`. It will be appropriately materialized on the
        target node of each edge (see also [`edge_index`](@ref GNNGraphs.edge_index)).
- `xj`: As `xi`, but now to be materialized on each edge's source node. 
- `e`: An array or a named tuple containing arrays whose last dimension's size is `g.num_edges`.
- `fmsg`: A function that takes as inputs the edge-materialized `xi`, `xj`, and `e`.
       These are arrays (or named tuples of arrays) whose last dimension' size is the size of
       a batch of edges. The output of `f` has to be an array (or a named tuple of arrays)
       with the same batch size. If also `layer` is passed to propagate,
      the signature of `fmsg` has to be `fmsg(layer, xi, xj, e)` 
      instead of `fmsg(xi, xj, e)`.

See also [`propagate`](@ref) and [`aggregate_neighbors`](@ref).
"""
function apply_edges end

function apply_edges(f, g::AbstractGNNGraph; xi = nothing, xj = nothing, e = nothing)
    apply_edges(f, g, xi, xj, e)
end

function apply_edges(f, g::AbstractGNNGraph, xi, xj, e = nothing)
    check_num_nodes(g, (xj, xi))
    check_num_edges(g, e)
    s, t = edge_index(g) # for heterographs, errors if more than one edge type
    xi = GNNGraphs._gather(xi, t)   # size: (D, num_nodes) -> (D, num_edges)
    xj = GNNGraphs._gather(xj, s)
    m = f(xi, xj, e)
    return m
end

##  AGGREGATE NEIGHBORS
@doc raw"""
    aggregate_neighbors(g, aggr, m)

Given a graph `g`, edge features `m`, and an aggregation
operator `aggr` (e.g `+, min, max, mean`), returns the new node
features 
```math
\mathbf{x}_i = \square_{j \in \mathcal{N}(i)} \mathbf{m}_{j\to i}
```

Neighborhood aggregation is the second step of [`propagate`](@ref), 
where it comes after [`apply_edges`](@ref).
"""
function aggregate_neighbors(g::GNNGraph, aggr, m)
    check_num_edges(g, m)
    s, t = edge_index(g)
    return GNNGraphs._scatter(aggr, m, t, g.num_nodes)
end

function aggregate_neighbors(g::GNNHeteroGraph, aggr, m)
    check_num_edges(g, m)
    s, t = edge_index(g)
    dest_node_t = only(g.etypes)[3]
    return GNNGraphs._scatter(aggr, m, t, g.num_nodes[dest_node_t])
end

### MESSAGE FUNCTIONS ###
"""
    copy_xj(xi, xj, e) = xj
"""
copy_xj(xi, xj, e) = xj

"""
    copy_xi(xi, xj, e) = xi
"""
copy_xi(xi, xj, e) = xi

"""
    xi_dot_xj(xi, xj, e) = sum(xi .* xj, dims=1)
"""
xi_dot_xj(xi, xj, e) = sum(xi .* xj, dims = 1)

"""
    xi_sub_xj(xi, xj, e) = xi .- xj
"""
xi_sub_xj(xi, xj, e) = xi .- xj

"""
    xj_sub_xi(xi, xj, e) = xj .- xi
"""
xj_sub_xi(xi, xj, e) = xj .- xi

"""
    e_mul_xj(xi, xj, e) = reshape(e, (...)) .* xj

Reshape `e` into a broadcast compatible shape with `xj`
(by prepending singleton dimensions) then perform
broadcasted multiplication.
"""
function e_mul_xj(xi, xj::AbstractArray{Tj, Nj},
                  e::AbstractArray{Te, Ne}) where {Tj, Te, Nj, Ne}
    @assert Ne <= Nj
    e = reshape(e, ntuple(_ -> 1, Nj - Ne)..., size(e)...)
    return e .* xj
end

"""
    w_mul_xj(xi, xj, w) = reshape(w, (...)) .* xj

Similar to [`e_mul_xj`](@ref) but specialized on scalar edge features (weights).
"""
w_mul_xj(xi, xj::AbstractArray, w::Nothing) = xj # same as copy_xj if no weights

function w_mul_xj(xi, xj::AbstractArray{Tj, Nj}, w::AbstractVector) where {Tj, Nj}
    w = reshape(w, ntuple(_ -> 1, Nj - 1)..., length(w))
    return w .* xj
end

###### PROPAGATE SPECIALIZATIONS ####################
## See also the methods defined in the package extensions.

## COPY_XJ 

function propagate(::typeof(copy_xj), g::GNNGraph, ::typeof(+), xi, xj::AbstractMatrix, e)
    A = adjacency_matrix(g, eltype(xj); weighted = false)
    return xj * A
end

## E_MUL_XJ 

# for weighted convolution
function propagate(::typeof(e_mul_xj), g::GNNGraph, ::typeof(+), xi, xj::AbstractMatrix,
                   e::AbstractVector)
    g = set_edge_weight(g, e)
    A = adjacency_matrix(g, weighted = true)
    return xj * A
end


## W_MUL_XJ 

# for weighted convolution
function propagate(::typeof(w_mul_xj), g::GNNGraph, ::typeof(+), xi, xj::AbstractMatrix,
                   e::Nothing)
    A = adjacency_matrix(g, eltype(xj); weighted = true)
    return xj * A
end


# function propagate(::typeof(copy_xj), g::GNNGraph, ::typeof(mean), xi, xj::AbstractMatrix, e)
#     A = adjacency_matrix(g, weighted=false)
#     D = compute_degree(A)
#     return xj * A * D
# end

# # Zygote bug. Error with sparse matrix without nograd
# compute_degree(A) = Diagonal(1f0 ./ vec(sum(A; dims=2)))

# Flux.Zygote.@nograd compute_degree
