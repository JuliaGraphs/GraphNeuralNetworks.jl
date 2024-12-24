@doc raw"""
    GlobalPool(aggr)

Global pooling layer for graph neural networks.
Takes a graph and feature nodes as inputs
and performs the operation

```math
\mathbf{u}_V = \square_{i \in V} \mathbf{x}_i
```

where ``V`` is the set of nodes of the input graph and 
the type of aggregation represented by ``\square`` is selected by the `aggr` argument. 
Commonly used aggregations are `mean`, `max`, and `+`.

See also [`GNNlib.reduce_nodes`](@ref).

# Examples

```julia
using Lux, GNNLux, Graphs, MLUtils

using Graphs
pool = GlobalPool(mean)

g = GNNGraph(erdos_renyi(10, 4))
X = rand(32, 10)
pool(g, X, ps, st) # => 32x1 matrix


g = MLUtils.batch([GNNGraph(erdos_renyi(10, 4)) for _ in 1:5])
X = rand(32, 50)
pool(g, X, ps, st) # => 32x5 matrix
```
"""
struct GlobalPool{F} <: GNNLayer
    aggr::F
end

(l::GlobalPool)(g::GNNGraph, x::AbstractArray, ps, st) = GNNlib.global_pool(l, g, x), st

(l::GlobalPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g), ps, st))