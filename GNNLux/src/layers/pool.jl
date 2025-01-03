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

@doc raw"""
    GlobalAttentionPool(fgate, ffeat=identity)

Global soft attention layer from the [Gated Graph Sequence Neural
Networks](https://arxiv.org/abs/1511.05493) paper

```math
\mathbf{u}_V = \sum_{i\in V} \alpha_i\, f_{feat}(\mathbf{x}_i)
```

where the coefficients ``\alpha_i`` are given by a [`GNNLib.softmax_nodes`](https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNlib.jl/stable/api/utils/#GNNlib.softmax_nodes)
operation:

```math
\alpha_i = \frac{e^{f_{gate}(\mathbf{x}_i)}}
                {\sum_{i'\in V} e^{f_{gate}(\mathbf{x}_{i'})}}.
```

# Arguments

- `fgate`: The function ``f_{gate}: \mathbb{R}^{D_{in}} \to \mathbb{R}``. 
           It is typically expressed by a neural network.

- `ffeat`: The function ``f_{feat}: \mathbb{R}^{D_{in}} \to \mathbb{R}^{D_{out}}``. 
           It is typically expressed by a neural network.

# Examples

```julia
using Graphs, LuxCore, Lux, GNNLux, Random

rng = Random.default_rng()
chin = 6
chout = 5    

fgate = Dense(chin, 1)
ffeat = Dense(chin, chout)
pool = GlobalAttentionPool(fgate, ffeat)

g = batch([GNNGraph(Graphs.random_regular_graph(10, 4), 
                         ndata=rand(Float32, chin, 10)) 
                for i=1:3])

ps = (fgate = LuxCore.initialparameters(rng, fgate), ffeat = LuxCore.initialparameters(rng, ffeat))
st = (fgate = LuxCore.initialstates(rng, fgate), ffeat = LuxCore.initialstates(rng, ffeat))

u, st = pool(g, g.ndata.x, ps, st)

@assert size(u) == (chout, g.num_graphs)
```
"""
struct GlobalAttentionPool <: GNNContainerLayer{(:fgate, :ffeat)}
    fgate
    ffeat
end

GlobalAttentionPool(fgate) = GlobalAttentionPool(fgate, identity)

function (l::GlobalAttentionPool)(g, x, ps, st)
    fgate = StatefulLuxLayer{true}(l.fgate, ps.fgate, _getstate(st, :fgate))
    ffeat = StatefulLuxLayer{true}(l.ffeat, ps.ffeat, _getstate(st, :ffeat))
    m = (; fgate, ffeat)
    return GNNlib.global_attention_pool(m, g, x), st
end

(l::GlobalAttentionPool)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g), ps, st))

"""
    TopKPool(adj, k, in_channel)

Top-k pooling layer.

# Arguments

- `adj`: Adjacency matrix of a graph.
- `k`: Top-k nodes are selected to pool together.
- `in_channel`: The dimension of input channel.
"""
struct TopKPool{T, S}
    A::AbstractMatrix{T}
    k::Int
    p::AbstractVector{S}
    Ã::AbstractMatrix{T}
end

function TopKPool(adj::AbstractMatrix, k::Int, in_channel::Int; init = glorot_uniform)
    TopKPool(adj, k, init(in_channel), similar(adj, k, k))
end

(t::TopKPool)(x::AbstractArray, ps, st) = GNNlib.topk_pool(t, x)


@doc raw"""
    Set2Set(n_in, n_iters, n_layers = 1)

Set2Set layer from the paper [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391).

For each graph in the batch, the layer computes an output vector of size `2*n_in` by iterating the following steps `n_iters` times:
```math
\mathbf{q} = \mathrm{LSTM}(\mathbf{q}_{t-1}^*)
\alpha_{i} = \frac{\exp(\mathbf{q}^T \mathbf{x}_i)}{\sum_{j=1}^N \exp(\mathbf{q}^T \mathbf{x}_j)} 
\mathbf{r} = \sum_{i=1}^N \alpha_{i} \mathbf{x}_i
\mathbf{q}^*_t = [\mathbf{q}; \mathbf{r}]
```
where `N` is the number of nodes in the graph, `LSTM` is a Long-Short-Term-Memory network with `n_layers` layers, input size `2*n_in` and output size `n_in`.

Given a batch of graphs `g` and node features `x`, the layer returns a matrix of size `(2*n_in, n_graphs)`.
```
"""
struct Set2Set{L} <: GNNContainerLayer{(:lstm,)}
    lstm::L
    num_iters::Int
end

function Set2Set(n_in::Int, n_iters::Int, n_layers::Int = 1)
    @assert n_layers == 1 "multiple layers not implemented yet" #TODO
    n_out = 2 * n_in
    lstm = Lux.LSTMCell(n_out => n_in)
    return Set2Set(lstm, n_iters)
end

function (l::Set2Set)(g, x, ps, st)
    lstm = StatefulLuxLayer{true}(l.lstm, ps.lstm, _getstate(st, :lstm))
    m = (; lstm, Wh = ps.lstm.weight_hh)
    return GNNlib.set2set_pool(m, g, x)
end

(l::Set2Set)(g::GNNGraph) = GNNGraph(g, gdata = l(g, node_features(g), ps, st))
