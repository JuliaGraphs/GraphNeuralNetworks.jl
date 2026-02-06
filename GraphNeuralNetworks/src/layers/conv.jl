# The implementations of the forward pass of the graph convolutional layers are in the `GNNlib` module,
# in the src/layers/conv.jl file. The `GNNlib` module is re-exported in the GraphNeuralNetworks module.
# This annoying for the readability of the code, as the user has to look at two different files to understand
# the implementation of a single layer, 
# but it is done for GraphNeuralNetworks.jl and GNNLux.jl to be able to share the same code.

@doc raw"""
    GCNConv(in => out, σ=identity; [bias, init, add_self_loops, use_edge_weight])

Graph convolutional layer from paper [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).

Performs the operation
```math
\mathbf{x}'_i = \sum_{j\in N(i)} a_{ij} W \mathbf{x}_j
```
where ``a_{ij} = 1 / \sqrt{|N(i)||N(j)|}`` is a normalization factor computed from the node degrees. 

If the input graph has weighted edges and `use_edge_weight=true`, than ``a_{ij}`` will be computed as
```math
a_{ij} = \frac{e_{j\to i}}{\sqrt{\sum_{j \in N(i)}  e_{j\to i}} \sqrt{\sum_{i \in N(j)}  e_{i\to j}}}
```

The input to the layer is a node feature array `X` of size `(num_features, num_nodes)`
and optionally an edge weight vector.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `σ`: Activation function. Default `identity`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.

# Forward

    (::GCNConv)(g::GNNGraph, x, edge_weight = nothing; norm_fn = d -> 1 ./ sqrt.(d), conv_weight = nothing) -> AbstractMatrix

Takes as input a graph `g`, a node feature matrix `x` of size `[in, num_nodes]`,
and optionally an edge weight vector. Returns a node feature matrix of size 
`[out, num_nodes]`.

The `norm_fn` parameter allows for custom normalization of the graph convolution operation by passing a function as argument. 
By default, it computes ``\frac{1}{\sqrt{d}}`` i.e the inverse square root of the degree (`d`) of each node in the graph. 
If `conv_weight` is an `AbstractMatrix` of size `[out, in]`, then the convolution is performed using that weight matrix instead of the weights stored in the model.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = GCNConv(3 => 5) 

# forward pass
y = l(g, x)       # size:  5 × num_nodes

# convolution with edge weights and custom normalization function
w = [1.1, 0.1, 2.3, 0.5]
custom_norm_fn(d) = 1 ./ sqrt.(d + 1)  # Custom normalization function
y = l(g, x, w; norm_fn = custom_norm_fn)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = GCNConv(3 => 5, use_edge_weight=true) 
y = l(g, x) # same as l(g, x, w) 
```
"""
struct GCNConv{W <: AbstractMatrix, B, F} <: GNNLayer
    weight::W
    bias::B
    σ::F
    add_self_loops::Bool
    use_edge_weight::Bool
end

Flux.@layer GCNConv

function GCNConv(ch::Pair{Int, Int}, σ = identity;
                 init = glorot_uniform,
                 bias::Bool = true,
                 add_self_loops = true,
                 use_edge_weight = false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    GCNConv(W, b, σ, add_self_loops, use_edge_weight)
end


function (l::GCNConv)(g, x, edge_weight = nothing;
                      norm_fn = d -> 1 ./ sqrt.(d),
                      conv_weight = nothing)

    return GNNlib.gcn_conv(l, g, x, edge_weight, norm_fn, conv_weight)
end


function Base.show(io::IO, l::GCNConv)
    out, in = size(l.weight)
    print(io, "GCNConv($in => $out")
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end

@doc raw"""
    ChebConv(in => out, k; bias=true, init=glorot_uniform)

Chebyshev spectral graph convolutional layer from
paper [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).

Implements

```math
X' = \sum^{K-1}_{k=0}  W^{(k)} Z^{(k)}
```

where ``Z^{(k)}`` is the ``k``-th term of Chebyshev polynomials, and can be calculated by the following recursive form:

```math
\begin{aligned}
Z^{(0)} &= X \\
Z^{(1)} &= \hat{L} X \\
Z^{(k)} &= 2 \hat{L} Z^{(k-1)} - Z^{(k-2)}
\end{aligned}
```

with ``\hat{L}`` the [`scaled_laplacian`](@ref).

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: The order of Chebyshev polynomial.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = ChebConv(3 => 5, 5) 

# forward pass
y = l(g, x)       # size:  5 × num_nodes
```
"""
struct ChebConv{W <: AbstractArray{<:Number, 3}, B} <: GNNLayer
    weight::W
    bias::B
    k::Int
end

function ChebConv(ch::Pair{Int, Int}, k::Int;
                  init = glorot_uniform, bias::Bool = true)
    in, out = ch
    W = init(out, in, k)
    b = bias ? Flux.create_bias(W, true, out) : false
    ChebConv(W, b, k)
end

Flux.@layer ChebConv

(l::ChebConv)(g, x) = GNNlib.cheb_conv(l, g, x)

function Base.show(io::IO, l::ChebConv)
    out, in, k = size(l.weight)
    print(io, "ChebConv(", in, " => ", out)
    print(io, ", k=", k)
    print(io, ")")
end

@doc raw"""
    GraphConv(in => out, σ=identity; aggr=+, bias=true, init=glorot_uniform)

Graph convolution layer from Reference: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244).

Performs:
```math
\mathbf{x}_i' = W_1 \mathbf{x}_i + \square_{j \in \mathcal{N}(i)} W_2 \mathbf{x}_j
```

where the aggregation type is selected by `aggr`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = GraphConv(in_channel => out_channel, relu, bias = false, aggr = mean)

# forward pass
y = l(g, x)       
```
"""
struct GraphConv{W <: AbstractMatrix, B, F, A} <: GNNLayer
    weight1::W
    weight2::W
    bias::B
    σ::F
    aggr::A
end

Flux.@layer GraphConv

function GraphConv(ch::Pair{Int, Int}, σ = identity; aggr = +,
                   init = glorot_uniform, bias::Bool = true)
    in, out = ch
    W1 = init(out, in)
    W2 = init(out, in)
    b = bias ? Flux.create_bias(W1, true, out) : false
    GraphConv(W1, W2, b, σ, aggr)
end

(l::GraphConv)(g, x) = GNNlib.graph_conv(l, g, x)

function Base.show(io::IO, l::GraphConv)
    in_channel = size(l.weight1, ndims(l.weight1))
    out_channel = size(l.weight1, ndims(l.weight1) - 1)
    print(io, "GraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

@doc raw"""
    GATConv(in => out, [σ; heads, concat, init, bias, negative_slope, add_self_loops])
    GATConv((in, ein) => out, ...)

Graph attentional layer from the paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903).

Implements the operation
```math
\mathbf{x}_i' = \sum_{j \in N(i) \cup \{i\}} \alpha_{ij} W \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} = \frac{1}{z_i} \exp(LeakyReLU(\mathbf{a}^T [W \mathbf{x}_i; W \mathbf{x}_j]))
```
with ``z_i`` a normalization factor. 

In case `ein > 0` is given, edge features of dimension `ein` will be expected in the forward pass 
and the attention coefficients will be calculated as  
```math
\alpha_{ij} = \frac{1}{z_i} \exp(LeakyReLU(\mathbf{a}^T [W_e \mathbf{e}_{j\to i}; W \mathbf{x}_i; W \mathbf{x}_j]))
```

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. Default 0 (i.e. no edge features passed in the forward).
- `out`: The dimension of output node features.
- `σ`: Activation function. Default `identity`.
- `bias`: Learn the additive bias if true. Default `true`.
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
- `dropout`: Dropout probability on the normalized attention coefficient. Default `0.0`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = GATConv(in_channel => out_channel, add_self_loops = false, bias = false; heads=2, concat=true)

# forward pass
y = l(g, x)       
```
"""
struct GATConv{DX<:Dense,DE<:Union{Dense, Nothing},DV,T,A<:AbstractMatrix,F,B} <: GNNLayer
    dense_x::DX
    dense_e::DE
    bias::B
    a::A
    σ::F
    negative_slope::T
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout::DV
end

Flux.@layer GATConv
Flux.trainable(l::GATConv) = (; l.dense_x, l.dense_e, l.bias, l.a)

GATConv(ch::Pair{Int, Int}, args...; kws...) = GATConv((ch[1], 0) => ch[2], args...; kws...)

function GATConv(ch::Pair{NTuple{2, Int}, Int}, σ = identity;
                 heads::Int = 1, concat::Bool = true, negative_slope = 0.2,
                 init = glorot_uniform, bias::Bool = true, add_self_loops = true, dropout=0.0)
    (in, ein), out = ch
    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_x = Dense(in, out * heads, bias = false)
    dense_e = ein > 0 ? Dense(ein, out * heads, bias = false) : nothing
    b = bias ? Flux.create_bias(dense_x.weight, true, concat ? out * heads : out) : false
    a = init(ein > 0 ? 3out : 2out, heads)
    negative_slope = convert(Float32, negative_slope)
    GATConv(dense_x, dense_e, b, a, σ, negative_slope, ch, heads, concat, add_self_loops, dropout)
end

(l::GATConv)(g::GNNGraph) = GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))

(l::GATConv)(g, x, e = nothing) = GNNlib.gat_conv(l, g, x, e)

function Base.show(io::IO, l::GATConv)
    (in, ein), out = l.channel
    print(io, "GATConv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end

@doc raw"""
    GATv2Conv(in => out, [σ; heads, concat, init, bias, negative_slope, add_self_loops])
    GATv2Conv((in, ein) => out, ...)


GATv2 attentional layer from the paper [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).

Implements the operation
```math
\mathbf{x}_i' = \sum_{j \in N(i) \cup \{i\}} \alpha_{ij} W_1 \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} = \frac{1}{z_i} \exp(\mathbf{a}^T LeakyReLU(W_2 \mathbf{x}_i + W_1 \mathbf{x}_j))
```
with ``z_i`` a normalization factor.

In case `ein > 0` is given, edge features of dimension `ein` will be expected in the forward pass 
and the attention coefficients will be calculated as  
```math
\alpha_{ij} = \frac{1}{z_i} \exp(\mathbf{a}^T LeakyReLU(W_3 \mathbf{e}_{j\to i} + W_2 \mathbf{x}_i + W_1 \mathbf{x}_j)).
```

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. Default 0 (i.e. no edge features passed in the forward).
- `out`: The dimension of output node features.
- `σ`: Activation function. Default `identity`.
- `bias`: Learn the additive bias if true. Default `true`.
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
- `dropout`: Dropout probability on the normalized attention coefficient. Default `0.0`.

# Examples
```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
ein = 3
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = GATv2Conv((in_channel, ein) => out_channel, add_self_loops = false)

# edge features
e = randn(Float32, ein, length(s))

# forward pass
y = l(g, x, e)    
```
"""
struct GATv2Conv{T, A1, A2, A3, DV, B, C <: AbstractMatrix, F} <: GNNLayer
    dense_i::A1
    dense_j::A2
    dense_e::A3
    bias::B
    a::C
    σ::F
    negative_slope::T
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout::DV
end

Flux.@layer GATv2Conv
Flux.trainable(l::GATv2Conv) = (dense_i = l.dense_i, dense_j = l.dense_j, dense_e = l.dense_e, bias = l.bias, a = l.a)

function GATv2Conv(ch::Pair{Int, Int}, args...; kws...)
    GATv2Conv((ch[1], 0) => ch[2], args...; kws...)
end

function GATv2Conv(ch::Pair{NTuple{2, Int}, Int},
                   σ = identity;
                   heads::Int = 1,
                   concat::Bool = true,
                   negative_slope = 0.2,
                   init = glorot_uniform,
                   bias::Bool = true,
                   add_self_loops = true,
                   dropout=0.0)
    (in, ein), out = ch

    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_i = Dense(in, out * heads; bias = bias, init = init)
    dense_j = Dense(in, out * heads; bias = false, init = init)
    if ein > 0
        dense_e = Dense(ein, out * heads; bias = false, init = init)
    else
        dense_e = nothing
    end
    b = bias ? Flux.create_bias(dense_i.weight, true, concat ? out * heads : out) : false
    a = init(out, heads)
    return GATv2Conv(dense_i, dense_j, dense_e, 
              b, a, σ, negative_slope, ch, heads, concat,
              add_self_loops, dropout)
end

(l::GATv2Conv)(g::GNNGraph) = GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))

(l::GATv2Conv)(g, x, e=nothing) = GNNlib.gatv2_conv(l, g, x, e)

function Base.show(io::IO, l::GATv2Conv)
    (in, ein), out = l.channel
    print(io, "GATv2Conv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end

@doc raw"""
    GatedGraphConv(out, num_layers; aggr=+, init=glorot_uniform)

Gated graph convolution layer from [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).

Implements the recursion
```math
\begin{aligned}
\mathbf{h}^{(0)}_i &= [\mathbf{x}_i; \mathbf{0}] \\
\mathbf{h}^{(l)}_i &= GRU(\mathbf{h}^{(l-1)}_i, \square_{j \in N(i)} W \mathbf{h}^{(l-1)}_j)
\end{aligned}
```

where ``\mathbf{h}^{(l)}_i`` denotes the ``l``-th hidden variables passing through GRU. The dimension of input ``\mathbf{x}_i`` needs to be less or equal to `out`.

# Arguments

- `out`: The dimension of output features.
- `num_layers`: The number of recursion steps.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `init`: Weight initialization function.

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
out_channel = 5
num_layers = 3
g = GNNGraph(s, t)

# create layer
l = GatedGraphConv(out_channel, num_layers)

# forward pass
y = l(g, x)   
```
"""
struct GatedGraphConv{W <: AbstractArray{<:Number, 3}, R, A} <: GNNLayer
    weight::W
    gru::R
    dims::Int
    num_layers::Int
    aggr::A
end

Flux.@layer GatedGraphConv

function GatedGraphConv(dims::Int, num_layers::Int;
                        aggr = +, init = glorot_uniform)
    w = init(dims, dims, num_layers)
    gru = GRUCell(dims => dims)
    GatedGraphConv(w, gru, dims, num_layers, aggr)
end


(l::GatedGraphConv)(g, H) = GNNlib.gated_graph_conv(l, g, H)

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv($(l.dims), $(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

@doc raw"""
    EdgeConv(nn; aggr=max)

Edge convolutional layer from paper [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

Performs the operation
```math
\mathbf{x}_i' = \square_{j \in N(i)}\, nn([\mathbf{x}_i; \mathbf{x}_j - \mathbf{x}_i])
```

where `nn` generally denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `nn`: A (possibly learnable) function. 
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)

# create layer
l = EdgeConv(Dense(2 * in_channel, out_channel), aggr = +)

# forward pass
y = l(g, x)
```
"""
struct EdgeConv{NN, A} <: GNNLayer
    nn::NN
    aggr::A
end

Flux.@layer :expand EdgeConv

EdgeConv(nn; aggr = max) = EdgeConv(nn, aggr)

(l::EdgeConv)(g, x) = GNNlib.edge_conv(l, g, x)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

@doc raw"""
    GINConv(f, ϵ; aggr=+)

Graph Isomorphism convolutional layer from paper [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826.pdf).

Implements the graph convolution
```math
\mathbf{x}_i' = f_\Theta\left((1 + \epsilon) \mathbf{x}_i + \sum_{j \in N(i)} \mathbf{x}_j \right)
```
where ``f_\Theta`` typically denotes a learnable function, e.g. a linear layer or a multi-layer perceptron.

# Arguments

- `f`: A (possibly learnable) function acting on node features. 
- `ϵ`: Weighting factor.

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)

# create dense layer
nn = Dense(in_channel, out_channel)

# create layer
l = GINConv(nn, 0.01f0, aggr = mean)

# forward pass
y = l(g, x)  
```
"""
struct GINConv{R <: Real, NN, A} <: GNNLayer
    nn::NN
    ϵ::R
    aggr::A
end

Flux.@layer :expand GINConv
Flux.trainable(l::GINConv) = (nn = l.nn,)

GINConv(nn, ϵ; aggr = +) = GINConv(nn, ϵ, aggr)

(l::GINConv)(g, x) = GNNlib.gin_conv(l, g, x)

function Base.show(io::IO, l::GINConv)
    print(io, "GINConv($(l.nn)")
    print(io, ", $(l.ϵ)")
    print(io, ")")
end

@doc raw"""
    NNConv(in => out, f, σ=identity; aggr=+, bias=true, init=glorot_uniform)

The continuous kernel-based convolutional operator from the 
[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) paper. 
This convolution is also known as the edge-conditioned convolution from the 
[Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs](https://arxiv.org/abs/1704.02901) paper.

Performs the operation

```math
\mathbf{x}_i' = W \mathbf{x}_i + \square_{j \in N(i)} f_\Theta(\mathbf{e}_{j\to i})\,\mathbf{x}_j
```

where ``f_\Theta``  denotes a learnable function (e.g. a linear layer or a multi-layer perceptron).
Given an input of batched edge features `e` of size `(num_edge_features, num_edges)`, 
the function `f` will return an batched matrices array whose size is `(out, in, num_edges)`.
For convenience, also functions returning a single `(out*in, num_edges)` matrix are allowed.

# Arguments

- `in`: The dimension of input node features.
- `out`: The dimension of output node features.
- `f`: A (possibly learnable) function acting on edge features.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `σ`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples:

```julia
n_in = 3
n_in_edge = 10
n_out = 5

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)

# create dense layer
nn = Dense(n_in_edge => n_out * n_in)

# create layer
l = NNConv(n_in => n_out, nn, tanh, bias = true, aggr = +)

x = randn(Float32, n_in, g.num_nodes)
e = randn(Float32, n_in_edge, g.num_edges)

# forward pass
y = l(g, x, e)  
```
"""
struct NNConv{W, B, NN, F, A} <: GNNLayer
    weight::W
    bias::B
    nn::NN
    σ::F
    aggr::A
end

Flux.@layer :expand NNConv

function NNConv(ch::Pair{Int, Int}, nn, σ = identity; aggr = +, bias = true,
                init = glorot_uniform)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    return NNConv(W, b, nn, σ, aggr)
end

(l::NNConv)(g, x, e) = GNNlib.nn_conv(l, g, x, e)

(l::NNConv)(g::GNNGraph) = GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))

function Base.show(io::IO, l::NNConv)
    out, in = size(l.weight)
    print(io, "NNConv($in => $out")
    print(io, ", ", l.nn)
    l.σ == identity || print(io, ", ", l.σ)
    (l.aggr == +) || print(io, "; aggr=", l.aggr)
    print(io, ")")
end

@doc raw"""
    SAGEConv(in => out, σ=identity; aggr=mean, bias=true, init=glorot_uniform)

GraphSAGE convolution layer from paper [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf).

Performs:
```math
\mathbf{x}_i' = W \cdot [\mathbf{x}_i; \square_{j \in \mathcal{N}(i)} \mathbf{x}_j]
```

where the aggregation type is selected by `aggr`.

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `σ`: Activation function.
- `aggr`: Aggregation operator for the incoming messages (e.g. `+`, `*`, `max`, `min`, and `mean`).
- `bias`: Add learnable bias.
- `init`: Weights' initializer.

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)

# create layer
l = SAGEConv(in_channel => out_channel, tanh, bias = false, aggr = +)

# forward pass
y = l(g, x)   
```
"""
struct SAGEConv{W <: AbstractMatrix, B, F, A} <: GNNLayer
    weight::W
    bias::B
    σ::F
    aggr::A
end

Flux.@layer SAGEConv

function SAGEConv(ch::Pair{Int, Int}, σ = identity; aggr = mean,
                  init = glorot_uniform, bias::Bool = true)
    in, out = ch
    W = init(out, 2 * in)
    b = bias ? Flux.create_bias(W, true, out) : false
    SAGEConv(W, b, σ, aggr)
end

(l::SAGEConv)(g, x) = GNNlib.sage_conv(l, g, x)

function Base.show(io::IO, l::SAGEConv)
    out_channel, in_channel = size(l.weight)
    print(io, "SAGEConv(", in_channel ÷ 2, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

@doc raw"""
    ResGatedGraphConv(in => out, act=identity; init=glorot_uniform, bias=true)

The residual gated graph convolutional operator from the [Residual Gated Graph ConvNets](https://arxiv.org/abs/1711.07553) paper.

The layer's forward pass is given by

```math
\mathbf{x}_i' = act\big(U\mathbf{x}_i + \sum_{j \in N(i)} \eta_{ij} V \mathbf{x}_j\big),
```
where the edge gates ``\eta_{ij}`` are given by

```math
\eta_{ij} = sigmoid(A\mathbf{x}_i + B\mathbf{x}_j).
```

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `act`: Activation function.
- `init`: Weight matrices' initializing function. 
- `bias`: Learn an additive bias if true.

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)

# create layer
l = ResGatedGraphConv(in_channel => out_channel, tanh, bias = true)

# forward pass
y = l(g, x)  
```
"""
struct ResGatedGraphConv{W, B, F} <: GNNLayer
    A::W
    B::W
    U::W
    V::W
    bias::B
    σ::F
end

Flux.@layer ResGatedGraphConv

function ResGatedGraphConv(ch::Pair{Int, Int}, σ = identity;
                           init = glorot_uniform, bias::Bool = true)
    in, out = ch
    A = init(out, in)
    B = init(out, in)
    U = init(out, in)
    V = init(out, in)
    b = bias ? Flux.create_bias(A, true, out) : false
    return ResGatedGraphConv(A, B, U, V, b, σ)
end

(l::ResGatedGraphConv)(g, x) = GNNlib.res_gated_graph_conv(l, g, x)

function Base.show(io::IO, l::ResGatedGraphConv)
    out_channel, in_channel = size(l.A)
    print(io, "ResGatedGraphConv(", in_channel, " => ", out_channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end

@doc raw"""
    CGConv((in, ein) => out, act=identity; bias=true, init=glorot_uniform, residual=false)
    CGConv(in => out, ...)

The crystal graph convolutional layer from the paper
[Crystal Graph Convolutional Neural Networks for an Accurate and
Interpretable Prediction of Material Properties](https://arxiv.org/pdf/1710.10324.pdf).
Performs the operation

```math
\mathbf{x}_i' = \mathbf{x}_i + \sum_{j\in N(i)}\sigma(W_f \mathbf{z}_{ij} + \mathbf{b}_f)\, act(W_s \mathbf{z}_{ij} + \mathbf{b}_s)
```

where ``\mathbf{z}_{ij}``  is the node and edge features concatenation 
``[\mathbf{x}_i; \mathbf{x}_j; \mathbf{e}_{j\to i}]`` 
and ``\sigma`` is the sigmoid function.
The residual ``\mathbf{x}_i`` is added only if `residual=true` and the output size is the same 
as the input size.

# Arguments

- `in`: The dimension of input node features.
- `ein`: The dimension of input edge features. 
If `ein` is not given, assumes that no edge features are passed as input in the forward pass.
- `out`: The dimension of output node features.
- `act`: Activation function.
- `bias`: Add learnable bias.
- `init`: Weights' initializer.
- `residual`: Add a residual connection.

# Examples 

```julia
g = rand_graph(5, 6)
x = rand(Float32, 2, g.num_nodes)
e = rand(Float32, 3, g.num_edges)

l = CGConv((2, 3) => 4, tanh)
y = l(g, x, e)    # size: (4, num_nodes)

# No edge features
l = CGConv(2 => 4, tanh)
y = l(g, x)    # size: (4, num_nodes)
```
"""
struct CGConv{D1, D2} <: GNNLayer
    ch::Pair{NTuple{2, Int}, Int}
    dense_f::D1
    dense_s::D2
    residual::Bool
end

Flux.@layer CGConv

CGConv(ch::Pair{Int, Int}, args...; kws...) = CGConv((ch[1], 0) => ch[2], args...; kws...)

function CGConv(ch::Pair{NTuple{2, Int}, Int}, act = identity; residual = false,
                bias = true, init = glorot_uniform)
    (nin, ein), out = ch
    dense_f = Dense(2nin + ein, out, sigmoid; bias, init)
    dense_s = Dense(2nin + ein, out, act; bias, init)
    return CGConv(ch, dense_f, dense_s, residual)
end

(l::CGConv)(g, x, e = nothing) = GNNlib.cg_conv(l, g, x, e)


(l::CGConv)(g::GNNGraph) = GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))

function Base.show(io::IO, l::CGConv)
    print(io, "CGConv($(l.ch)")
    l.dense_s.σ == identity || print(io, ", ", l.dense_s.σ)
    print(io, ", residual=$(l.residual)")
    print(io, ")")
end

@doc raw"""
    AGNNConv(; init_beta=1.0f0, trainable=true, add_self_loops=true)

Attention-based Graph Neural Network layer from paper [Attention-based
Graph Neural Network for Semi-Supervised Learning](https://arxiv.org/abs/1803.03735).

The forward pass is given by
```math
\mathbf{x}_i' = \sum_{j \in N(i)} \alpha_{ij} \mathbf{x}_j
```
where the attention coefficients ``\alpha_{ij}`` are given by
```math
\alpha_{ij} =\frac{e^{\beta \cos(\mathbf{x}_i, \mathbf{x}_j)}}
                  {\sum_{j'}e^{\beta \cos(\mathbf{x}_i, \mathbf{x}_{j'})}}
```
with the cosine distance defined by
```math 
\cos(\mathbf{x}_i, \mathbf{x}_j) = 
  \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\lVert\mathbf{x}_i\rVert \lVert\mathbf{x}_j\rVert}
```
and ``\beta`` a trainable parameter if `trainable=true`.

# Arguments

- `init_beta`: The initial value of ``\beta``. Default 1.0f0.
- `trainable`: If true, ``\beta`` is trainable. Default `true`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.

# Examples:

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)

# create layer
l = AGNNConv(init_beta=2.0f0)

# forward pass
y = l(g, x)   
```
"""
struct AGNNConv{A <: AbstractVector} <: GNNLayer
    β::A
    add_self_loops::Bool
    trainable::Bool
end

Flux.@layer AGNNConv

Flux.trainable(l::AGNNConv) = l.trainable ? (; l.β) : (;)

function AGNNConv(; init_beta = 1.0f0, add_self_loops = true, trainable = true)
    AGNNConv([init_beta], add_self_loops, trainable)
end

(l::AGNNConv)(g, x) = GNNlib.agnn_conv(l, g, x)

@doc raw"""
    MEGNetConv(ϕe, ϕv; aggr=mean)
    MEGNetConv(in => out; aggr=mean)

Convolution from [Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://arxiv.org/pdf/1812.05055.pdf)
paper. In the forward pass, takes as inputs node features `x` and edge features `e` and returns
updated features `x'` and `e'` according to 

```math
\begin{aligned}
\mathbf{e}_{i\to j}'  = \phi_e([\mathbf{x}_i;\,  \mathbf{x}_j;\,  \mathbf{e}_{i\to j}]),\\
\mathbf{x}_{i}'  = \phi_v([\mathbf{x}_i;\, \square_{j\in \mathcal{N}(i)}\,\mathbf{e}_{j\to i}']).
\end{aligned}
```

`aggr` defines the aggregation to be performed.

If the neural networks `ϕe` and  `ϕv` are not provided, they will be constructed from
the `in` and `out` arguments instead as multi-layer perceptron with one hidden layer and `relu` 
activations.

# Examples

```julia
g = rand_graph(10, 30)
x = randn(Float32, 3, 10)
e = randn(Float32, 3, 30)
m = MEGNetConv(3 => 3)
x′, e′ = m(g, x, e)
```
"""
struct MEGNetConv{TE, TV, A} <: GNNLayer
    ϕe::TE
    ϕv::TV
    aggr::A
end

Flux.@layer :expand MEGNetConv

MEGNetConv(ϕe, ϕv; aggr = mean) = MEGNetConv(ϕe, ϕv, aggr)

function MEGNetConv(ch::Pair{Int, Int}; aggr = mean)
    nin, nout = ch
    ϕe = Chain(Dense(3nin, nout, relu),
               Dense(nout, nout))

    ϕv = Chain(Dense(nin + nout, nout, relu),
               Dense(nout, nout))

    return MEGNetConv(ϕe, ϕv; aggr)
end

function (l::MEGNetConv)(g::GNNGraph)
    x, e = l(g, g.ndata.x, g.edata.e)
    return GNNGraph(g, ndata = x, edata = e)
end

(l::MEGNetConv)(g, x, e) = GNNlib.megnet_conv(l, g, x, e)

@doc raw"""
    GMMConv((in, ein) => out, σ=identity; K=1, bias=true, init=glorot_uniform, residual=false)

Graph mixture model convolution layer from the paper [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402)
Performs the operation
```math
\mathbf{x}_i' = \mathbf{x}_i + \frac{1}{|N(i)|} \sum_{j\in N(i)}\frac{1}{K}\sum_{k=1}^K \mathbf{w}_k(\mathbf{e}_{j\to i}) \odot \Theta_k \mathbf{x}_j
```
where ``w^a_{k}(e^a)`` for feature `a` and kernel `k` is given by
```math
w^a_{k}(e^a) = \exp(-\frac{1}{2}(e^a - \mu^a_k)^T (\Sigma^{-1})^a_k(e^a - \mu^a_k))
```
``\Theta_k, \mu^a_k, (\Sigma^{-1})^a_k`` are learnable parameters.

The input to the layer is a node feature array `x` of size `(num_features, num_nodes)` and
edge pseudo-coordinate array `e` of size `(num_features, num_edges)`
The residual ``\mathbf{x}_i`` is added only if `residual=true` and the output size is the same 
as the input size.

# Arguments 

- `in`: Number of input node features.
- `ein`: Number of input edge features.
- `out`: Number of output features.
- `σ`: Activation function. Default `identity`.
- `K`: Number of kernels. Default `1`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `residual`: Residual conncetion. Default `false`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s,t)
nin, ein, out, K = 4, 10, 7, 8 
x = randn(Float32, nin, g.num_nodes)
e = randn(Float32, ein, g.num_edges)

# create layer
l = GMMConv((nin, ein) => out, K=K)

# forward pass
l(g, x, e)
```
"""
struct GMMConv{A <: AbstractMatrix, B, F} <: GNNLayer
    mu::A
    sigma_inv::A
    bias::B
    σ::F
    ch::Pair{NTuple{2, Int}, Int}
    K::Int
    dense_x::Dense
    residual::Bool
end

Flux.@layer GMMConv

function GMMConv(ch::Pair{NTuple{2, Int}, Int},
                 σ = identity;
                 K::Int = 1,
                 bias::Bool = true,
                 init = Flux.glorot_uniform,
                 residual = false)
    (nin, ein), out = ch
    mu = init(ein, K)
    sigma_inv = init(ein, K)
    b = bias ? Flux.create_bias(mu, true, out) : false
    dense_x = Dense(nin, out * K, bias = false)
    GMMConv(mu, sigma_inv, b, σ, ch, K, dense_x, residual)
end

(l::GMMConv)(g::GNNGraph, x, e) = GNNlib.gmm_conv(l, g, x, e)

(l::GMMConv)(g::GNNGraph) = GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))

function Base.show(io::IO, l::GMMConv)
    (nin, ein), out = l.ch
    print(io, "GMMConv((", nin, ",", ein, ")=>", out)
    l.σ == identity || print(io, ", σ=", l.dense_s.σ)
    print(io, ", K=", l.K)
    print(io, ", residual=", l.residual)
    print(io, ")")
end

@doc raw"""
    SGConv(int => out, k=1; [bias, init, add_self_loops, use_edge_weight])
                                
SGC layer from [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153.pdf)
Performs operation
```math
H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta
```
where ``\tilde{A}`` is ``A + I``.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k` : Number of hops k. Default `1`.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. Default `false`.

# Examples

```julia
# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(Float32, 3, g.num_nodes)

# create layer
l = SGConv(3 => 5; add_self_loops = true) 

# forward pass
y = l(g, x)       # size:  5 × num_nodes

# convolution with edge weights
w = [1.1, 0.1, 2.3, 0.5]
y = l(g, x, w)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = SGConv(3 => 5, add_self_loops = true, use_edge_weight=true) 
y = l(g, x) # same as l(g, x, w) 
```
"""
struct SGConv{A <: AbstractMatrix, B} <: GNNLayer
    weight::A
    bias::B
    k::Int
    add_self_loops::Bool
    use_edge_weight::Bool
end

Flux.@layer SGConv

function SGConv(ch::Pair{Int, Int}, k = 1;
                init = glorot_uniform,
                bias::Bool = true,
                add_self_loops = true,
                use_edge_weight = false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    return SGConv(W, b, k, add_self_loops, use_edge_weight)
end

(l::SGConv)(g, x, edge_weight = nothing) = GNNlib.sg_conv(l, g, x, edge_weight)

function Base.show(io::IO, l::SGConv)
    out, in = size(l.weight)
    print(io, "SGConv($in => $out")
    l.k == 1 || print(io, ", ", l.k)
    print(io, ")")
end

@doc raw"""
    TAGConv(in => out, k=3; bias=true, init=glorot_uniform, add_self_loops=true, use_edge_weight=false)

TAGConv layer from [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/pdf/1710.10370.pdf).
This layer extends the idea of graph convolutions by applying filters that adapt to the topology of the data. 
It performs the operation:

```math
H^{K} = {\sum}_{k=0}^K (D^{-1/2} A D^{-1/2})^{k} X {\Theta}_{k}
```

where `A` is the adjacency matrix of the graph, `D` is the degree matrix, `X` is the input feature matrix, and ``{\Theta}_{k}`` is a unique weight matrix for each hop `k`.

# Arguments
- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Maximum number of hops to consider. Default is `3`.
- `bias`: Whether to include a learnable bias term. Default is `true`.
- `init`: Initialization function for the weights. Default is `glorot_uniform`.
- `add_self_loops`: Whether to add self-loops to the adjacency matrix. Default is `true`.
- `use_edge_weight`: If `true`, edge weights are considered in the computation (if available). Default is `false`.

# Examples

```julia
# Example graph data
s = [1, 1, 2, 3]
t = [2, 3, 1, 1]
g = GNNGraph(s, t)  # Create a graph
x = randn(Float32, 3, g.num_nodes)  # Random features for each node

# Create a TAGConv layer
l = TAGConv(3 => 5, k=3; add_self_loops=true)

# Apply the TAGConv layer
y = l(g, x)  # Output size: 5 × num_nodes
```
"""
struct TAGConv{A <: AbstractMatrix, B} <: GNNLayer
    weight::A
    bias::B
    k::Int
    add_self_loops::Bool
    use_edge_weight::Bool
end

Flux.@layer TAGConv

function TAGConv(ch::Pair{Int, Int}, k = 3;
                  init = glorot_uniform,
                  bias::Bool = true,
                  add_self_loops = true,
                  use_edge_weight = false)
    in, out = ch
    W = init(out, in)
    b = bias ? Flux.create_bias(W, true, out) : false
    return TAGConv(W, b, k, add_self_loops, use_edge_weight)
end

(l::TAGConv)(g, x, edge_weight = nothing) = GNNlib.tag_conv(l, g, x, edge_weight)

function Base.show(io::IO, l::TAGConv)
    out, in = size(l.weight)
    print(io, "TAGConv($in => $out")
    l.k == 1 || print(io, ", ", l.k)
    print(io, ")")
end

@doc raw"""
    EGNNConv((in, ein) => out; hidden_size=2in, residual=false)
    EGNNConv(in => out; hidden_size=2in, residual=false)

Equivariant Graph Convolutional Layer from [E(n) Equivariant Graph
Neural Networks](https://arxiv.org/abs/2102.09844).

The layer performs the following operation:

```math
\begin{aligned}
\mathbf{m}_{j\to i} &=\phi_e(\mathbf{h}_i, \mathbf{h}_j, \lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2, \mathbf{e}_{j\to i}),\\
\mathbf{x}_i' &= \mathbf{x}_i + C_i\sum_{j\in\mathcal{N}(i)}(\mathbf{x}_i-\mathbf{x}_j)\phi_x(\mathbf{m}_{j\to i}),\\
\mathbf{m}_i &= C_i\sum_{j\in\mathcal{N}(i)} \mathbf{m}_{j\to i},\\
\mathbf{h}_i' &= \mathbf{h}_i + \phi_h(\mathbf{h}_i, \mathbf{m}_i)
\end{aligned}
```
where ``\mathbf{h}_i``, ``\mathbf{x}_i``, ``\mathbf{e}_{j\to i}`` are invariant node features, equivariant node
features, and edge features respectively. ``\phi_e``, ``\phi_h``, and
``\phi_x`` are two-layer MLPs. `C` is a constant for normalization,
computed as ``1/|\mathcal{N}(i)|``.


# Constructor Arguments

- `in`: Number of input features for `h`.
- `out`: Number of output features for `h`.
- `ein`: Number of input edge features.
- `hidden_size`: Hidden representation size.
- `residual`: If `true`, add a residual connection. Only possible if `in == out`. Default `false`.

# Forward Pass 

    l(g, x, h, e=nothing)
                     
## Forward Pass Arguments:

- `g` : The graph.
- `x` : Matrix of equivariant node coordinates.
- `h` : Matrix of invariant node features.
- `e` : Matrix of invariant edge features. Default `nothing`.

Returns updated `h` and `x`.

# Examples

```julia
g = rand_graph(10, 10)
h = randn(Float32, 5, g.num_nodes)
x = randn(Float32, 3, g.num_nodes)
egnn = EGNNConv(5 => 6, 10)
hnew, xnew = egnn(g, h, x)
```
"""
struct EGNNConv{TE, TX, TH, NF} <: GNNLayer
    ϕe::TE
    ϕx::TX
    ϕh::TH
    num_features::NF
    residual::Bool
end

Flux.@layer EGNNConv

function EGNNConv(ch::Pair{Int, Int}, hidden_size = 2 * ch[1]; residual = false)
    return EGNNConv((ch[1], 0) => ch[2]; hidden_size, residual)
end

#Follows reference implementation at https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
function EGNNConv(ch::Pair{NTuple{2, Int}, Int}; hidden_size::Int = 2 * ch[1][1],
                  residual = false)
    (in_size, edge_feat_size), out_size = ch
    act_fn = swish

    # +1 for the radial feature: ||x_i - x_j||^2
    ϕe = Chain(Dense(in_size * 2 + edge_feat_size + 1 => hidden_size, act_fn),
               Dense(hidden_size => hidden_size, act_fn))

    ϕh = Chain(Dense(in_size + hidden_size, hidden_size, swish),
               Dense(hidden_size, out_size))

    ϕx = Chain(Dense(hidden_size, hidden_size, swish),
               Dense(hidden_size, 1, bias = false))

    num_features = (in = in_size, edge = edge_feat_size, out = out_size,
                    hidden = hidden_size)
    if residual
        @assert in_size==out_size "Residual connection only possible if in_size == out_size"
    end
    return EGNNConv(ϕe, ϕx, ϕh, num_features, residual)
end

(l::EGNNConv)(g, h, x, e = nothing) = GNNlib.egnn_conv(l, g, h, x, e)

function Base.show(io::IO, l::EGNNConv)
    ne = l.num_features.edge
    nin = l.num_features.in
    nout = l.num_features.out
    nh = l.num_features.hidden
    print(io, "EGNNConv(($nin, $ne) => $nout; hidden_size=$nh")
    if l.residual
        print(io, ", residual=true")
    end
    print(io, ")")
end

@doc raw"""
    TransformerConv((in, ein) => out; [heads, concat, init, add_self_loops, bias_qkv,
        bias_root, root_weight, gating, skip_connection, batch_norm, ff_channels]))

The transformer-like multi head attention convolutional operator from the 
[Masked Label Prediction: Unified Message Passing Model for Semi-Supervised 
Classification](https://arxiv.org/abs/2009.03509) paper, which also considers 
edge features.
It further contains options to also be configured as the transformer-like convolutional operator from the 
[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1706.03762) paper,
including a successive feed-forward network as well as skip layers and batch normalization.

The layer's basic forward pass is given by
```math
x_i' = W_1x_i + \sum_{j\in N(i)} \alpha_{ij} (W_2 x_j + W_6e_{ij})
```
where the attention scores are
```math
\alpha_{ij} = \mathrm{softmax}\left(\frac{(W_3x_i)^T(W_4x_j+
W_6e_{ij})}{\sqrt{d}}\right).
```

Optionally, a combination of the aggregated value with transformed root node features 
by a gating mechanism via
```math
x'_i = \beta_i W_1 x_i + (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
\alpha_{i,j} W_2 x_j \right)}_{=m_i}
```
with
```math
\beta_i = \textrm{sigmoid}(W_5^{\top} [ W_1 x_i, m_i, W_1 x_i - m_i ]).
```
can be performed.

# Arguments 

- `in`: Dimension of input features, which also corresponds to the dimension of 
    the output features.
- `ein`: Dimension of the edge features; if 0, no edge features will be used.
- `out`: Dimension of the output.
- `heads`: Number of heads in output. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged
    over the heads. Default `true`.
- `init`: Weight matrices' initializing function. Default `glorot_uniform`.
- `add_self_loops`: Add self loops to the input graph. Default `false`.
- `bias_qkv`: If set, bias is used in the key, query and value transformations for nodes.
    Default `true`.
- `bias_root`: If set, the layer will also learn an additive bias for the root when root 
    weight is used. Default `true`.
- `root_weight`: If set, the layer will add the transformed root node features
    to the output. Default `true`.
- `gating`: If set, will combine aggregation and transformed root node features by a
    gating mechanism. Default `false`.
- `skip_connection`: If set, a skip connection will be made from the input and 
    added to the output. Default `false`.
- `batch_norm`: If set, a batch normalization will be applied to the output. Default `false`.
- `ff_channels`: If positive, a feed-forward NN is appended, with the first having the given
    number of hidden nodes; this NN also gets a skip connection and batch normalization 
    if the respective parameters are set. Default: `0`.

# Examples

```julia
N, in_channel, out_channel = 4, 3, 5
ein, heads = 2, 3
g = GNNGraph([1,1,2,4], [2,3,1,1])
l = TransformerConv((in_channel, ein) => in_channel; heads, gating = true, bias_qkv = true)
x = rand(Float32, in_channel, N)
e = rand(Float32, ein, g.num_edges)
l(g, x, e)
```        
"""
struct TransformerConv{TW1, TW2, TW3, TW4, TW5, TW6, TFF, TBN1, TBN2} <: GNNLayer
    W1::TW1
    W2::TW2
    W3::TW3
    W4::TW4
    W5::TW5
    W6::TW6
    FF::TFF
    BN1::TBN1
    BN2::TBN2
    channels::Pair{NTuple{2, Int}, Int}
    heads::Int
    add_self_loops::Bool
    concat::Bool
    skip_connection::Bool
    sqrt_out::Float32
end

Flux.@layer TransformerConv

function Flux.trainable(l::TransformerConv)
    (; l.W1, l.W2, l.W3, l.W4, l.W5, l.W6, l.FF, l.BN1, l.BN2)
end

function TransformerConv(ch::Pair{Int, Int}, args...; kws...)
    TransformerConv((ch[1], 0) => ch[2], args...; kws...)
end

function TransformerConv(ch::Pair{NTuple{2, Int}, Int};
                         heads::Int = 1,
                         concat::Bool = true,
                         init = glorot_uniform,
                         add_self_loops::Bool = false,
                         bias_qkv = true,
                         bias_root::Bool = true,
                         root_weight::Bool = true,
                         gating::Bool = false,
                         skip_connection::Bool = false,
                         batch_norm::Bool = false,
                         ff_channels::Int = 0)
    (in, ein), out = ch

    if add_self_loops
        @assert iszero(ein) "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    W1 = root_weight ?
         Dense(in, out * (concat ? heads : 1); bias = bias_root, init = init) : nothing
    W2 = Dense(in => out * heads; bias = bias_qkv, init = init)
    W3 = Dense(in => out * heads; bias = bias_qkv, init = init)
    W4 = Dense(in => out * heads; bias = bias_qkv, init = init)
    out_mha = out * (concat ? heads : 1)
    W5 = gating ? Dense(3 * out_mha => 1, sigmoid; bias = false, init = init) : nothing
    W6 = ein > 0 ? Dense(ein => out * heads; bias = bias_qkv, init = init) : nothing
    FF = ff_channels > 0 ?
         Chain(Dense(out_mha => ff_channels, relu),
               Dense(ff_channels => out_mha)) : nothing
    BN1 = batch_norm ? BatchNorm(out_mha) : nothing
    BN2 = (batch_norm && ff_channels > 0) ? BatchNorm(out_mha) : nothing

    return TransformerConv(W1, W2, W3, W4, W5, W6, FF, BN1, BN2,
                           ch, heads, add_self_loops, concat, skip_connection,
                           Float32(√out))
end

(l::TransformerConv)(g, x, e = nothing) = GNNlib.transformer_conv(l, g, x, e)

function (l::TransformerConv)(g::GNNGraph)
    GNNGraph(g, ndata = l(g, g.ndata.x, g.edata.e))
end

function Base.show(io::IO, l::TransformerConv)
    (in, ein), out = l.channels
    print(io, "TransformerConv(($in, $ein) => $out, heads=$(l.heads))")
end

"""
    DConv(ch::Pair{Int, Int}, k::Int; init = glorot_uniform, bias = true)

Diffusion convolution layer from the paper [Diffusion Convolutional Recurrent Neural Networks: Data-Driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926).

# Arguments

- `ch`: Pair of input and output dimensions.
- `k`: Number of diffusion steps.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `bias`: Add learnable bias. Default `true`.

# Examples
```
julia> g = GNNGraph(rand(10, 10), ndata = rand(Float32, 2, 10));

julia> dconv = DConv(2 => 4, 4)
DConv(2 => 4, 4)

julia> y = dconv(g, g.ndata.x);

julia> size(y)
(4, 10)
```
"""
struct DConv <: GNNLayer
    in::Int
    out::Int
    weights::AbstractArray
    bias::AbstractArray
    k::Int
end

Flux.@layer DConv

function DConv(ch::Pair{Int, Int}, k::Int; init = glorot_uniform, bias = true)
    in, out = ch
    weights = init(2, k, out, in)
    b = bias ? Flux.create_bias(weights, true, out) : false
    return DConv(in, out, weights, b, k)
end

(l::DConv)(g, x) = GNNlib.d_conv(l, g, x)

function Base.show(io::IO, l::DConv)
    print(io, "DConv($(l.in) => $(l.out), $(l.k))")
end
