_getbias(ps) = hasproperty(ps, :bias) ? getproperty(ps, :bias) : false
_getstate(st, name) = hasproperty(st, name) ? getproperty(st, name) : NamedTuple()
_getstate(s::StatefulLuxLayer{Val{true}}) = s.st
_getstate(s::StatefulLuxLayer{Static.True}) = s.st
_getstate(s::StatefulLuxLayer{Val{false}}) = s.st_any
_getstate(s::StatefulLuxLayer{Static.False}) = s.st_any

@doc raw"""
    GCNConv(in => out, σ=identity; [init_weight, init_bias, use_bias, add_self_loops, use_edge_weight])

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

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `σ`: Activation function. Default `identity`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.

# Forward

    (::GCNConv)(g, x, [edge_weight], ps, st; norm_fn = d -> 1 ./ sqrt.(d), conv_weight=nothing)

Takes as input a graph `g`, a node feature matrix `x` of size `[in, num_nodes]`, optionally an edge weight vector and the parameter and state of the layer. Returns a node feature matrix of size 
`[out, num_nodes]`.

The `norm_fn` parameter allows for custom normalization of the graph convolution operation by passing a function as argument. 
By default, it computes ``\frac{1}{\sqrt{d}}`` i.e the inverse square root of the degree (`d`) of each node in the graph. 
If `conv_weight` is an `AbstractMatrix` of size `[out, in]`, then the convolution is performed using that weight matrix.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = GCNConv(3 => 5) 

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y = l(g, x, ps, st)       # size of the output first entry:  5 × num_nodes

# convolution with edge weights and custom normalization function
w = [1.1, 0.1, 2.3, 0.5]
custom_norm_fn(d) = 1 ./ sqrt.(d + 1)  # Custom normalization function
y = l(g, x, w, ps, st; norm_fn = custom_norm_fn)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = GCNConv(3 => 5, use_edge_weight=true)
ps, st = Lux.setup(rng, l)
y = l(g, x, ps, st) # same as l(g, x, w) 
```
"""
@concrete struct GCNConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    add_self_loops::Bool
    use_edge_weight::Bool
    init_weight
    init_bias
    σ
end

function GCNConv(ch::Pair{Int, Int}, σ = identity;
                init_weight = glorot_uniform,
                init_bias = zeros32,
                use_bias::Bool = true,
                add_self_loops::Bool = true,
                use_edge_weight::Bool = false)
    in_dims, out_dims = ch
    σ = NNlib.fast_act(σ)
    return GCNConv(in_dims, out_dims, use_bias, add_self_loops, use_edge_weight, init_weight, init_bias, σ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GCNConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::GCNConv) = l.use_bias ? l.in_dims * l.out_dims + l.out_dims : l.in_dims * l.out_dims
LuxCore.outputsize(d::GCNConv) = (d.out_dims,)

function Base.show(io::IO, l::GCNConv)
    print(io, "GCNConv(", l.in_dims, " => ", l.out_dims)
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    l.add_self_loops || print(io, ", add_self_loops=false")
    !l.use_edge_weight || print(io, ", use_edge_weight=true")
    print(io, ")")
end

(l::GCNConv)(g, x, ps, st; conv_weight=nothing, edge_weight=nothing, norm_fn= d -> 1 ./ sqrt.(d)) = 
    l(g, x, edge_weight, ps, st; conv_weight, norm_fn)

function (l::GCNConv)(g, x, edge_weight, ps, st; 
            norm_fn = d -> 1 ./ sqrt.(d),
            conv_weight=nothing)

    m = (; ps.weight, bias = _getbias(ps), 
           l.add_self_loops, l.use_edge_weight, l.σ)
    y = GNNlib.gcn_conv(m, g, x, edge_weight, norm_fn, conv_weight)
    return y, st
end
@doc raw"""
    ChebConv(in => out, k; init_weight = glorot_uniform, init_bias = zeros32, use_bias = true)

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
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = ChebConv(3 => 5, 5)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size of the output y:  5 × num_nodes
```
"""
@concrete struct ChebConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    k::Int
    init_weight
    init_bias
end

function ChebConv(ch::Pair{Int, Int}, k::Int;
                  init_weight = glorot_uniform,
                  init_bias = zeros32,
                  use_bias::Bool = true)
    in_dims, out_dims = ch
    return ChebConv(in_dims, out_dims, use_bias, k, init_weight, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ChebConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims, l.k)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::ChebConv) = l.use_bias ? l.in_dims * l.out_dims * l.k + l.out_dims : 
                                                    l.in_dims * l.out_dims * l.k
LuxCore.statelength(d::ChebConv) = 0
LuxCore.outputsize(d::ChebConv) = (d.out_dims,)

function Base.show(io::IO, l::ChebConv)
    print(io, "ChebConv(", l.in_dims, " => ", l.out_dims, ", k=", l.k)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

function (l::ChebConv)(g, x, ps, st)
    m = (; ps.weight, bias = _getbias(ps), l.k)
    y = GNNlib.cheb_conv(m, g, x)
    return y, st

end
@doc raw"""
    GraphConv(in => out, σ = identity; aggr = +, init_weight = glorot_uniform,init_bias = zeros32, use_bias = true)

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
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = GraphConv(in_channel => out_channel, relu, use_bias = false, aggr = mean)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size of the output y:  5 × num_nodes
```
"""
@concrete struct GraphConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight
    init_bias
    σ
    aggr
end

function GraphConv(ch::Pair{Int, Int}, σ = identity;
            aggr = +,
            init_weight = glorot_uniform,
            init_bias = zeros32, 
            use_bias::Bool = true)
    in_dims, out_dims = ch
    σ = NNlib.fast_act(σ)
    return GraphConv(in_dims, out_dims, use_bias, init_weight, init_bias, σ, aggr)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GraphConv)
    weight1 = l.init_weight(rng, l.out_dims, l.in_dims)
    weight2 = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight1, weight2, bias)
    else
        return (; weight1, weight2)
    end
end

function LuxCore.parameterlength(l::GraphConv)
    if l.use_bias
        return 2 * l.in_dims * l.out_dims + l.out_dims
    else
        return 2 * l.in_dims * l.out_dims
    end
end

LuxCore.statelength(d::GraphConv) = 0
LuxCore.outputsize(d::GraphConv) = (d.out_dims,)

function Base.show(io::IO, l::GraphConv)
    print(io, "GraphConv(", l.in_dims, " => ", l.out_dims)
    (l.σ == identity) || print(io, ", ", l.σ)
    (l.aggr == +) || print(io, ", aggr=", l.aggr)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

function (l::GraphConv)(g, x, ps, st)
    m = (; ps.weight1, ps.weight2, bias = _getbias(ps), 
          l.σ, l.aggr)
    return GNNlib.graph_conv(m, g, x), st
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
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)

# create layer
l = AGNNConv(init_beta=2.0f0)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)   
```
"""
@concrete struct AGNNConv <: GNNLayer
    init_beta <: AbstractVector
    add_self_loops::Bool
    trainable::Bool
end

function AGNNConv(; init_beta = 1.0f0, add_self_loops = true, trainable = true)
    return AGNNConv([init_beta], add_self_loops, trainable)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::AGNNConv)
    if l.trainable
        return (; β = l.init_beta)
    else
        return (;)
    end
end

LuxCore.parameterlength(l::AGNNConv) = l.trainable ? 1 : 0
LuxCore.statelength(d::AGNNConv) = 0

function Base.show(io::IO, l::AGNNConv)
    print(io, "AGNNConv(", l.init_beta)
    l.add_self_loops || print(io, ", add_self_loops=false")
    l.trainable || print(io, ", trainable=false")
    print(io, ")")
end

function (l::AGNNConv)(g, x::AbstractMatrix, ps, st)
    β = l.trainable ? ps.β : l.init_beta
    m = (;  β, l.add_self_loops)
    return GNNlib.agnn_conv(m, g, x), st
end

@doc raw"""
    CGConv((in, ein) => out, act = identity; residual = false,
                use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
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
- `residual`: Add a residual connection.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.

# Examples 

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create random graph
g = rand_graph(rng, 5, 6)
x = rand(rng, Float32, 2, g.num_nodes)
e = rand(rng, Float32, 3, g.num_edges)

l = CGConv((2, 3) => 4, tanh)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, e, ps, st)    # size: (4, num_nodes)

# No edge features
l = CGConv(2 => 4, tanh)
ps, st = LuxCore.setup(rng, l)
y, st = l(g, x, ps, st)    # size: (4, num_nodes)
```
"""
@concrete struct CGConv <: GNNContainerLayer{(:dense_f, :dense_s)}
    in_dims::NTuple{2, Int}
    out_dims::Int
    dense_f
    dense_s
    residual::Bool
    init_weight
    init_bias
end

CGConv(ch::Pair{Int, Int}, args...; kws...) = CGConv((ch[1], 0) => ch[2], args...; kws...)

function CGConv(ch::Pair{NTuple{2, Int}, Int}, act = identity; residual = false,
                use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
    (nin, ein), out = ch
    dense_f = Dense(2nin + ein => out, sigmoid; use_bias, init_weight, init_bias)
    dense_s = Dense(2nin + ein => out, act; use_bias, init_weight, init_bias)
    return CGConv((nin, ein), out, dense_f, dense_s, residual, init_weight, init_bias)
end

LuxCore.outputsize(l::CGConv) = (l.out_dims,)

(l::CGConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::CGConv)(g, x, e, ps, st)
    dense_f = StatefulLuxLayer{true}(l.dense_f, ps.dense_f, _getstate(st, :dense_f))
    dense_s = StatefulLuxLayer{true}(l.dense_s, ps.dense_s, _getstate(st, :dense_s))
    m = (; dense_f, dense_s, l.residual)
    return GNNlib.cg_conv(m, g, x, e), st
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
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = rand(rng, Float32, in_channel, g.num_nodes)

# create layer
l = EdgeConv(Dense(2 * in_channel, out_channel), aggr = +)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)
```
"""
@concrete struct EdgeConv <: GNNContainerLayer{(:nn,)}
    nn <: AbstractLuxLayer
    aggr
end

EdgeConv(nn; aggr = max) = EdgeConv(nn, aggr)

function Base.show(io::IO, l::EdgeConv)
    print(io, "EdgeConv(", l.nn)
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end


function (l::EdgeConv)(g::AbstractGNNGraph, x, ps, st)
    nn = StatefulLuxLayer{true}(l.nn, ps.nn, st.nn)
    m = (; nn, l.aggr)
    y = GNNlib.edge_conv(m, g, x)
    stnew = (; nn = _getstate(nn)) # TODO: support also aggr state if present
    return y, stnew
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

    l(g, x, h, e=nothing, ps, st)
                     
## Forward Pass Arguments:

- `g` : The graph.
- `x` : Matrix of equivariant node coordinates.
- `h` : Matrix of invariant node features.
- `e` : Matrix of invariant edge features. Default `nothing`.
- `ps` : Parameters.
- `st` : State.

Returns updated `h` and `x`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create random graph
g = rand_graph(rng, 10, 10)
h = randn(rng, Float32, 5, g.num_nodes)
x = randn(rng, Float32, 3, g.num_nodes)

egnn = EGNNConv(5 => 6, 10)

# setup layer
ps, st = LuxCore.setup(rng, egnn)

# forward pass
(hnew, xnew), st = egnn(g, h, x, ps, st)
```
"""
@concrete struct EGNNConv <: GNNContainerLayer{(:ϕe, :ϕx, :ϕh)}
    ϕe
    ϕx
    ϕh
    num_features
    residual::Bool
end

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

    ϕh = Chain(Dense(in_size + hidden_size => hidden_size, swish),
               Dense(hidden_size => out_size))

    ϕx = Chain(Dense(hidden_size => hidden_size, swish),
               Dense(hidden_size => 1, use_bias = false))

    num_features = (in = in_size, edge = edge_feat_size, out = out_size,
                    hidden = hidden_size)
    if residual
        @assert in_size==out_size "Residual connection only possible if in_size == out_size"
    end
    return EGNNConv(ϕe, ϕx, ϕh, num_features, residual)
end

LuxCore.outputsize(l::EGNNConv) = (l.num_features.out,)

(l::EGNNConv)(g, h, x, ps, st) = l(g, h, x, nothing, ps, st)

function (l::EGNNConv)(g, h, x, e, ps, st)
    ϕe = StatefulLuxLayer{true}(l.ϕe, ps.ϕe, _getstate(st, :ϕe))
    ϕx = StatefulLuxLayer{true}(l.ϕx, ps.ϕx, _getstate(st, :ϕx))
    ϕh = StatefulLuxLayer{true}(l.ϕh, ps.ϕh, _getstate(st, :ϕh))
    m = (; ϕe, ϕx, ϕh, l.residual, l.num_features)
    return GNNlib.egnn_conv(m, g, h, x, e), st
end

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

"""
    DConv(in => out, k; init_weight = glorot_uniform, init_bias = zeros32, use_bias = true)

Diffusion convolution layer from the paper [Diffusion Convolutional Recurrent Neural Networks: Data-Driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926).

# Arguments

- `in`: The dimension of input features.
- `out`: The dimension of output features.
- `k`: Number of diffusion steps.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.


# Examples
```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create random graph
g = GNNGraph(rand(rng, 10, 10), ndata = rand(rng, Float32, 2, 10))

dconv = DConv(2 => 4, 4)

# setup layer
ps, st = LuxCore.setup(rng, dconv)

# forward pass
y, st = dconv(g, g.ndata.x, ps, st)   # size: (4, num_nodes)
```
"""
@concrete struct DConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    k::Int
    init_weight
    init_bias
    use_bias::Bool
end

function DConv(ch::Pair{Int, Int}, k::Int; 
        init_weight = glorot_uniform, 
        init_bias = zeros32,
        use_bias = true)
    in, out = ch
    return DConv(in, out, k, init_weight, init_bias, use_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::DConv)
    weights = l.init_weight(rng, 2, l.k, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weights, bias)
    else
        return (; weights)
    end
end

LuxCore.outputsize(l::DConv) = (l.out_dims,)
LuxCore.parameterlength(l::DConv) = l.use_bias ? 2 * l.in_dims * l.out_dims * l.k + l.out_dims : 
                                                 2 * l.in_dims * l.out_dims * l.k

function (l::DConv)(g, x, ps, st)
    m = (; ps.weights, bias = _getbias(ps), l.k)
    return GNNlib.d_conv(m, g, x), st
end

function Base.show(io::IO, l::DConv)
    print(io, "DConv($(l.in_dims) => $(l.out_dims), k=$(l.k))")
end

@doc raw"""
    GATConv(in => out, σ = identity; heads = 1, concat = true, negative_slope = 0.2, init_weight = glorot_uniform, init_bias = zeros32, use_bias = true, add_self_loops = true, dropout=0.0)
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
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
- `dropout`: Dropout probability on the normalized attention coefficient. Default `0.0`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = GATConv(in_channel => out_channel; add_self_loops = false, use_bias = false, heads=2, concat=true)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       
```
"""
@concrete struct GATConv <: GNNLayer
    dense_x
    dense_e
    init_weight
    init_bias
    use_bias::Bool
    σ
    negative_slope
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout
end


GATConv(ch::Pair{Int, Int}, args...; kws...) = GATConv((ch[1], 0) => ch[2], args...; kws...)

function GATConv(ch::Pair{NTuple{2, Int}, Int}, σ = identity;
                 heads::Int = 1, concat::Bool = true, negative_slope = 0.2,
                 init_weight = glorot_uniform, init_bias = zeros32,
                 use_bias::Bool = true, 
                 add_self_loops = true, dropout=0.0)
    (in, ein), out = ch
    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_x = Dense(in => out * heads, use_bias = false)
    dense_e = ein > 0 ? Dense(ein => out * heads, use_bias = false) : nothing
    negative_slope = convert(Float32, negative_slope)
    return GATConv(dense_x, dense_e, init_weight, init_bias, use_bias, 
                 σ, negative_slope, ch, heads, concat, add_self_loops, dropout)
end

LuxCore.outputsize(l::GATConv) = (l.concat ? l.channel[2]*l.heads : l.channel[2],)
##TODO: parameterlength

function LuxCore.initialparameters(rng::AbstractRNG, l::GATConv)
    (in, ein), out = l.channel
    dense_x = LuxCore.initialparameters(rng, l.dense_x)
    a = l.init_weight(ein > 0 ? 3out : 2out, l.heads)
    ps = (; dense_x, a)
    if ein > 0
        ps = (ps..., dense_e = LuxCore.initialparameters(rng, l.dense_e))
    end
    if l.use_bias
        ps = (ps..., bias = l.init_bias(rng, l.concat ? out * l.heads : out))
    end
    return ps
end

(l::GATConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::GATConv)(g, x, e, ps, st)
    dense_x = StatefulLuxLayer{true}(l.dense_x, ps.dense_x, _getstate(st, :dense_x))
    dense_e = l.dense_e === nothing ? nothing : 
              StatefulLuxLayer{true}(l.dense_e, ps.dense_e, _getstate(st, :dense_e))

    m = (; l.add_self_loops, l.channel, l.heads, l.concat, l.dropout, l.σ, 
           ps.a, bias = _getbias(ps), dense_x, dense_e, l.negative_slope)
    return GNNlib.gat_conv(m, g, x, e), st
end

function Base.show(io::IO, l::GATConv)
    (in, ein), out = l.channel
    print(io, "GATConv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end

@doc raw"""
    GATv2Conv(in => out, σ = identity; heads = 1, concat = true, negative_slope = 0.2, init_weight = glorot_uniform, init_bias = zeros32, use_bias = true, add_self_loops = true, dropout=0.0)
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
- `heads`: Number attention heads. Default `1`.
- `concat`: Concatenate layer output or not. If not, layer output is averaged over the heads. Default `true`.
- `negative_slope`: The parameter of LeakyReLU.Default `0.2`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `true`.
- `dropout`: Dropout probability on the normalized attention coefficient. Default `0.0`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.


# Examples
```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
ein = 3
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = GATv2Conv((in_channel, ein) => out_channel, add_self_loops = false)

# setup layer
ps, st = LuxCore.setup(rng, l)

# edge features
e = randn(rng, Float32, ein, length(s))

# forward pass
y, st = l(g, x, e, ps, st)    
```
"""
@concrete struct GATv2Conv <: GNNLayer
    dense_i
    dense_j
    dense_e
    init_weight
    init_bias
    use_bias::Bool
    σ
    negative_slope
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout
end

function GATv2Conv(ch::Pair{Int, Int}, args...; kws...)
    GATv2Conv((ch[1], 0) => ch[2], args...; kws...)
end

function GATv2Conv(ch::Pair{NTuple{2, Int}, Int},
                   σ = identity;
                   heads::Int = 1,
                   concat::Bool = true,
                   negative_slope = 0.2,
                   init_weight = glorot_uniform,
                   init_bias = zeros32,
                   use_bias::Bool = true,
                   add_self_loops = true,
                   dropout=0.0)

    (in, ein), out = ch

    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_i = Dense(in => out * heads; use_bias, init_weight, init_bias)
    dense_j = Dense(in => out * heads; use_bias = false, init_weight)
    if ein > 0
        dense_e = Dense(ein => out * heads; use_bias = false, init_weight)
    else
        dense_e = nothing
    end
    return GATv2Conv(dense_i, dense_j, dense_e, 
                     init_weight, init_bias, use_bias, 
                    σ, negative_slope, 
                    ch, heads, concat, add_self_loops, dropout)
end


LuxCore.outputsize(l::GATv2Conv) = (l.concat ? l.channel[2]*l.heads : l.channel[2],)
##TODO: parameterlength

function LuxCore.initialparameters(rng::AbstractRNG, l::GATv2Conv)
    (in, ein), out = l.channel
    dense_i = LuxCore.initialparameters(rng, l.dense_i)
    dense_j = LuxCore.initialparameters(rng, l.dense_j)
    a = l.init_weight(out, l.heads)
    ps = (; dense_i, dense_j, a)
    if ein > 0
        ps = (ps..., dense_e = LuxCore.initialparameters(rng, l.dense_e))
    end
    if l.use_bias
        ps = (ps..., bias = l.init_bias(rng, l.concat ? out * l.heads : out))
    end
    return ps
end

(l::GATv2Conv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::GATv2Conv)(g, x, e, ps, st)
    dense_i = StatefulLuxLayer{true}(l.dense_i, ps.dense_i, _getstate(st, :dense_i))
    dense_j = StatefulLuxLayer{true}(l.dense_j, ps.dense_j, _getstate(st, :dense_j))
    dense_e = l.dense_e === nothing ? nothing : 
              StatefulLuxLayer{true}(l.dense_e, ps.dense_e, _getstate(st, :dense_e))

    m = (; l.add_self_loops, l.channel, l.heads, l.concat, l.dropout, l.σ, 
           ps.a, bias = _getbias(ps), dense_i, dense_j, dense_e, l.negative_slope)
    return GNNlib.gatv2_conv(m, g, x, e), st
end

function Base.show(io::IO, l::GATv2Conv)
    (in, ein), out = l.channel
    print(io, "GATv2Conv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end

@doc raw"""
    SGConv(int => out, k = 1; init_weight = glorot_uniform, init_bias = zeros32, use_bias = true, add_self_loops = true,use_edge_weight = false)
                                
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
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. Default `false`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.


# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(rng, Float32, 3, g.num_nodes)

# create layer
l = SGConv(3 => 5; add_self_loops = true) 

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size:  5 × num_nodes

# convolution with edge weights
w = [1.1, 0.1, 2.3, 0.5]
y = l(g, x, w, ps, st)

# Edge weights can also be embedded in the graph.
g = GNNGraph(s, t, w)
l = SGConv(3 => 5, add_self_loops = true, use_edge_weight=true) 
ps, st = LuxCore.setup(rng, l)
y, st = l(g, x, ps, st) # same as l(g, x, w) 
```
"""
@concrete struct SGConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    k::Int
    use_bias::Bool
    add_self_loops::Bool
    use_edge_weight::Bool
    init_weight
    init_bias    
end

function SGConv(ch::Pair{Int, Int}, k = 1;
                init_weight = glorot_uniform,
                init_bias = zeros32,
                use_bias::Bool = true,
                add_self_loops::Bool = true,
                use_edge_weight::Bool = false)
    in_dims, out_dims = ch
    return SGConv(in_dims, out_dims, k, use_bias, add_self_loops, use_edge_weight, init_weight, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SGConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::SGConv) = l.use_bias ? l.in_dims * l.out_dims + l.out_dims : l.in_dims * l.out_dims
LuxCore.outputsize(d::SGConv) = (d.out_dims,)

function Base.show(io::IO, l::SGConv)
    print(io, "SGConv(", l.in_dims, " => ", l.out_dims)
    l.k || print(io, ", ", l.k)
    l.use_bias || print(io, ", use_bias=false")
    l.add_self_loops || print(io, ", add_self_loops=false")
    !l.use_edge_weight || print(io, ", use_edge_weight=true")
    print(io, ")")
end

(l::SGConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::SGConv)(g, x, edge_weight, ps, st)
    m = (; ps.weight, bias = _getbias(ps), 
           l.add_self_loops, l.use_edge_weight, l.k)
    y = GNNlib.sg_conv(m, g, x, edge_weight)
    return y, st
end

@doc raw"""
    GatedGraphConv(out, num_layers; 
            aggr = +, init_weight = glorot_uniform)

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
- `init_weight`: Weights' initializer. Default `glorot_uniform`.

# Examples:

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
out_channel = 5
num_layers = 3
g = GNNGraph(s, t)

# create layer
l = GatedGraphConv(out_channel, num_layers)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size:  out_channel × num_nodes  
```
"""
@concrete struct GatedGraphConv <: GNNLayer
    gru
    init_weight
    dims::Int
    num_layers::Int
    aggr
end

function GatedGraphConv(dims::Int, num_layers::Int; 
            aggr = +, init_weight = glorot_uniform)
    gru = GRUCell(dims => dims)
    return GatedGraphConv(gru, init_weight, dims, num_layers, aggr)
end

LuxCore.outputsize(l::GatedGraphConv) = (l.dims,)

function LuxCore.initialparameters(rng::AbstractRNG, l::GatedGraphConv)
    gru = LuxCore.initialparameters(rng, l.gru)
    weight = l.init_weight(rng, l.dims, l.dims, l.num_layers)
    return (; gru, weight)
end

LuxCore.parameterlength(l::GatedGraphConv) = parameterlength(l.gru) + l.dims^2*l.num_layers


function (l::GatedGraphConv)(g, x, ps, st)
    gru = StatefulLuxLayer{true}(l.gru, ps.gru, _getstate(st, :gru))
    # make the forward compatible with Flux.GRUCell style
    function fgru(x, h)
        y, (h, ) = gru((x, (h,)))
        return y, h
    end
    m = (; gru=fgru, ps.weight, l.num_layers, l.aggr, l.dims)
    return GNNlib.gated_graph_conv(m, g, x), st
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv($(l.dims), $(l.num_layers)")
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
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(rng, Float32, in_channel, g.num_nodes)

# create dense layer
nn = Dense(in_channel, out_channel)

# create layer
l = GINConv(nn, 0.01f0, aggr = mean)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size:  out_channel × num_nodes
```
"""
@concrete struct GINConv <: GNNContainerLayer{(:nn,)}
    nn <: AbstractLuxLayer
    ϵ <: Real
    aggr
end

GINConv(nn, ϵ; aggr = +) = GINConv(nn, ϵ, aggr)

function (l::GINConv)(g, x, ps, st)
    nn = StatefulLuxLayer{true}(l.nn, ps.nn, st.nn)
    m = (; nn, l.ϵ, l.aggr)
    y = GNNlib.gin_conv(m, g, x)
    stnew = (; nn = _getstate(nn))
    return y, stnew
end

function Base.show(io::IO, l::GINConv)
    print(io, "GINConv($(l.nn)")
    print(io, ", $(l.ϵ)")
    print(io, ")")
end

@doc raw"""
    GMMConv((in, ein) => out, σ=identity; K = 1, residual = false init_weight = glorot_uniform, init_bias = zeros32, use_bias = true)

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
- `residual`: Residual conncetion. Default `false`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s,t)
nin, ein, out, K = 4, 10, 7, 8 
x = randn(rng, Float32, nin, g.num_nodes)
e = randn(rng, Float32, ein, g.num_edges)

# create layer
l = GMMConv((nin, ein) => out, K=K)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, e, ps, st)       # size:  out × num_nodes
```
"""
@concrete struct GMMConv <: GNNLayer
    σ
    ch::Pair{NTuple{2, Int}, Int}
    K::Int
    residual::Bool
    init_weight
    init_bias
    use_bias::Bool
    dense_x
end

function GMMConv(ch::Pair{NTuple{2, Int}, Int}, 
                    σ = identity; 
                    K::Int = 1, 
                    residual = false,
                    init_weight = glorot_uniform, 
                    init_bias = zeros32, 
                    use_bias = true)
    dense_x = Dense(ch[1][1] => ch[2] * K, use_bias = false)
    return GMMConv(σ, ch, K, residual, init_weight, init_bias, use_bias, dense_x)
end


function LuxCore.initialparameters(rng::AbstractRNG, l::GMMConv)
    ein = l.ch[1][2]
    mu = l.init_weight(rng, ein, l.K)
    sigma_inv = l.init_weight(rng, ein, l.K)
    ps = (; mu, sigma_inv, dense_x = LuxCore.initialparameters(rng, l.dense_x))
    if l.use_bias
        bias = l.init_bias(rng, l.ch[2])
        ps = (; ps..., bias)
    end
    return ps
end

LuxCore.outputsize(l::GMMConv) = (l.ch[2],)

function LuxCore.parameterlength(l::GMMConv)
    n = 2 * l.ch[1][2] * l.K
    n += parameterlength(l.dense_x)
    if l.use_bias
        n += l.ch[2]
    end
    return n
end

function (l::GMMConv)(g::GNNGraph, x, e, ps, st)
    dense_x = StatefulLuxLayer{true}(l.dense_x, ps.dense_x, _getstate(st, :dense_x))
    m = (; ps.mu, ps.sigma_inv, dense_x, l.σ, l.ch, l.K, l.residual, bias = _getbias(ps))
    return GNNlib.gmm_conv(m, g, x, e), st
end

function Base.show(io::IO, l::GMMConv)
    (nin, ein), out = l.ch
    print(io, "GMMConv((", nin, ",", ein, ")=>", out)
    l.σ == identity || print(io, ", σ=", l.dense_s.σ)
    print(io, ", K=", l.K)
    print(io, ", residual=", l.residual)
    l.use_bias == true || print(io, ", use_bias=false")
    print(io, ")")
end

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
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create a random graph
g = rand_graph(rng, 10, 30)
x = randn(rng, Float32, 3, 10)
e = randn(rng, Float32, 3, 30)

# create a MEGNetConv layer
m = MEGNetConv(3 => 3)

# setup layer
ps, st = LuxCore.setup(rng, m)

# forward pass
(x′, e′), st = m(g, x, e, ps, st)
```
"""
@concrete struct MEGNetConv{TE, TV, A} <: GNNContainerLayer{(:ϕe, :ϕv)}
    in_dims::Int
    out_dims::Int
    ϕe::TE
    ϕv::TV
    aggr::A
end

function MEGNetConv(in_dims::Int, out_dims::Int, ϕe::TE, ϕv::TV; aggr::A = mean) where {TE, TV, A}
    return MEGNetConv{TE, TV, A}(in_dims, out_dims, ϕe, ϕv, aggr)
end

function MEGNetConv(ch::Pair{Int, Int}; aggr = mean)
    nin, nout = ch
    ϕe = Chain(Dense(3nin, nout, relu),
               Dense(nout, nout))

    ϕv = Chain(Dense(nin + nout, nout, relu),
               Dense(nout, nout))

    return MEGNetConv(nin, nout, ϕe, ϕv, aggr=aggr)
end

function (l::MEGNetConv)(g, x, e, ps, st)
    ϕe = StatefulLuxLayer{true}(l.ϕe, ps.ϕe, _getstate(st, :ϕe))
    ϕv = StatefulLuxLayer{true}(l.ϕv, ps.ϕv, _getstate(st, :ϕv))    
    m = (; ϕe, ϕv, aggr=l.aggr)
    return GNNlib.megnet_conv(m, g, x, e), st
end


LuxCore.outputsize(l::MEGNetConv) = (l.out_dims,)

(l::MEGNetConv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function Base.show(io::IO, l::MEGNetConv)
    nin = l.in_dims
    nout = l.out_dims
    print(io, "MEGNetConv(", nin, " => ", nout)
    print(io, ")")
end

@doc raw"""
    NNConv(in => out, f, σ=identity; aggr=+, init_bias = zeros32, use_bias = true, init_weight = glorot_uniform)

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
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.

# Examples:

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
n_in = 3
n_in_edge = 10
n_out = 5

s = [1,1,2,3]
t = [2,3,1,1]
g = GNNGraph(s, t)
x = randn(rng, Float32, n_in, g.num_nodes)
e = randn(rng, Float32, n_in_edge, g.num_edges)

# create dense layer
nn = Dense(n_in_edge => n_out * n_in)

# create layer
l = NNConv(n_in => n_out, nn, tanh, use_bias = true, aggr = +)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, e, ps, st)       # size:  n_out × num_nodes 
```
"""
@concrete struct NNConv <: GNNContainerLayer{(:nn,)}
    nn <: AbstractLuxLayer    
    aggr
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight
    init_bias
    σ
end

function NNConv(ch::Pair{Int, Int}, nn, σ = identity; 
                aggr = +, 
                init_bias = zeros32,
                use_bias::Bool = true,
                init_weight = glorot_uniform)
    in_dims, out_dims = ch
    σ = NNlib.fast_act(σ)
    return NNConv(nn, aggr, in_dims, out_dims, use_bias, init_weight, init_bias, σ)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::NNConv)
    weight = l.init_weight(rng, l.out_dims, l.in_dims)
    ps = (; nn = LuxCore.initialparameters(rng, l.nn), weight)
    if l.use_bias
        ps = (; ps..., bias = l.init_bias(rng, l.out_dims))
    end
    return ps
end

function LuxCore.initialstates(rng::AbstractRNG, l::NNConv)
    return (; nn = LuxCore.initialstates(rng, l.nn))
end

function LuxCore.parameterlength(l::NNConv)
    n = parameterlength(l.nn) + l.in_dims * l.out_dims
    if l.use_bias
        n += l.out_dims
    end
    return n
end

LuxCore.outputsize(l::NNConv) = (l.out_dims,)

LuxCore.statelength(l::NNConv) = statelength(l.nn)

function (l::NNConv)(g, x, e, ps, st)
    nn = StatefulLuxLayer{true}(l.nn, ps.nn, st.nn)
    m = (; nn, l.aggr, ps.weight, bias = _getbias(ps), l.σ)
    y = GNNlib.nn_conv(m, g, x, e)
    stnew = _getstate(nn)
    return y, stnew
end

function Base.show(io::IO, l::NNConv)
    print(io, "NNConv($(l.in_dims) => $(l.out_dims), $(l.nn)")
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

@doc raw"""
    ResGatedGraphConv(in => out, act=identity; init_weight = glorot_uniform, init_bias = zeros32, use_bias = true)

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
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.


# Examples:

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = randn(rng, Float32, in_channel, g.num_nodes)

# create layer
l = ResGatedGraphConv(in_channel => out_channel, tanh, use_bias = true)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size:  out_channel × num_nodes  
```
"""
@concrete struct ResGatedGraphConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    σ
    init_bias
    init_weight
    use_bias::Bool
end

function ResGatedGraphConv(ch::Pair{Int, Int}, σ = identity; 
                           init_weight = glorot_uniform, 
                           init_bias = zeros32, 
                           use_bias::Bool = true)
    in_dims, out_dims = ch
    return ResGatedGraphConv(in_dims, out_dims, σ, init_bias, init_weight, use_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ResGatedGraphConv)
    A = l.init_weight(rng, l.out_dims, l.in_dims)
    B = l.init_weight(rng, l.out_dims, l.in_dims)
    U = l.init_weight(rng, l.out_dims, l.in_dims)
    V = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; A, B, U, V, bias)
    else
        return (; A, B, U, V)
    end    
end

function LuxCore.parameterlength(l::ResGatedGraphConv)
    n = 4 * l.in_dims * l.out_dims
    if l.use_bias
        n += l.out_dims
    end
    return n
end

LuxCore.outputsize(l::ResGatedGraphConv) = (l.out_dims,)

function (l::ResGatedGraphConv)(g, x, ps, st)
    m = (; ps.A, ps.B, ps.U, ps.V, bias = _getbias(ps), l.σ)
    return GNNlib.res_gated_graph_conv(m, g, x), st
end

function Base.show(io::IO, l::ResGatedGraphConv)
    print(io, "ResGatedGraphConv(", l.in_dims, " => ", l.out_dims)
    l.σ == identity || print(io, ", ", l.σ)
    l.use_bias || print(io, ", use_bias=false")
    print(io, ")")
end

@doc raw"""
    SAGEConv(in => out, σ=identity; aggr=mean, init_weight = glorot_uniform, init_bias = zeros32, use_bias=true)
    
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
- `init_bias`: Bias initializer. Default `zeros32`.
- `use_bias`: Add learnable bias. Default `true`.


# Examples:

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
s = [1,1,2,3]
t = [2,3,1,1]
in_channel = 3
out_channel = 5
g = GNNGraph(s, t)
x = rand(rng, Float32, in_channel, g.num_nodes)

# create layer
l = SAGEConv(in_channel => out_channel, tanh, use_bias = false, aggr = +)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)       # size:  out_channel × num_nodes   
```
"""
@concrete struct SAGEConv <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight
    init_bias
    σ
    aggr
end

function SAGEConv(ch::Pair{Int, Int}, σ = identity; 
                aggr = mean,
                init_weight = glorot_uniform,
                init_bias = zeros32, 
                use_bias::Bool = true)
    in_dims, out_dims = ch
    σ = NNlib.fast_act(σ)
    return SAGEConv(in_dims, out_dims, use_bias, init_weight, init_bias, σ, aggr)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SAGEConv)
    weight = l.init_weight(rng, l.out_dims, 2 * l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        return (; weight, bias)
    else
        return (; weight)
    end
end

LuxCore.parameterlength(l::SAGEConv) = l.use_bias ? l.out_dims * 2 * l.in_dims + l.out_dims : 
                                                  l.out_dims * 2 * l.in_dims
LuxCore.outputsize(d::SAGEConv) = (d.out_dims,)

function Base.show(io::IO, l::SAGEConv)
    print(io, "SAGEConv(", l.in_dims, " => ", l.out_dims)
    (l.σ == identity) || print(io, ", ", l.σ)
    (l.aggr == mean) || print(io, ", aggr=", l.aggr)
    l.use_bias || print(io, ", use_bias=false")    
    print(io, ")")
end

function (l::SAGEConv)(g, x, ps, st)
    m = (; ps.weight, bias = _getbias(ps), 
          l.σ, l.aggr)
    return GNNlib.sage_conv(m, g, x), st
end