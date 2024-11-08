_getbias(ps) = hasproperty(ps, :bias) ? getproperty(ps, :bias) : false
_getstate(st, name) = hasproperty(st, name) ? getproperty(st, name) : NamedTuple()
_getstate(s::StatefulLuxLayer{true}) = s.st
_getstate(s::StatefulLuxLayer{Static.True}) = s.st
_getstate(s::StatefulLuxLayer{false}) = s.st_any
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
    fgru = (h, x) -> gru((x, (h,)))  # make the forward compatible with Flux.GRUCell style
    m = (; gru=fgru, ps.weight, l.num_layers, l.aggr, l.dims)
    return GNNlib.gated_graph_conv(m, g, x), st
end

function Base.show(io::IO, l::GatedGraphConv)
    print(io, "GatedGraphConv($(l.dims), $(l.num_layers)")
    print(io, ", aggr=", l.aggr)
    print(io, ")")
end

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

@concrete struct TransformerConv <: GNNContainerLayer{(:W1, :W2, :W3, :W4, :W5, :W6, :FF, :BN1, :BN2)}
    in_dims::NTuple{2, Int}
    out_dims::Int
    heads::Int
    add_self_loops::Bool
    concat::Bool
    skip_connection::Bool
    sqrt_out::Float32
    W1 
    W2 
    W3 
    W4 
    W5 
    W6 
    FF 
    BN1 
    BN2 
end

function TransformerConv(ch::Pair{Int, Int}, args...; kws...)
    return TransformerConv((ch[1], 0) => ch[2], args...; kws...)
end

function TransformerConv(ch::Pair{NTuple{2, Int}, Int};
                         heads::Int = 1,
                         concat::Bool = true,
                         init_weight = glorot_uniform,
                         init_bias = zeros32,
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

    if skip_connection
        @assert in == (concat ? out * heads : out) "In-channels must correspond to out-channels * heads (or just out_channels if concat=false) if skip_connection is used"
    end

    W1 = root_weight ? Dense(in => out * (concat ? heads : 1); use_bias=bias_root, init_weight, init_bias) : nothing
    W2 = Dense(in => out * heads; use_bias=bias_qkv, init_weight, init_bias)
    W3 = Dense(in => out * heads; use_bias=bias_qkv, init_weight, init_bias)
    W4 = Dense(in => out * heads; use_bias=bias_qkv, init_weight, init_bias)
    out_mha = out * (concat ? heads : 1)
    W5 = gating ? Dense(3 * out_mha => 1, sigmoid; use_bias=false, init_weight, init_bias) : nothing
    W6 = ein > 0 ? Dense(ein => out * heads; use_bias=bias_qkv, init_weight, init_bias) : nothing
    FF = ff_channels > 0 ?
         Chain(Dense(out_mha => ff_channels, relu; init_weight, init_bias),
               Dense(ff_channels => out_mha; init_weight, init_bias)) : nothing
    BN1 = batch_norm ? BatchNorm(out_mha) : nothing
    BN2 = (batch_norm && ff_channels > 0) ? BatchNorm(out_mha) : nothing

    return TransformerConv((in, ein), out, heads, add_self_loops, concat, 
                           skip_connection, Float32(√out), W1, W2, W3, W4, W5, W6, FF, BN1, BN2)
end

LuxCore.outputsize(l::TransformerConv) = (l.concat ? l.out_dims * l.heads : l.out_dims,) 

function (l::TransformerConv)(g, x, ps, st) 
    return l(g, x, nothing, ps, st)
end

function (l::TransformerConv)(g, x, e, ps, st)
    W1 = l.W1 === nothing ? nothing : 
            StatefulLuxLayer{true}(l.W1, ps.W1, _getstate(st, :W1))
    W2 = StatefulLuxLayer{true}(l.W2, ps.W2, _getstate(st, :W2))
    W3 = StatefulLuxLayer{true}(l.W3, ps.W3, _getstate(st, :W3))
    W4 = StatefulLuxLayer{true}(l.W4, ps.W4, _getstate(st, :W4))
    W5 = l.W5 === nothing ? nothing : 
            StatefulLuxLayer{true}(l.W5, ps.W5, _getstate(st, :W5))
    W6 = l.W6 === nothing ? nothing : 
            StatefulLuxLayer{true}(l.W6, ps.W6, _getstate(st, :W6))
    FF = l.FF === nothing ? nothing : 
            StatefulLuxLayer{true}(l.FF, ps.FF, _getstate(st, :FF))
    BN1 = l.BN1 === nothing ? nothing : 
            StatefulLuxLayer{true}(l.BN1, ps.BN1, _getstate(st, :BN1))
    BN2 = l.BN2 === nothing ? nothing : 
            StatefulLuxLayer{true}(l.BN2, ps.BN2, _getstate(st, :BN2))
    m = (; W1, W2, W3, W4, W5, W6, FF, BN1, BN2, l.sqrt_out, 
           l.heads, l.concat, l.skip_connection, l.add_self_loops, l.in_dims, l.out_dims)
    return GNNlib.transformer_conv(m, g, x, e), st
end

function LuxCore.parameterlength(l::TransformerConv)
    n = parameterlength(l.W2) + parameterlength(l.W3) + parameterlength(l.W4) 
    n += l.W1 === nothing ? 0 : parameterlength(l.W1)
    n += l.W5 === nothing ? 0 : parameterlength(l.W5)
    n += l.W6 === nothing ? 0 : parameterlength(l.W6)
    n += l.FF === nothing ? 0 : parameterlength(l.FF)
    n += l.BN1 === nothing ? 0 : parameterlength(l.BN1)
    n += l.BN2 === nothing ? 0 : parameterlength(l.BN2)
    return n
end

function LuxCore.statelength(l::TransformerConv)
    n = statelength(l.W2) + statelength(l.W3) + statelength(l.W4)
    n += l.W1 === nothing ? 0 : statelength(l.W1)
    n += l.W5 === nothing ? 0 : statelength(l.W5)
    n += l.W6 === nothing ? 0 : statelength(l.W6) 
    n += l.FF === nothing ? 0 : statelength(l.FF)
    n += l.BN1 === nothing ? 0 : statelength(l.BN1)
    n += l.BN2 === nothing ? 0 : statelength(l.BN2)
    return n
end

function Base.show(io::IO, l::TransformerConv)
    (in, ein), out = (l.in_dims, l.out_dims)
    print(io, "TransformerConv(($in, $ein) => $out, heads=$(l.heads))")
end
      
      
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
