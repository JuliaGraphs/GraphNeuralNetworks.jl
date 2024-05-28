# Adapting Flux.Recur to work with GNNGraphs
function (m::Flux.Recur)(g::GNNGraph, x)
    m.state, y = m.cell(m.state, g, x)
    return y
end
    
function (m::Flux.Recur)(g::GNNGraph, x::AbstractArray{T, 3}) where T
    h = [m(g, x_t) for x_t in Flux.eachlastdim(x)]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], sze[2], length(h))
end

struct TGCNCell <: GNNLayer
    conv::GCNConv
    gru::Flux.GRUv3Cell
    state0
    in::Int
    out::Int
end

Flux.@functor TGCNCell

function TGCNCell(ch::Pair{Int, Int};
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  init_state = Flux.zeros32,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    conv = GCNConv(in => out, sigmoid; init, bias, add_self_loops,
                   use_edge_weight)
    gru = Flux.GRUv3Cell(out, out)
    state0 = init_state(out,1)
    return TGCNCell(conv, gru, state0, in,out)
end

function (tgcn::TGCNCell)(h, g::GNNGraph, x::AbstractArray)
    x̃ = tgcn.conv(g, x)
    h, x̃ = tgcn.gru(h, x̃)
    return h, x̃
end

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in) => $(tgcn.out))")
end

"""
    TGCN(in => out; [bias, init, init_state, add_self_loops, use_edge_weight])

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
# Examples

```jldoctest
julia> tgcn = TGCN(2 => 6)
Recur(
  TGCNCell(
    GCNConv(2 => 6, σ),                 # 18 parameters
    GRUv3Cell(6 => 6),                  # 240 parameters
    Float32[0.0; 0.0; … ; 0.0; 0.0;;],  # 6 parameters  (all zero)
    2,
    6,
  ),
)         # Total: 8 trainable arrays, 264 parameters,
          # plus 1 non-trainable, 6 parameters, summarysize 1.492 KiB.

julia> g, x = rand_graph(5, 10), rand(Float32, 2, 5);

julia> y = tgcn(g, x);

julia> size(y)
(6, 5)

julia> Flux.reset!(tgcn);

julia> tgcn(rand_graph(5, 10), rand(Float32, 2, 5, 20)) |> size # batch size of 20
(6, 5, 20)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior.
"""
TGCN(ch; kwargs...) = Flux.Recur(TGCNCell(ch; kwargs...))

Flux.Recur(tgcn::TGCNCell) = Flux.Recur(tgcn, tgcn.state0)

# make TGCN compatible with GNNChain
(l::Flux.Recur{TGCNCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{TGCNCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{TGCNCell}, g::GNNGraph) = l(g)


"""
    A3TGCN(in => out; [bias, init, init_state, add_self_loops, use_edge_weight])

Attention Temporal Graph Convolutional Network (A3T-GCN) model from the paper [A3T-GCN: Attention Temporal Graph
Convolutional Network for Traffic Forecasting](https://arxiv.org/pdf/2006.11583.pdf).

Performs a TGCN layer, followed by a soft attention layer.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `bias`: Add learnable bias. Default `true`.
- `init`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
# Examples

```jldoctest
julia> a3tgcn = A3TGCN(2 => 6)
A3TGCN(2 => 6)

julia> g, x = rand_graph(5, 10), rand(Float32, 2, 5);

julia> y = a3tgcn(g,x);

julia> size(y)
(6, 5)

julia> Flux.reset!(a3tgcn);

julia> y = a3tgcn(rand_graph(5, 10), rand(Float32, 2, 5, 20));

julia> size(y)
(6, 5)
```

!!! warning "Batch size changes"
    Failing to call `reset!` when the input batch size changes can lead to unexpected behavior.
"""
struct A3TGCN <: GNNLayer
    tgcn::Flux.Recur{TGCNCell}
    dense1::Dense
    dense2::Dense
    in::Int
    out::Int
end

Flux.@functor A3TGCN

function A3TGCN(ch::Pair{Int, Int},
                  bias::Bool = true,
                  init = Flux.glorot_uniform,
                  init_state = Flux.zeros32,
                  add_self_loops = false,
                  use_edge_weight = true)
    in, out = ch
    tgcn = TGCN(in => out; bias, init, init_state, add_self_loops, use_edge_weight)
    dense1 = Dense(out, out)
    dense2 = Dense(out, out)
    return A3TGCN(tgcn, dense1, dense2, in, out)
end

function (a3tgcn::A3TGCN)(g::GNNGraph, x::AbstractArray)
    h = a3tgcn.tgcn(g, x)
    e = a3tgcn.dense1(h)
    e = a3tgcn.dense2(e)
    a = softmax(e, dims = 3)
    c = sum(a .* h , dims = 3)
    if length(size(c)) == 3
        c = dropdims(c, dims = 3)
    end
    return c
end

function Base.show(io::IO, a3tgcn::A3TGCN)
    print(io, "A3TGCN($(a3tgcn.in) => $(a3tgcn.out))")
end

struct GConvLSTMCell{W <: AbstractArray{<:Number, 2}, B} <: GNNLayer
    conv_x_i::ChebConv
    conv_h_i::ChebConv
    w_i::W
    b_i::B
    conv_x_f::ChebConv
    conv_h_f::ChebConv
    w_f::W
    b_f::B
    conv_x_c::ChebConv
    conv_h_c::ChebConv
    w_c::W
    b_c::B
    conv_x_o::ChebConv
    conv_h_o::ChebConv
    w_o::W
    b_o::B
    k::Int
    state0
    in::Int
    out::Int
end

Flux.@functor GConvLSTMCell

function GConvLSTMCell(ch::Pair{Int, Int}, k::Int, n::Int;
                        bias::Bool = true,
                        init = Flux.glorot_uniform,
                        init_state = Flux.zeros32)
    in, out = ch
    # input gate
    conv_x_i = ChebConv(in => out, k; bias, init)
    conv_h_i = ChebConv(out => out, k; bias, init)
    w_i = init(out, 1)
    b_i = bias ? Flux.create_bias(w_i, true, out) : false
    # forget gate
    conv_x_f = ChebConv(in => out, k; bias, init)
    conv_h_f = ChebConv(out => out, k; bias, init)
    w_f = init(out, 1)
    b_f = bias ? Flux.create_bias(w_f, true, out) : false
    # cell state
    conv_x_c = ChebConv(in => out, k; bias, init)
    conv_h_c = ChebConv(out => out, k; bias, init)
    w_c = init(out, 1)
    b_c = bias ? Flux.create_bias(w_c, true, out) : false
    # output gate
    conv_x_o = ChebConv(in => out, k; bias, init)
    conv_h_o = ChebConv(out => out, k; bias, init)
    w_o = init(out, 1)
    b_o = bias ? Flux.create_bias(w_o, true, out) : false
    state0 = (init_state(out, n), init_state(out, n))
    return GConvLSTMCell(conv_x_i, conv_h_i, w_i, b_i,
                         conv_x_f, conv_h_f, w_f, b_f,
                         conv_x_c, conv_h_c, w_c, b_c,
                         conv_x_o, conv_h_o, w_o, b_o,
                         k, state0, in, out)
end

function (gclstm::GConvLSTMCell)((h, c), g::GNNGraph, x)
    # input gate
    i = gclstm.conv_x_i(g, x) .+ gclstm.conv_h_i(g, h) .+ gclstm.w_i .* c .+ gclstm.b_i 
    i = Flux.sigmoid_fast(i)
    # forget gate
    f = gclstm.conv_x_f(g, x) .+ gclstm.conv_h_f(g, h) .+ gclstm.w_f .* c .+ gclstm.b_f
    f = Flux.sigmoid_fast(f)
    # cell state
    c = f .* c .+ i .* Flux.tanh_fast(gclstm.conv_x_c(g, x) .+ gclstm.conv_h_c(g, h) .+ gclstm.w_c .* c .+ gclstm.b_c)
    # output gate
    o = gclstm.conv_x_o(g, x) .+ gclstm.conv_h_o(g, h) .+ gclstm.w_o .* c .+ gclstm.b_o
    o = Flux.sigmoid_fast(o)
    h =  o .* Flux.tanh_fast(c)
    return (h,c), h
end

function Base.show(io::IO, gclstm::GConvLSTMCell)
    print(io, "GConvLSTMCell($(gclstm.in) => $(gclstm.out))")
end

GConvLSTM(ch,k,n; kwargs...) = Flux.Recur(GConvLSTMCell(ch,k,n; kwargs...))
Flux.Recur(tgcn::GConvLSTMCell) = Flux.Recur(tgcn, tgcn.state0)

(l::Flux.Recur{GConvLSTMCell})(g::GNNGraph) = GNNGraph(g, ndata = l(g, node_features(g)))
_applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph, x) = l(g, x)
_applylayer(l::Flux.Recur{GConvLSTMCell}, g::GNNGraph) = l(g)

function (l::GINConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::ChebConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GATConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GATv2Conv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GatedGraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::CGConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::SGConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::TransformerConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GCNConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::ResGatedGraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::SAGEConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end

function (l::GraphConv)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector)
    return l.(tg.snapshots, x)
end
