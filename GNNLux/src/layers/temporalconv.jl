# Carrier builders: assemble the lightweight namedtuples that the shared GNNlib
# forward passes expect, merging the layer configuration with the parameters `ps`.
_cheb_carrier(conv, ps) = (; ps.weight, bias = _getbias(ps), conv.k)
_dconv_carrier(conv, ps) = (; ps.weights, bias = _getbias(ps), conv.k)
_dense_carrier(dense, ps) = (; ps.weight, bias = _getbias(ps), σ = dense.activation)

@doc raw"""
    GNNRecurrence(cell; return_sequence = true)

Recurrent layer analogous to [`Lux.Recurrence`](https://lux.csail.mit.edu/stable/api/Lux/layers#Lux.Recurrence)
that wraps a graph recurrent `cell` and applies it over an entire temporal
sequence of node features at once.

The `cell` has to follow the recurrent-cell interface
`(out, carry), st = cell(g, (x, carry), ps, st)`, with the convenience method
`(out, carry), st = cell(g, x, ps, st)` initializing the carry to zeros.

The layer constructors [`TGCN`](@ref), [`GConvGRU`](@ref), [`GConvLSTM`](@ref),
[`DCGRU`](@ref) and [`EvolveGCNO`](@ref) all return a `GNNRecurrence` wrapping
the corresponding cell.

# Arguments

- `cell`: A graph recurrent cell (e.g. [`TGCNCell`](@ref)).
- `return_sequence`: If `true` the whole sequence of outputs is returned,
  otherwise only the last output. Default `true`.

# Forward

    layer(g, x, ps, st)

- `g`: The input `GNNGraph` or `TemporalSnapshotsGNNGraph`.
    - If a `GNNGraph`, the same graph is used at every timestep.
    - If a `TemporalSnapshotsGNNGraph`, a different graph (snapshot) is used at
      each timestep. Not all cells support this.
- `x`: The time-varying node features.
    - If `g` is a `GNNGraph`, an array of size `in x timesteps x num_nodes`.
    - If `g` is a `TemporalSnapshotsGNNGraph`, a vector of length `timesteps`
      whose `t`-th element has size `in x num_nodes_t`.

Returns the updated node features and state:
- If `return_sequence == true` and `g` is a `GNNGraph`, the output is an array of
  size `out x timesteps x num_nodes`; if `g` is a `TemporalSnapshotsGNNGraph`, it
  is a vector of length `timesteps`.
- If `return_sequence == false`, only the last timestep's output is returned.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
num_nodes, num_edges = 5, 10
d_in, d_out, timesteps = 2, 3, 5

g = rand_graph(rng, num_nodes, num_edges)
x = rand(rng, Float32, d_in, timesteps, num_nodes)

cell = GConvLSTMCell(d_in => d_out, 2)
layer = GNNRecurrence(cell)
ps, st = LuxCore.setup(rng, layer)

y, st = layer(g, x, ps, st)   # size(y) == (d_out, timesteps, num_nodes)
```
"""
@concrete struct GNNRecurrence <: GNNContainerLayer{(:cell,)}
    cell
    return_sequence <: StaticBool
end

GNNRecurrence(cell; return_sequence::Bool = true) = GNNRecurrence(cell, static(return_sequence))

_reduce_sequence(::Static.True, outs) = stack(outs; dims = 2)
_reduce_sequence(::Static.False, outs) = last(outs)
_reduce_snapshots(::Static.True, outs) = outs
_reduce_snapshots(::Static.False, outs) = last(outs)

function (r::GNNRecurrence)(g::GNNGraph, x::AbstractArray{<:Any, 3}, ps, st::NamedTuple)
    (out, carry), cst = r.cell(g, x[:, 1, :], ps.cell, st.cell)
    outs = [out]
    for t in 2:size(x, 2)
        (out, carry), cst = r.cell(g, (x[:, t, :], carry), ps.cell, cst)
        outs = vcat(outs, [out])
    end
    return _reduce_sequence(r.return_sequence, outs), (; cell = cst)
end

function (r::GNNRecurrence)(tg::TemporalSnapshotsGNNGraph, x::AbstractVector, ps, st::NamedTuple)
    (out, carry), cst = r.cell(tg.snapshots[1], x[1], ps.cell, st.cell)
    outs = [out]
    for t in 2:length(x)
        (out, carry), cst = r.cell(tg.snapshots[t], (x[t], carry), ps.cell, cst)
        outs = vcat(outs, [out])
    end
    return _reduce_snapshots(r.return_sequence, outs), (; cell = cst)
end

LuxCore.outputsize(r::GNNRecurrence) = LuxCore.outputsize(r.cell)

function Base.show(io::IO, r::GNNRecurrence)
    print(io, "GNNRecurrence($(r.cell))")
end

@doc raw"""
    TGCNCell(in => out; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32,
             add_self_loops = true, use_edge_weight = false, act = relu)

Recurrent graph convolutional cell from the paper
[T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320).

Uses two stacked [`GCNConv`](@ref) layers to model spatial dependencies and a
GRU mechanism to model temporal dependencies.

# Arguments

- `in => out`: A pair where `in` is the number of input node features and `out`
  the number of output node features.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Convolution weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before the convolution. Default `true`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available). Default `false`.
- `act`: Activation function of the first `GCNConv` layer. Default `relu`.

# Forward

    cell(g, x, ps, st)
    cell(g, (x, h), ps, st)

Performs one recurrence step and returns `(h, h), st`, where `h` is the updated
hidden state of size `out x num_nodes`. If the carry `h` is not provided, it is
initialized to zeros.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

cell = TGCNCell(2 => 6)
ps, st = LuxCore.setup(rng, cell)
(y, h), st = cell(g, x, ps, st)   # size(y) == (6, 5)
```
"""
@concrete struct TGCNCell <: GNNContainerLayer{(:conv_z, :dense_z, :conv_r, :dense_r, :conv_h, :dense_h)}
    in_dims::Int
    out_dims::Int
    conv_z
    dense_z
    conv_r
    dense_r
    conv_h
    dense_h
end

function TGCNCell((in_dims, out_dims)::Pair{Int, Int}; act = relu,
                  use_bias = true, init_weight = glorot_uniform, init_bias = zeros32,
                  add_self_loops = true, use_edge_weight = false)
    convkws = (; use_bias, init_weight, init_bias, add_self_loops, use_edge_weight)
    mkconv(σ) = GNNChain(GCNConv(in_dims => out_dims, σ; convkws...),
                         GCNConv(out_dims => out_dims; convkws...))
    conv_z = mkconv(act); dense_z = Dense(2 * out_dims => out_dims, sigmoid)
    conv_r = mkconv(act); dense_r = Dense(2 * out_dims => out_dims, sigmoid)
    conv_h = mkconv(act); dense_h = Dense(2 * out_dims => out_dims, tanh)
    return TGCNCell(in_dims, out_dims, conv_z, dense_z, conv_r, dense_r, conv_h, dense_h)
end

function (cell::TGCNCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    h = zeros_like(x, (cell.out_dims, g.num_nodes))
    return cell(g, (x, h), ps, st)
end

function (cell::TGCNCell)(g, xh::Tuple, ps, st::NamedTuple)
    x, h = xh
    cz, _ = cell.conv_z(g, x, ps.conv_z, st.conv_z)
    cr, _ = cell.conv_r(g, x, ps.conv_r, st.conv_r)
    ch, _ = cell.conv_h(g, x, ps.conv_h, st.conv_h)
    l = (; dense_z = _dense_carrier(cell.dense_z, ps.dense_z),
           dense_r = _dense_carrier(cell.dense_r, ps.dense_r),
           dense_h = _dense_carrier(cell.dense_h, ps.dense_h))
    h = GNNlib.tgcn(l, cz, cr, ch, h)
    return (h, h), st
end

LuxCore.outputsize(cell::TGCNCell) = (cell.out_dims,)

Base.show(io::IO, cell::TGCNCell) = print(io, "TGCNCell($(cell.in_dims) => $(cell.out_dims))")

"""
    TGCN(in => out; kws...)

Construct a [`GNNRecurrence`](@ref) layer from a [`TGCNCell`](@ref).
The arguments are passed to the [`TGCNCell`](@ref) constructor.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5, 5)  # (in, timesteps, num_nodes)

layer = TGCN(2 => 6)
ps, st = LuxCore.setup(rng, layer)
y, st = layer(g, x, ps, st)   # size(y) == (6, 5, 5)
```
"""
TGCN(ch::Pair{Int, Int}; kws...) = GNNRecurrence(TGCNCell(ch; kws...))

@doc raw"""
    GConvGRUCell(in => out, k; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)

Graph Convolutional Gated Recurrent Unit (GConvGRU) recurrent cell from the paper
[Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659).

Uses [`ChebConv`](@ref) to model spatial dependencies, followed by a Gated
Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in => out`: A pair where `in` is the number of input node features and `out`
  the number of output node features.
- `k`: Chebyshev polynomial order.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Forward

    cell(g, x, ps, st)
    cell(g, (x, h), ps, st)

Performs one recurrence step and returns `(h, h), st`, where `h` is the updated
hidden state of size `out x num_nodes`. If the carry `h` is not provided, it is
initialized to zeros.
"""
@concrete struct GConvGRUCell <: GNNContainerLayer{(:conv_x_r, :conv_h_r, :conv_x_z, :conv_h_z, :conv_x_h, :conv_h_h)}
    in_dims::Int
    out_dims::Int
    k::Int
    conv_x_r
    conv_h_r
    conv_x_z
    conv_h_z
    conv_x_h
    conv_h_h
end

function GConvGRUCell((in_dims, out_dims)::Pair{Int, Int}, k::Int;
                      use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
    convkws = (; use_bias, init_weight, init_bias)
    conv_x_r = ChebConv(in_dims => out_dims, k; convkws...)
    conv_h_r = ChebConv(out_dims => out_dims, k; convkws...)
    conv_x_z = ChebConv(in_dims => out_dims, k; convkws...)
    conv_h_z = ChebConv(out_dims => out_dims, k; convkws...)
    conv_x_h = ChebConv(in_dims => out_dims, k; convkws...)
    conv_h_h = ChebConv(out_dims => out_dims, k; convkws...)
    return GConvGRUCell(in_dims, out_dims, k, conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h)
end

function (cell::GConvGRUCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    h = zeros_like(x, (cell.out_dims, g.num_nodes))
    return cell(g, (x, h), ps, st)
end

function (cell::GConvGRUCell)(g, xh::Tuple, ps, st::NamedTuple)
    x, h = xh
    l = (; conv_x_r = _cheb_carrier(cell.conv_x_r, ps.conv_x_r),
           conv_h_r = _cheb_carrier(cell.conv_h_r, ps.conv_h_r),
           conv_x_z = _cheb_carrier(cell.conv_x_z, ps.conv_x_z),
           conv_h_z = _cheb_carrier(cell.conv_h_z, ps.conv_h_z),
           conv_x_h = _cheb_carrier(cell.conv_x_h, ps.conv_x_h),
           conv_h_h = _cheb_carrier(cell.conv_h_h, ps.conv_h_h))
    h = GNNlib.gconv_gru(l, g, x, h)
    return (h, h), st
end

LuxCore.outputsize(cell::GConvGRUCell) = (cell.out_dims,)

Base.show(io::IO, cell::GConvGRUCell) = print(io, "GConvGRUCell($(cell.in_dims) => $(cell.out_dims), $(cell.k))")

"""
    GConvGRU(in => out, k; kws...)

Construct a [`GNNRecurrence`](@ref) layer from a [`GConvGRUCell`](@ref).
The arguments are passed to the [`GConvGRUCell`](@ref) constructor.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5, 5)  # (in, timesteps, num_nodes)

layer = GConvGRU(2 => 5, 2)
ps, st = LuxCore.setup(rng, layer)
y, st = layer(g, x, ps, st)   # size(y) == (5, 5, 5)
```
"""
GConvGRU(ch::Pair{Int, Int}, k::Int; kws...) = GNNRecurrence(GConvGRUCell(ch, k; kws...))

@doc raw"""
    GConvLSTMCell(in => out, k; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)

Graph Convolutional Long Short-Term Memory (GConvLSTM) recurrent cell from the
paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659).

Uses [`ChebConv`](@ref) to model spatial dependencies, followed by a Long
Short-Term Memory (LSTM) cell with peephole connections to model temporal
dependencies.

# Arguments

- `in => out`: A pair where `in` is the number of input node features and `out`
  the number of output node features.
- `k`: Chebyshev polynomial order.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Forward

    cell(g, x, ps, st)
    cell(g, (x, (h, c)), ps, st)

Performs one recurrence step and returns `(h, (h, c)), st`, where `h` is the
updated hidden state and `c` the updated cell state, both of size `out x num_nodes`.
If the carry `(h, c)` is not provided, it is initialized to zeros.
"""
@concrete struct GConvLSTMCell <: GNNContainerLayer{(:conv_x_i, :conv_h_i, :scale_i, :conv_x_f, :conv_h_f, :scale_f, :conv_x_c, :conv_h_c, :scale_c, :conv_x_o, :conv_h_o, :scale_o)}
    in_dims::Int
    out_dims::Int
    k::Int
    conv_x_i
    conv_h_i
    scale_i
    conv_x_f
    conv_h_f
    scale_f
    conv_x_c
    conv_h_c
    scale_c
    conv_x_o
    conv_h_o
    scale_o
end

function GConvLSTMCell((in_dims, out_dims)::Pair{Int, Int}, k::Int;
                       use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
    convkws = (; use_bias, init_weight, init_bias)
    mkconv(d) = ChebConv(d, k; convkws...)
    mkscale() = Lux.Scale(out_dims; use_bias, init_weight, init_bias)
    return GConvLSTMCell(in_dims, out_dims, k,
        mkconv(in_dims => out_dims), mkconv(out_dims => out_dims), mkscale(),
        mkconv(in_dims => out_dims), mkconv(out_dims => out_dims), mkscale(),
        mkconv(in_dims => out_dims), mkconv(out_dims => out_dims), mkscale(),
        mkconv(in_dims => out_dims), mkconv(out_dims => out_dims), mkscale())
end

function (cell::GConvLSTMCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    h = zeros_like(x, (cell.out_dims, g.num_nodes))
    c = zeros_like(x, (cell.out_dims, g.num_nodes))
    return cell(g, (x, (h, c)), ps, st)
end

function (cell::GConvLSTMCell)(g, xhc::Tuple, ps, st::NamedTuple)
    x, (h, c) = xhc
    l = (; conv_x_i = _cheb_carrier(cell.conv_x_i, ps.conv_x_i),
           conv_h_i = _cheb_carrier(cell.conv_h_i, ps.conv_h_i),
           w_i = ps.scale_i.weight, b_i = _getbias(ps.scale_i),
           conv_x_f = _cheb_carrier(cell.conv_x_f, ps.conv_x_f),
           conv_h_f = _cheb_carrier(cell.conv_h_f, ps.conv_h_f),
           w_f = ps.scale_f.weight, b_f = _getbias(ps.scale_f),
           conv_x_c = _cheb_carrier(cell.conv_x_c, ps.conv_x_c),
           conv_h_c = _cheb_carrier(cell.conv_h_c, ps.conv_h_c),
           w_c = ps.scale_c.weight, b_c = _getbias(ps.scale_c),
           conv_x_o = _cheb_carrier(cell.conv_x_o, ps.conv_x_o),
           conv_h_o = _cheb_carrier(cell.conv_h_o, ps.conv_h_o),
           w_o = ps.scale_o.weight, b_o = _getbias(ps.scale_o))
    h, c = GNNlib.gconv_lstm(l, g, x, h, c)
    return (h, (h, c)), st
end

LuxCore.outputsize(cell::GConvLSTMCell) = (cell.out_dims,)

Base.show(io::IO, cell::GConvLSTMCell) = print(io, "GConvLSTMCell($(cell.in_dims) => $(cell.out_dims), $(cell.k))")

"""
    GConvLSTM(in => out, k; kws...)

Construct a [`GNNRecurrence`](@ref) layer from a [`GConvLSTMCell`](@ref).
The arguments are passed to the [`GConvLSTMCell`](@ref) constructor.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5, 5)  # (in, timesteps, num_nodes)

layer = GConvLSTM(2 => 5, 2)
ps, st = LuxCore.setup(rng, layer)
y, st = layer(g, x, ps, st)   # size(y) == (5, 5, 5)
```
"""
GConvLSTM(ch::Pair{Int, Int}, k::Int; kws...) = GNNRecurrence(GConvLSTMCell(ch, k; kws...))

@doc raw"""
    DCGRUCell(in => out, k; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)

Diffusion Convolutional Recurrent Neural Network (DCGRU) cell from the paper
[Diffusion Convolutional Recurrent Neural Network: Data-driven Traffic Forecasting](https://arxiv.org/abs/1707.01926).

Uses a [`DConv`](@ref) layer to model spatial dependencies, in combination with a
Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in => out`: A pair where `in` is the number of input node features and `out`
  the number of output node features.
- `k`: Diffusion step for the `DConv`.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Convolution weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Forward

    cell(g, x, ps, st)
    cell(g, (x, h), ps, st)

Performs one recurrence step and returns `(h, h), st`, where `h` is the updated
hidden state of size `out x num_nodes`. If the carry `h` is not provided, it is
initialized to zeros.
"""
@concrete struct DCGRUCell <: GNNContainerLayer{(:dconv_u, :dconv_r, :dconv_c)}
    in_dims::Int
    out_dims::Int
    k::Int
    dconv_u
    dconv_r
    dconv_c
end

function DCGRUCell((in_dims, out_dims)::Pair{Int, Int}, k::Int;
                   use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
    convkws = (; use_bias, init_weight, init_bias)
    dconv_u = DConv((in_dims + out_dims) => out_dims, k; convkws...)
    dconv_r = DConv((in_dims + out_dims) => out_dims, k; convkws...)
    dconv_c = DConv((in_dims + out_dims) => out_dims, k; convkws...)
    return DCGRUCell(in_dims, out_dims, k, dconv_u, dconv_r, dconv_c)
end

function (cell::DCGRUCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    h = zeros_like(x, (cell.out_dims, g.num_nodes))
    return cell(g, (x, h), ps, st)
end

function (cell::DCGRUCell)(g, xh::Tuple, ps, st::NamedTuple)
    x, h = xh
    l = (; dconv_u = _dconv_carrier(cell.dconv_u, ps.dconv_u),
           dconv_r = _dconv_carrier(cell.dconv_r, ps.dconv_r),
           dconv_c = _dconv_carrier(cell.dconv_c, ps.dconv_c))
    h = GNNlib.dcgru(l, g, x, h)
    return (h, h), st
end

LuxCore.outputsize(cell::DCGRUCell) = (cell.out_dims,)

Base.show(io::IO, cell::DCGRUCell) = print(io, "DCGRUCell($(cell.in_dims) => $(cell.out_dims), $(cell.k))")

"""
    DCGRU(in => out, k; kws...)

Construct a [`GNNRecurrence`](@ref) layer from a [`DCGRUCell`](@ref).
The arguments are passed to the [`DCGRUCell`](@ref) constructor.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5, 5)  # (in, timesteps, num_nodes)

layer = DCGRU(2 => 5, 2)
ps, st = LuxCore.setup(rng, layer)
y, st = layer(g, x, ps, st)   # size(y) == (5, 5, 5)
```
"""
DCGRU(ch::Pair{Int, Int}, k::Int; kws...) = GNNRecurrence(DCGRUCell(ch, k; kws...))

@doc raw"""
    EvolveGCNOCell(in => out; use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)

Evolving Graph Convolutional Network cell of type "-O" from the paper
[EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191).

Uses a [`GCNConv`](@ref) layer whose weight matrix is evolved across the temporal
sequence by an `LSTMCell`. It can work with time-varying graphs and node features.

# Arguments

- `in => out`: A pair where `in` is the number of input node features and `out`
  the number of output node features.
- `use_bias`: Add learnable bias for the convolution and the LSTM cell. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Forward

    cell(g, x, ps, st)
    cell(g, (x, state), ps, st)

Performs one recurrence step and returns `(y, state), st`, where `y` is the
convolution output of size `out x num_nodes` and `state` is the updated
`(weight, lstm_carry)` carry. If the carry is not provided, the convolution
weight is initialized from the parameters and the LSTM carry to zeros.
"""
@concrete struct EvolveGCNOCell <: GNNContainerLayer{(:conv, :lstm)}
    in_dims::Int
    out_dims::Int
    conv
    lstm
end

function EvolveGCNOCell((in_dims, out_dims)::Pair{Int, Int};
                        use_bias = true, init_weight = glorot_uniform, init_bias = zeros32)
    conv = GCNConv(in_dims => out_dims; use_bias, init_weight, init_bias)
    lstm = Lux.LSTMCell(in_dims * out_dims => in_dims * out_dims; use_bias)
    return EvolveGCNOCell(in_dims, out_dims, conv, lstm)
end

function (cell::EvolveGCNOCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    w0 = reshape(ps.conv.weight, :, 1)
    (w, lstm_carry), st_lstm = cell.lstm(w0, ps.lstm, st.lstm)
    return _evolvegcno_apply(cell, g, x, w, lstm_carry, ps, st, st_lstm)
end

function (cell::EvolveGCNOCell)(g, xs::Tuple, ps, st::NamedTuple)
    x, (wprev, lstm_carry) = xs
    (w, lstm_carry), st_lstm = cell.lstm((wprev, lstm_carry), ps.lstm, st.lstm)
    return _evolvegcno_apply(cell, g, x, w, lstm_carry, ps, st, st_lstm)
end

function _evolvegcno_apply(cell::EvolveGCNOCell, g, x, w, lstm_carry, ps, st, st_lstm)
    conv_weight = reshape(w, cell.out_dims, cell.in_dims)
    y, st_conv = cell.conv(g, x, ps.conv, st.conv; conv_weight)
    return (y, (w, lstm_carry)), (; conv = st_conv, lstm = st_lstm)
end

LuxCore.outputsize(cell::EvolveGCNOCell) = (cell.out_dims,)

Base.show(io::IO, cell::EvolveGCNOCell) = print(io, "EvolveGCNOCell($(cell.in_dims) => $(cell.out_dims))")

"""
    EvolveGCNO(in => out; kws...)

Construct a [`GNNRecurrence`](@ref) layer from an [`EvolveGCNOCell`](@ref).
It can process an entire temporal sequence of graphs and node features at once.
The arguments are passed to the [`EvolveGCNOCell`](@ref) constructor.

# Examples

```julia
using GNNLux, Lux, Random

rng = Random.default_rng()
tg = TemporalSnapshotsGNNGraph([rand_graph(rng, 10, 20), rand_graph(rng, 10, 14), rand_graph(rng, 10, 22)])
x = [rand(rng, Float32, 4, 10) for _ in 1:tg.num_snapshots]

layer = EvolveGCNO(4 => 5)
ps, st = LuxCore.setup(rng, layer)
y, st = layer(tg, x, ps, st)   # length(y) == 3, size(y[1]) == (5, 10)
```
"""
EvolveGCNO(ch::Pair{Int, Int}; kws...) = GNNRecurrence(EvolveGCNOCell(ch; kws...))
