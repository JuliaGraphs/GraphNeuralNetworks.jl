@concrete struct StatefulRecurrentCell <: AbstractLuxContainerLayer{(:cell,)}     
    cell <: Union{<:Lux.AbstractRecurrentCell, <:GNNContainerLayer}
end

function LuxCore.initialstates(rng::AbstractRNG, r::GNNLux.StatefulRecurrentCell)
    return (cell=LuxCore.initialstates(rng, r.cell), carry=nothing)
end

function (r::StatefulRecurrentCell)(g, x::AbstractMatrix, ps, st::NamedTuple)
    (out, carry), st = applyrecurrentcell(r.cell, g, x, ps.cell, st.cell, st.carry)
    return out, (; cell=st, carry)
end

function (r::StatefulRecurrentCell)(g, x::AbstractVector, ps, st::NamedTuple)
    stcell, carry = st.cell, st.carry
    for xᵢ in x
        (out, carry), stcell = applyrecurrentcell(r.cell, g, xᵢ, ps.cell, stcell, carry)
    end
    return out, (; cell=stcell, carry)
end

function applyrecurrentcell(l, g, x, ps, st, carry)
    return Lux.apply(l, g, (x, carry), ps, st)
end

LuxCore.apply(m::GNNContainerLayer, g, x, ps, st) = m(g, x, ps, st)

@concrete struct TGCNCell <: GNNContainerLayer{(:conv, :gru)}
    in_dims::Int
    out_dims::Int
    conv
    gru
    init_state::Function
end

function TGCNCell(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, 
                    init_state = zeros32, init_bias = zeros32, add_self_loops = false, 
                    use_edge_weight = true, act = sigmoid) 
    in_dims, out_dims = ch
    conv = GCNConv(ch, act; init_weight, init_bias, use_bias, add_self_loops, use_edge_weight)
    gru = Lux.GRUCell(out_dims => out_dims; use_bias, init_weight = (init_weight, init_weight, init_weight), init_bias = (init_bias, init_bias, init_bias), init_state = init_state)
    return TGCNCell(in_dims, out_dims, conv, gru, init_state)
end

function (l::TGCNCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, 1)
    end
    x̃, stconv = l.conv(g, x, ps.conv, st.conv)
    (h, (h,)), stgru = l.gru((x̃,(h,)), ps.gru,st.gru)
    return  (h, h), (conv=stconv, gru=stgru)
end

LuxCore.outputsize(l::TGCNCell) = (l.out_dims,)
LuxCore.outputsize(l::GNNLux.StatefulRecurrentCell) = (l.cell.out_dims,)

function Base.show(io::IO, tgcn::TGCNCell)
    print(io, "TGCNCell($(tgcn.in_dims) => $(tgcn.out_dims))")
end

"""
    TGCN(in => out; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true, act = sigmoid) 

Temporal Graph Convolutional Network (T-GCN) recurrent layer from the paper [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/pdf/1811.05320.pdf).

Performs a layer of GCNConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
- `act`: Activation function used in the GCNConv layer. Default `sigmoid`.


# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

# create TGCN layer 
tgcn = TGCN(2 => 6)

# create TGCN layer with custom activation
tgcn_relu = TGCN(2 => 6, act = relu)

# setup layer
ps, st = LuxCore.setup(rng, tgcn)

# forward pass
y, st = tgcn(g, x, ps, st)      # result size (6, 5)
```
"""
TGCN(ch::Pair{Int, Int}; kwargs...) = GNNLux.StatefulRecurrentCell(TGCNCell(ch; kwargs...))

"""
    A3TGCN(in => out; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true) 

Attention Temporal Graph Convolutional Network (A3T-GCN) model from the paper [A3T-GCN: Attention Temporal Graph
Convolutional Network for Traffic Forecasting](https://arxiv.org/pdf/2006.11583.pdf).

Performs a TGCN layer, followed by a soft attention layer.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.
- `add_self_loops`: Add self loops to the graph before performing the convolution. Default `false`.
- `use_edge_weight`: If `true`, consider the edge weights in the input graph (if available).
                     If `add_self_loops=true` the new weights will be set to 1. 
                     This option is ignored if the `edge_weight` is explicitly provided in the forward pass.
                     Default `false`.
# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

# create A3TGCN layer
l = A3TGCN(2 => 6)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)      # result size (6, 5)
```
"""
@concrete struct A3TGCN <: GNNContainerLayer{(:tgcn, :dense1, :dense2)}
    in_dims::Int
    out_dims::Int
    tgcn
    dense1
    dense2
end

function A3TGCN(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32, add_self_loops = false, use_edge_weight = true) 
    in_dims, out_dims = ch
    tgcn = TGCN(ch; use_bias, init_weight, init_state, init_bias, add_self_loops, use_edge_weight)
    dense1 = Dense(out_dims, out_dims)
    dense2 = Dense(out_dims, out_dims)
    return A3TGCN(in_dims, out_dims, tgcn, dense1, dense2)
end

function (l::A3TGCN)(g, x, ps, st)
    dense1 = StatefulLuxLayer{true}(l.dense1, ps.dense1, _getstate(st, :dense1))
    dense2 = StatefulLuxLayer{true}(l.dense2, ps.dense2, _getstate(st, :dense2))
    h, st = l.tgcn(g, x, ps.tgcn, st.tgcn)
    x = dense1(h)
    x = dense2(x)
    a = NNlib.softmax(x, dims = 3)
    c = sum(a .* h , dims = 3)
    if length(size(c)) == 3
        c = dropdims(c, dims = 3)
    end
    return c, st
end

LuxCore.outputsize(l::A3TGCN) = (l.out_dims,)

function Base.show(io::IO, l::A3TGCN)
    print(io, "A3TGCN($(l.in_dims) => $(l.out_dims))")
end

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
    init_state::Function
end

function GConvGRUCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    #reset gate
    conv_x_r = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_r = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    #update gate
    conv_x_z = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_z = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    #hidden state
    conv_x_h = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_h = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    return GConvGRUCell(in_dims, out_dims, k, conv_x_r, conv_h_r, conv_x_z, conv_h_z, conv_x_h, conv_h_h, init_state)
end

function (l::GConvGRUCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
    end
    xr, st_conv_xr =  l.conv_x_r(g, x, ps.conv_x_r, st.conv_x_r)
    hr, st_conv_hr =  l.conv_h_r(g, h, ps.conv_h_r, st.conv_h_r)
    r = xr .+ hr
    r = NNlib.sigmoid_fast(r)
    xz, st_conv_x_z = l.conv_x_z(g, x, ps.conv_x_z, st.conv_x_z)
    hz, st_conv_h_z =  l.conv_h_z(g, h, ps.conv_h_z, st.conv_h_z)
    z = xz .+ hz
    z = NNlib.sigmoid_fast(z)
    xh, st_conv_x_h =  l.conv_x_h(g, x, ps.conv_x_h, st.conv_x_h)
    hh, st_conv_h_h =  l.conv_h_h(g, r .* h, ps.conv_h_h, st.conv_h_h)
    h̃ = xh .+ hh
    h̃ = NNlib.tanh_fast(h)
    h = (1 .- z).* h̃ + z.* h
    return (h, h), (conv_x_r = st_conv_xr, conv_h_r = st_conv_hr, conv_x_z = st_conv_x_z, conv_h_z = st_conv_h_z, conv_x_h = st_conv_x_h, conv_h_h = st_conv_h_h)
end

function Base.show(io::IO, l::GConvGRUCell)
    print(io, "GConvGRUCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::GConvGRUCell) = (l.out_dims,)

"""
    GConvGRU(in => out, k; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 

Graph Convolutional Gated Recurrent Unit (GConvGRU) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659).

Performs a layer of ChebConv to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Chebyshev polynomial order.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

# create layer
l = GConvGRU(2 => 5, 2)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)      # result size (5, 5)
```
"""
GConvGRU(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(GConvGRUCell(ch, k; kwargs...))

@concrete struct GConvLSTMCell <: GNNContainerLayer{(:conv_x_i, :conv_h_i, :dense_i, :conv_x_f, :conv_h_f, :dense_f, :conv_x_c, :conv_h_c, :dense_c, :conv_x_o, :conv_h_o, :dense_o)}
    in_dims::Int
    out_dims::Int
    k::Int
    conv_x_i
    conv_h_i
    dense_i
    conv_x_f
    conv_h_f
    dense_f
    conv_x_c
    conv_h_c
    dense_c
    conv_x_o
    conv_h_o
    dense_o
    init_state::Function
end

function GConvLSTMCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    #input gate
    conv_x_i = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_i = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_i = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #forget gate
    conv_x_f = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_f = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_f = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #cell gate
    conv_x_c = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_c = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_c = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    #output gate
    conv_x_o = ChebConv(in_dims => out_dims, k; use_bias, init_weight, init_bias)
    conv_h_o = ChebConv(out_dims => out_dims, k; use_bias, init_weight, init_bias)
    dense_o = Dense(out_dims, 1; use_bias, init_weight, init_bias)
    return GConvLSTMCell(in_dims, out_dims, k, conv_x_i, conv_h_i, dense_i, conv_x_f, conv_h_f, dense_f, conv_x_c, conv_h_c, dense_c, conv_x_o, conv_h_o, dense_o, init_state)
end

function (l::GConvLSTMCell)(g, (x, m), ps, st)
    if m === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
        c = l.init_state(l.out_dims, g.num_nodes)
    else 
        h, c = m
    end

    dense_i = StatefulLuxLayer{true}(l.dense_i, ps.dense_i, _getstate(st, :dense_i))
    dense_f = StatefulLuxLayer{true}(l.dense_f, ps.dense_f, _getstate(st, :dense_f))
    dense_c = StatefulLuxLayer{true}(l.dense_c, ps.dense_c, _getstate(st, :dense_c))
    dense_o = StatefulLuxLayer{true}(l.dense_o, ps.dense_o, _getstate(st, :dense_o))
    
    xi, st_conv_x_i = l.conv_x_i(g, x, ps.conv_x_i, st.conv_x_i)
    hi, st_conv_h_i = l.conv_h_i(g, h, ps.conv_h_i, st.conv_h_i)
    i = xi .+ hi .+ dense_i(c)
    i = NNlib.sigmoid_fast(i)
   
    xf, st_conv_x_f = l.conv_x_f(g, x, ps.conv_x_f, st.conv_x_f)
    hf, st_conv_h_f = l.conv_h_f(g, h, ps.conv_h_f, st.conv_h_f)
    f = xf .+ hf .+ dense_f(c)
    f =  NNlib.sigmoid_fast(f)
    
    xc, st_conv_x_c = l.conv_x_c(g, x, ps.conv_x_c, st.conv_x_c)
    hc, st_conv_h_c = l.conv_h_c(g, h, ps.conv_h_c, st.conv_h_c)
    c = f .* c + i.* NNlib.tanh_fast(xc .+ hc .+ dense_c(c))
    
    xo, st_conv_x_o = l.conv_x_o(g, x, ps.conv_x_o, st.conv_x_o)
    ho, st_conv_h_o = l.conv_h_o(g, h, ps.conv_h_o, st.conv_h_o)
    o = xo .+ ho .+ dense_o(c)
    o = NNlib.sigmoid_fast(o)
    h = o.* NNlib.tanh_fast(c)
    return (h, (h, c)), (conv_x_i = st_conv_x_i, conv_h_i = st_conv_h_i, conv_x_f = st_conv_x_f, conv_h_f = st_conv_h_f, conv_x_c = st_conv_x_c, conv_h_c = st_conv_h_c, conv_x_o = st_conv_x_o, conv_h_o = st_conv_h_o)
end

function Base.show(io::IO, l::GConvLSTMCell)
    print(io, "GConvLSTMCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::GConvLSTMCell) = (l.out_dims,)

"""
    GConvLSTM(in => out, k; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 

Graph Convolutional Long Short-Term Memory (GConvLSTM) recurrent layer from the paper [Structured Sequence Modeling with Graph Convolutional Recurrent Networks](https://arxiv.org/pdf/1612.07659). 

Performs a layer of ChebConv to model spatial dependencies, followed by a Long Short-Term Memory (LSTM) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Chebyshev polynomial order.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

# create GConvLSTM layer
l = GConvLSTM(2 => 5, 2)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)      # result size (5, 5)
```
"""
GConvLSTM(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(GConvLSTMCell(ch, k; kwargs...))

@concrete struct DCGRUCell <: GNNContainerLayer{(:dconv_u, :dconv_r, :dconv_c)}
    in_dims::Int
    out_dims::Int
    k::Int
    dconv_u
    dconv_r
    dconv_c
    init_state::Function
end

function DCGRUCell(ch::Pair{Int, Int}, k::Int; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 
    in_dims, out_dims = ch
    dconv_u = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    dconv_r = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    dconv_c = DConv((in_dims + out_dims) => out_dims, k; use_bias = use_bias, init_weight = init_weight, init_bias = init_bias)
    return DCGRUCell(in_dims, out_dims, k, dconv_u, dconv_r, dconv_c, init_state)
end

function (l::DCGRUCell)(g, (x, h), ps, st)
    if h === nothing
        h = l.init_state(l.out_dims, g.num_nodes)
    end
    h̃ = vcat(x, h)
    z, st_dconv_u = l.dconv_u(g, h̃, ps.dconv_u, st.dconv_u)
    z = NNlib.sigmoid_fast.(z)
    r, st_dconv_r = l.dconv_r(g, h̃, ps.dconv_r, st.dconv_r)
    r = NNlib.sigmoid_fast.(r)
    ĥ = vcat(x, h .* r)
    c, st_dconv_c = l.dconv_c(g, ĥ, ps.dconv_c, st.dconv_c)
    c = NNlib.tanh_fast.(c)
    h = z.* h + (1 .- z).* c
    return (h, h), (dconv_u = st_dconv_u, dconv_r = st_dconv_r, dconv_c = st_dconv_c)
end

function Base.show(io::IO, l::DCGRUCell)
    print(io, "DCGRUCell($(l.in_dims) => $(l.out_dims))")
end

LuxCore.outputsize(l::DCGRUCell) = (l.out_dims,)

"""
    DCGRU(in => out, k; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32) 

Diffusion Convolutional Recurrent Neural Network (DCGRU) layer from the paper [Diffusion Convolutional Recurrent Neural
Network: Data-driven Traffic Forecasting](https://arxiv.org/pdf/1707.01926).

Performs a Diffusion Convolutional layer to model spatial dependencies, followed by a Gated Recurrent Unit (GRU) cell to model temporal dependencies.

# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `k`: Diffusion step.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
g = rand_graph(rng, 5, 10)
x = rand(rng, Float32, 2, 5)

# create layer
l = DCGRU(2 => 5, 2)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(g, x, ps, st)      # result size (5, 5)
```
"""
DCGRU(ch::Pair{Int, Int}, k::Int; kwargs...) = GNNLux.StatefulRecurrentCell(DCGRUCell(ch, k; kwargs...))

"""
    EvolveGCNO(ch; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32)

Evolving Graph Convolutional Network (EvolveGCNO) layer from the paper [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/pdf/1902.10191).

Perfoms a Graph Convolutional layer with parameters derived from a Long Short-Term Memory (LSTM) layer across the snapshots of the temporal graph.


# Arguments

- `in`: Number of input features.
- `out`: Number of output features.
- `use_bias`: Add learnable bias. Default `true`.
- `init_weight`: Weights' initializer. Default `glorot_uniform`.
- `init_state`: Initial state of the hidden stat of the GRU layer. Default `zeros32`.
- `init_bias`: Bias initializer. Default `zeros32`.

# Examples

```julia
using GNNLux, Lux, Random

# initialize random number generator
rng = Random.default_rng()

# create data
tg = TemporalSnapshotsGNNGraph([rand_graph(rng, 10, 20; ndata = rand(rng, 4, 10)), rand_graph(rng, 10, 14; ndata = rand(rng, 4, 10)), rand_graph(rng, 10, 22; ndata = rand(rng, 4, 10))])

# create layer
l = EvolveGCNO(4 => 5)

# setup layer
ps, st = LuxCore.setup(rng, l)

# forward pass
y, st = l(tg, tg.ndata.x , ps, st)      # result size 3, size y[1] (5, 10)
```
"""
@concrete struct EvolveGCNO <: GNNLayer
    in_dims::Int
    out_dims::Int
    use_bias::Bool
    init_weight
    init_state::Function
    init_bias
end

function EvolveGCNO(ch::Pair{Int, Int}; use_bias = true, init_weight = glorot_uniform, init_state = zeros32, init_bias = zeros32)
    in_dims, out_dims = ch
    return EvolveGCNO(in_dims, out_dims, use_bias, init_weight, init_state, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::EvolveGCNO)
    weight = l.init_weight(rng, l.out_dims, l.in_dims)
    Wf = l.init_weight(rng, l.out_dims, l.in_dims)
    Uf = l.init_weight(rng, l.out_dims, l.in_dims)
    Wi = l.init_weight(rng, l.out_dims, l.in_dims)
    Ui = l.init_weight(rng, l.out_dims, l.in_dims)
    Wo = l.init_weight(rng, l.out_dims, l.in_dims)
    Uo = l.init_weight(rng, l.out_dims, l.in_dims)
    Wc = l.init_weight(rng, l.out_dims, l.in_dims)
    Uc = l.init_weight(rng, l.out_dims, l.in_dims)
    if l.use_bias
        bias = l.init_bias(rng, l.out_dims)
        Bf = l.init_bias(rng, l.out_dims, l.in_dims)
        Bi = l.init_bias(rng, l.out_dims, l.in_dims)
        Bo = l.init_bias(rng, l.out_dims, l.in_dims)
        Bc = l.init_bias(rng, l.out_dims, l.in_dims)
        return (; conv = (; weight, bias), lstm = (; Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc, Bf, Bi, Bo, Bc))
    else
        return (; conv = (; weight), lstm = (; Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc))
    end
end

function LuxCore.initialstates(rng::AbstractRNG, l::EvolveGCNO)
    h = l.init_state(rng, l.out_dims, l.in_dims)
    c = l.init_state(rng, l.out_dims, l.in_dims)
    return (; conv = (;), lstm = (; h, c))
end

function (l::EvolveGCNO)(tg::TemporalSnapshotsGNNGraph, x, ps::NamedTuple, st::NamedTuple)
    H, C = st.lstm
    W = ps.conv.weight
    m = (; ps.conv.weight, bias = _getbias(ps), 
    add_self_loops =true, use_edge_weight=true, σ = identity)

    X = map(1:tg.num_snapshots) do i
        F = NNlib.sigmoid_fast.(ps.lstm.Wf .* W .+ ps.lstm.Uf .* H .+ ps.lstm.Bf)
        I = NNlib.sigmoid_fast.(ps.lstm.Wi .* W .+ ps.lstm.Ui .* H .+ ps.lstm.Bi)
        O = NNlib.sigmoid_fast.(ps.lstm.Wo .* W .+ ps.lstm.Uo .* H .+ ps.lstm.Bo)
        C̃ = NNlib.tanh_fast.(ps.lstm.Wc .* W .+ ps.lstm.Uc .* H .+ ps.lstm.Bc)
        C = F .* C + I .* C̃
        H = O .* NNlib.tanh_fast.(C)
        W = H
        GNNlib.gcn_conv(m,tg.snapshots[i], x[i], nothing, d -> 1 ./ sqrt.(d), W)
    end
    return X, (; conv = (;), lstm = (h = H, c = C))
end
