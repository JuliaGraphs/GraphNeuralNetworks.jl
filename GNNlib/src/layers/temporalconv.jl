#=
Framework-agnostic forward passes for the recurrent temporal cells.

Each function receives a lightweight carrier `l` exposing the cell's sub-modules
and parameters, the graph `g`, the input node features `x`, and the current
recurrent state, and returns the updated state. Both the Flux (`GraphNeuralNetworks`)
and the Lux (`GNNLux`) frontends build such a carrier and call into these
functions, so the recurrence math lives in a single place.
=#

"""
    tgcn(l, cz, cr, ch, h)

GRU gating of the T-GCN cell. `cz`, `cr`, `ch` are the spatial-convolution
outputs of the three gates (computed by the frontend), `h` is the current
hidden state, and `l` carries the three gate `Dense` sub-modules
(`l.dense_z`, `l.dense_r`, `l.dense_h`, each exposing `weight`, `bias`, `σ`).
Returns the updated hidden state.
"""
function tgcn(l, cz, cr, ch, h)
    z = l.dense_z.σ.(l.dense_z.weight * vcat(cz, h) .+ l.dense_z.bias)
    r = l.dense_r.σ.(l.dense_r.weight * vcat(cr, h) .+ l.dense_r.bias)
    h̃ = l.dense_h.σ.(l.dense_h.weight * vcat(ch, r .* h) .+ l.dense_h.bias)
    h = (1 .- z) .* h .+ z .* h̃
    return h
end

"""
    gconv_gru(l, g, x, h)

Forward pass of the GConvGRU cell. `l` carries the six `ChebConv` sub-modules
(`conv_x_r`, `conv_h_r`, `conv_x_z`, `conv_h_z`, `conv_x_h`, `conv_h_h`).
Returns the updated hidden state.
"""
function gconv_gru(l, g::GNNGraph, x, h)
    r = NNlib.sigmoid_fast.(cheb_conv(l.conv_x_r, g, x) .+ cheb_conv(l.conv_h_r, g, h))
    z = NNlib.sigmoid_fast.(cheb_conv(l.conv_x_z, g, x) .+ cheb_conv(l.conv_h_z, g, h))
    h̃ = NNlib.tanh_fast.(cheb_conv(l.conv_x_h, g, x) .+ cheb_conv(l.conv_h_h, g, r .* h))
    h = (1 .- z) .* h̃ .+ z .* h
    return h
end

"""
    gconv_lstm(l, g, x, h, c)

Forward pass of the GConvLSTM cell. `l` carries the eight `ChebConv` sub-modules
and the four peephole scalings/biases (`w_i`, `b_i`, `w_f`, `b_f`, `w_c`, `b_c`,
`w_o`, `b_o`). Returns the updated `(h, c)` state.
"""
function gconv_lstm(l, g::GNNGraph, x, h, c)
    # input gate
    i = cheb_conv(l.conv_x_i, g, x) .+ cheb_conv(l.conv_h_i, g, h) .+ l.w_i .* c .+ l.b_i
    i = NNlib.sigmoid_fast.(i)
    # forget gate
    f = cheb_conv(l.conv_x_f, g, x) .+ cheb_conv(l.conv_h_f, g, h) .+ l.w_f .* c .+ l.b_f
    f = NNlib.sigmoid_fast.(f)
    # cell state
    c = f .* c .+ i .* NNlib.tanh_fast.(cheb_conv(l.conv_x_c, g, x) .+ cheb_conv(l.conv_h_c, g, h) .+ l.w_c .* c .+ l.b_c)
    # output gate
    o = cheb_conv(l.conv_x_o, g, x) .+ cheb_conv(l.conv_h_o, g, h) .+ l.w_o .* c .+ l.b_o
    o = NNlib.sigmoid_fast.(o)
    h = o .* NNlib.tanh_fast.(c)
    return h, c
end

"""
    dcgru(l, g, x, h)

Forward pass of the DCGRU cell. `l` carries the three `DConv` sub-modules
(`dconv_u`, `dconv_r`, `dconv_c`). Returns the updated hidden state.
"""
function dcgru(l, g::GNNGraph, x, h)
    h̃ = vcat(x, h)
    z = NNlib.sigmoid_fast.(d_conv(l.dconv_u, g, h̃))
    r = NNlib.sigmoid_fast.(d_conv(l.dconv_r, g, h̃))
    ĥ = vcat(x, h .* r)
    c = NNlib.tanh_fast.(d_conv(l.dconv_c, g, ĥ))
    h = z .* h .+ (1 .- z) .* c
    return h
end
