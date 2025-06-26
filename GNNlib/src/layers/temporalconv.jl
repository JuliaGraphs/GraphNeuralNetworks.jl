function a3tgcn_conv(a3tgcn, g::GNNGraph, x::AbstractArray)
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


function gconvgrucell_frwd(cell, g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    # reset gate
    r = cell.conv_x_r(g, x) .+ cell.conv_h_r(g, h)
    r = NNlib.sigmoid_fast(r)
    # update gate
    z = cell.conv_x_z(g, x) .+ cell.conv_h_z(g, h)
    z = NNlib.sigmoid_fast(z)
    # new gate
    h̃ = cell.conv_x_h(g, x) .+ cell.conv_h_h(g, r .* h)
    h̃ = NNlib.tanh_fast(h̃)
    h = (1 .- z) .* h̃ .+ z .* h 
    return h, h
end


function gconvlstmcell_frwd(cell, g::GNNGraph, x::AbstractMatrix, (h, c))
    # input gate
    i = cell.conv_x_i(g, x) .+ cell.conv_h_i(g, h) .+ cell.w_i .* c .+ cell.b_i 
    i = NNlib.sigmoid_fast(i)
    # forget gate
    f = cell.conv_x_f(g, x) .+ cell.conv_h_f(g, h) .+ cell.w_f .* c .+ cell.b_f
    f = NNlib.sigmoid_fast(f)
    # cell state
    c = f .* c .+ i .* NNlib.tanh_fast(cell.conv_x_c(g, x) .+ cell.conv_h_c(g, h) .+ cell.w_c .* c .+ cell.b_c)
    # output gate
    o = cell.conv_x_o(g, x) .+ cell.conv_h_o(g, h) .+ cell.w_o .* c .+ cell.b_o
    o = NNlib.sigmoid_fast(o)
    h =  o .* NNlib.tanh_fast(c)
    return h, (h, c)
end

function dcgrucell_frwd(cell, g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    h̃ = vcat(x, h)
    z = cell.dconv_u(g, h̃)
    z = NNlib.sigmoid_fast.(z)
    r = cell.dconv_r(g, h̃)
    r = NNlib.sigmoid_fast.(r)
    ĥ = vcat(x, h .* r)
    c = cell.dconv_c(g, ĥ)
    c = NNlib.tanh_fast.(c)
    h = z.* h + (1 .- z) .* c
    return h, h
end


function evolvegcnocell_frwd(cell, g::GNNGraph, x::AbstractMatrix, state)
    weight, state_lstm = cell.lstm(state.weight, state.lstm)
    x = cell.conv(g, x, conv_weight = reshape(weight, (cell.out, cell.in)))
    return x, (; weight, lstm = state_lstm)
end


function tgcncell_frwd(cell, g::GNNGraph, x::AbstractMatrix, h::AbstractMatrix)
    z = cell.conv_z(g, x)
    z = cell.dense_z(vcat(z, h))
    r = cell.conv_r(g, x)
    r = cell.dense_r(vcat(r, h))
    h̃ = cell.conv_h(g, x)
    h̃ = cell.dense_h(vcat(h̃, r .* h))
    h = (1 .- z) .* h .+ z .* h̃
    return h, h
end