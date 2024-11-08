####################### GCNConv ######################################

check_gcnconv_input(g::AbstractGNNGraph{<:ADJMAT_T}, edge_weight::AbstractVector) = 
    throw(ArgumentError("Providing external edge_weight is not yet supported for adjacency matrix graphs"))

function check_gcnconv_input(g::AbstractGNNGraph, edge_weight::AbstractVector)
    if length(edge_weight) !== g.num_edges 
        throw(ArgumentError("Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"))
    end
end

check_gcnconv_input(g::AbstractGNNGraph, edge_weight::Nothing) = nothing

function gcn_conv(l, g::AbstractGNNGraph, x, edge_weight::EW, norm_fn::F, conv_weight::CW) where 
        {EW <: Union{Nothing, AbstractVector}, CW<:Union{Nothing,AbstractMatrix}, F}
    check_gcnconv_input(g, edge_weight)
    if conv_weight === nothing
        weight = l.weight
    else
        weight = conv_weight
        if size(weight) != size(l.weight)
            throw(ArgumentError("The weight matrix has the wrong size. Expected $(size(l.weight)) but got $(size(weight))"))
        end
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            # Pad weights with ones
            # TODO for ADJMAT_T the new edges are not generally at the end
            edge_weight = [edge_weight; ones_like(edge_weight, g.num_nodes)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(weight)
    if Dout < Din && !(g isa GNNHeteroGraph)
        # multiply before convolution if it is more convenient, otherwise multiply after
        # (this works only for homogenous graph)
        x = weight * x
    end

    xj, xi = expand_srcdst(g, x) # expand only after potential multiplication
    T = eltype(xi)

    if g isa GNNHeteroGraph
        din = degree(g, g.etypes[1], T; dir = :in)
        dout = degree(g, g.etypes[1], T; dir = :out)

        cout = norm_fn(dout)
        cin = norm_fn(din)
    else
        if edge_weight !== nothing
            d = degree(g, T; dir = :in, edge_weight)
        else
            d = degree(g, T; dir = :in, edge_weight = l.use_edge_weight)
        end
        cin = cout = norm_fn(d)
    end
    xj = xj .* cout'
    if edge_weight !== nothing
        x = propagate(e_mul_xj, g, +, xj = xj, e = edge_weight)
    elseif l.use_edge_weight
        x = propagate(w_mul_xj, g, +, xj = xj)
    else
        x = propagate(copy_xj, g, +, xj = xj)
    end
    x = x .* cin'
    if Dout >= Din || g isa GNNHeteroGraph
        x = weight * x
    end
    return l.σ.(x .+ l.bias)
end

# when we also have edge_weight we need to convert the graph to COO
function gcn_conv(l, g::GNNGraph{<:ADJMAT_T}, x, edge_weight::EW, norm_fn::F, conv_weight::CW) where 
        {EW <: Union{Nothing, AbstractVector}, CW<:Union{Nothing,AbstractMatrix}, F} 
    g = GNNGraph(g, graph_type = :coo)
    return gcn_conv(l, g, x, edge_weight, norm_fn, conv_weight)
end

####################### ChebConv ######################################

function cheb_conv(l, g::GNNGraph, X::AbstractMatrix{T}) where {T}
    check_num_nodes(g, X)
    @assert size(X, 1) == size(l.weight, 2) "Input feature size must match input channel size."

    L̃ = scaled_laplacian(g, eltype(X))

    Z_prev = X
    Z = X * L̃
    Y = view(l.weight, :, :, 1) * Z_prev
    Y = Y .+ view(l.weight, :, :, 2) * Z
    for k in 3:(l.k)
        Z, Z_prev = 2 * Z * L̃ - Z_prev, Z
        Y = Y .+ view(l.weight, :, :, k) * Z
    end
    return Y .+ l.bias
end

####################### GraphConv ######################################

function graph_conv(l, g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)
    m = propagate(copy_xj, g, l.aggr, xj = xj)
    x = l.weight1 * xi .+ l.weight2 * m
    return l.σ.(x .+ l.bias)
end

####################### GATConv ######################################

function gat_conv(l, g::AbstractGNNGraph, x, e::Union{Nothing, AbstractMatrix} = nothing)
    check_num_nodes(g, x)
    @assert !((e === nothing) && (l.dense_e !== nothing)) "Input edge features required for this layer"
    @assert !((e !== nothing) && (l.dense_e === nothing)) "Input edge features were not specified in the layer constructor"

    xj, xi = expand_srcdst(g, x)

    if l.add_self_loops
        @assert e===nothing "Using edge features and setting add_self_loops=true at the same time is not yet supported."
        g = add_self_loops(g)
    end

    _, chout = l.channel
    heads = l.heads

    Wxi = Wxj = l.dense_x(xj)
    Wxi = Wxj = reshape(Wxj, chout, heads, :)                   

    if xi !== xj
        Wxi = l.dense_x(xi)
        Wxi = reshape(Wxi, chout, heads, :)                   
    end

    # a hand-written message passing
    message = Fix1(gat_message, l)
    m = apply_edges(message, g, Wxi, Wxj, e)
    α = softmax_edge_neighbors(g, m.logα)
    α = dropout(α, l.dropout)
    β = α .* m.Wxj
    x = aggregate_neighbors(g, +, β)

    if !l.concat
        x = mean(x, dims = 2)
    end
    x = reshape(x, :, size(x, 3))  # return a matrix
    x = l.σ.(x .+ l.bias)

    return x
end

function gat_message(l, Wxi, Wxj, e)
    _, chout = l.channel
    heads = l.heads

    if e === nothing
        Wxx = vcat(Wxi, Wxj)
    else
        We = l.dense_e(e)
        We = reshape(We, chout, heads, :)                   # chout × nheads × nnodes
        Wxx = vcat(Wxi, Wxj, We)
    end
    aWW = sum(l.a .* Wxx, dims = 1)   # 1 × nheads × nedges
    slope = convert(eltype(aWW), l.negative_slope)
    logα = leakyrelu.(aWW, slope)
    return (; logα, Wxj)
end

####################### GATv2Conv ######################################

function gatv2_conv(l, g::AbstractGNNGraph, x, e::Union{Nothing, AbstractMatrix} = nothing)
    check_num_nodes(g, x)
    @assert !((e === nothing) && (l.dense_e !== nothing)) "Input edge features required for this layer"
    @assert !((e !== nothing) && (l.dense_e === nothing)) "Input edge features were not specified in the layer constructor"

    xj, xi = expand_srcdst(g, x)

    if l.add_self_loops
        @assert e===nothing "Using edge features and setting add_self_loops=true at the same time is not yet supported."
        g = add_self_loops(g)
    end
    _, out = l.channel
    heads = l.heads

    Wxi = reshape(l.dense_i(xi), out, heads, :)                                  # out × heads × nnodes
    Wxj = reshape(l.dense_j(xj), out, heads, :)                                  # out × heads × nnodes

    message = Fix1(gatv2_message, l)
    m = apply_edges(message, g, Wxi, Wxj, e)
    α = softmax_edge_neighbors(g, m.logα)
    α = dropout(α, l.dropout)
    β = α .* m.Wxj
    x = aggregate_neighbors(g, +, β)

    if !l.concat
        x = mean(x, dims = 2)
    end
    x = reshape(x, :, size(x, 3))
    x = l.σ.(x .+ l.bias)
    return x
end

function gatv2_message(l, Wxi, Wxj, e)
    _, out = l.channel
    heads = l.heads

    Wx = Wxi + Wxj  # Note: this is equivalent to W * vcat(x_i, x_j) as in "How Attentive are Graph Attention Networks?"
    if e !== nothing
        Wx += reshape(l.dense_e(e), out, heads, :)
    end
    slope = convert(eltype(Wx), l.negative_slope)
    logα = sum(l.a .* leakyrelu.(Wx, slope), dims = 1)   # 1 × heads × nedges
    return (; logα, Wxj)
end

####################### GatedGraphConv ######################################

function gated_graph_conv(l, g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    m, n = size(x)
    @assert m <= l.dims "number of input features must be less or equal to output features."
    if m < l.dims
        xpad = zeros_like(x, (l.dims - m, n))
        x = vcat(x, xpad)
    end
    h = x
    for i in 1:(l.num_layers)
        m = view(l.weight, :, :, i) * h
        m = propagate(copy_xj, g, l.aggr; xj = m)
        # in gru forward, hidden state is first argument, input is second
        h, _ = l.gru(h, m)
    end
    return h
end

####################### EdgeConv ######################################

function edge_conv(l, g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)

    message = Fix1(edge_conv_message, l)
    x = propagate(message, g, l.aggr; xi, xj, e = nothing)
    return x
end

edge_conv_message(l, xi, xj, e) = l.nn(vcat(xi, xj .- xi))

####################### GINConv ######################################

function gin_conv(l, g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x) 
 
    m = propagate(copy_xj, g, l.aggr, xj = xj)
    
    return l.nn((1 .+ ofeltype(xi, l.ϵ)) .* xi .+ m)
end

####################### NNConv ######################################

function nn_conv(l, g::GNNGraph, x::AbstractMatrix, e)
    check_num_nodes(g, x)
    message = Fix1(nn_conv_message, l)
    m = propagate(message, g, l.aggr, xj = x, e = e)
    return l.σ.(l.weight * x .+ m .+ l.bias)
end

function nn_conv_message(l, xi, xj, e)
    nin, nedges = size(xj)
    W = reshape(l.nn(e), (:, nin, nedges))
    xj = reshape(xj, (nin, 1, nedges)) # needed by batched_mul
    m = NNlib.batched_mul(W, xj)
    return reshape(m, :, nedges)
end

####################### SAGEConv ######################################

function sage_conv(l, g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)
    m = propagate(copy_xj, g, l.aggr, xj = xj)
    x = l.σ.(l.weight * vcat(xi, m) .+ l.bias)
    return x
end

####################### ResGatedConv ######################################

function res_gated_graph_conv(l, g::AbstractGNNGraph, x)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)

    message(xi, xj, e) = sigmoid.(xi.Ax .+ xj.Bx) .* xj.Vx

    Ax = l.A * xi
    Bx = l.B * xj
    Vx = l.V * xj

    m = propagate(message, g, +, xi = (; Ax), xj = (; Bx, Vx))

    return l.σ.(l.U * xi .+ m .+ l.bias)
end

####################### CGConv ######################################

function cg_conv(l, g::AbstractGNNGraph, x, e::Union{Nothing, AbstractMatrix} = nothing)
    check_num_nodes(g, x)
    xj, xi = expand_srcdst(g, x)
    
    if e !== nothing
        check_num_edges(g, e)
    end

    message = Fix1(cg_message, l)
    m = propagate(message, g, +, xi = xi, xj = xj, e = e)

    if l.residual
        if size(x, 1) == size(m, 1)
            m += x
        else
            @warn "number of output features different from number of input features, residual not applied."
        end
    end

    return m
end

function cg_message(l, xi, xj, e)
    if e !== nothing
        z = vcat(xi, xj, e)
    else
        z = vcat(xi, xj)
    end
    return l.dense_f(z) .* l.dense_s(z)
end

####################### AGNNConv ######################################

function agnn_conv(l, g::GNNGraph, x::AbstractMatrix)
    check_num_nodes(g, x)
    if l.add_self_loops
        g = add_self_loops(g)
    end

    xn = x ./ sqrt.(sum(x .^ 2, dims = 1))
    cos_dist = apply_edges(xi_dot_xj, g, xi = xn, xj = xn)
    α = softmax_edge_neighbors(g, l.β .* cos_dist)

    x = propagate(g, +; xj = x, e = α) do xi, xj, α
        α .* xj 
    end

    return x
end

####################### MegNetConv ######################################

function megnet_conv(l, g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
    check_num_nodes(g, x)

    ē = apply_edges(g, xi = x, xj = x, e = e) do xi, xj, e
        l.ϕe(vcat(xi, xj, e))
    end

    xᵉ = aggregate_neighbors(g, l.aggr, ē)

    x̄ = l.ϕv(vcat(x, xᵉ))

    return x̄, ē
end

####################### GMMConv ######################################

function gmm_conv(l, g::GNNGraph, x::AbstractMatrix, e::AbstractMatrix)
    (nin, ein), out = l.ch #Notational Simplicity

    @assert (ein == size(e)[1]&&g.num_edges == size(e)[2]) "Pseudo-cordinate dimension is not equal to (ein,num_edge)"

    num_edges = g.num_edges
    w = reshape(e, (ein, 1, num_edges))
    mu = reshape(l.mu, (ein, l.K, 1))

    w = @. ((w - mu)^2) / 2
    w = w .* reshape(l.sigma_inv .^ 2, (ein, l.K, 1))
    w = exp.(sum(w, dims = 1)) # (1, K, num_edge) 

    xj = reshape(l.dense_x(x), (out, l.K, :)) # (out, K, num_nodes) 

    m = propagate(e_mul_xj, g, mean, xj = xj, e = w)
    m = dropdims(mean(m, dims = 2), dims = 2) # (out, num_nodes)  

    m = l.σ.(m .+ l.bias)

    if l.residual
        if size(x, 1) == size(m, 1)
            m += x
        else
            @warn "Residual not applied : output feature is not equal to input_feature"
        end
    end

    return m
end

####################### SGCConv ######################################

# this layer is not stable enough to be supported by GNNHeteroGraph type
# due to it's looping mechanism
function sgc_conv(l, g::GNNGraph, x::AbstractMatrix{T},
                     edge_weight::EW = nothing) where
    {T, EW <: Union{Nothing, AbstractVector}}
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight)==g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            edge_weight = [edge_weight; onse_like(edge_weight, g.num_nodes)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if Dout < Din
        x = l.weight * x
    end
    if edge_weight !== nothing
        d = degree(g, T; dir = :in, edge_weight)
    else
        d = degree(g, T; dir = :in, edge_weight=l.use_edge_weight)
    end
    c = 1 ./ sqrt.(d)
    for iter in 1:(l.k)
        x = x .* c'
        if edge_weight !== nothing
            x = propagate(e_mul_xj, g, +, xj = x, e = edge_weight)
        elseif l.use_edge_weight
            x = propagate(w_mul_xj, g, +, xj = x)
        else
            x = propagate(copy_xj, g, +, xj = x)
        end
        x = x .* c'
    end
    if Dout >= Din
        x = l.weight * x
    end
    return (x .+ l.bias)
end

# when we also have edge_weight we need to convert the graph to COO
function sgc_conv(l, g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix,
                     edge_weight::AbstractVector)
    g = GNNGraph(g; graph_type=:coo)
    return sgc_conv(l, g, x, edge_weight)
end

####################### EGNNGConv ######################################

function egnn_conv(l, g::GNNGraph, h::AbstractMatrix, x::AbstractMatrix, e = nothing)
    if l.num_features.edge > 0
        @assert e!==nothing "Edge features must be provided."
    end
    @assert size(h, 1)==l.num_features.in "Input features must match layer input size."

    x_diff = apply_edges(xi_sub_xj, g, x, x)
    sqnorm_xdiff = sum(x_diff .^ 2, dims = 1)
    x_diff = x_diff ./ (sqrt.(sqnorm_xdiff) .+ 1.0f-6)

    message = Fix1(egnn_message, l)
    msg = apply_edges(message, g,
                      xi = (; h), xj = (; h), e = (; e, x_diff, sqnorm_xdiff))
    h_aggr = aggregate_neighbors(g, +, msg.h)
    x_aggr = aggregate_neighbors(g, mean, msg.x)

    hnew = l.ϕh(vcat(h, h_aggr))
    if l.residual
        h = h .+ hnew
    else
        h = hnew
    end
    x = x .+ x_aggr
    return h, x
end

function egnn_message(l, xi, xj, e)
    if l.num_features.edge > 0
        f = vcat(xi.h, xj.h, e.sqnorm_xdiff, e.e)
    else
        f = vcat(xi.h, xj.h, e.sqnorm_xdiff)
    end

    msg_h = l.ϕe(f)
    msg_x = l.ϕx(msg_h) .* e.x_diff
    return (; x = msg_x, h = msg_h)
end

######################## SGConv ######################################

# this layer is not stable enough to be supported by GNNHeteroGraph type
# due to it's looping mechanism
function sg_conv(l, g::GNNGraph, x::AbstractMatrix{T},
                edge_weight::EW = nothing) where
                {T, EW <: Union{Nothing, AbstractVector}}
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight)==g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            edge_weight = [edge_weight; ones_like(edge_weight, g.num_nodes)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if Dout < Din
        x = l.weight * x
    end
    if edge_weight !== nothing
        d = degree(g, T; dir = :in, edge_weight)
    else
        d = degree(g, T; dir = :in, edge_weight=l.use_edge_weight)
    end
    c = 1 ./ sqrt.(d)
    for iter in 1:(l.k)
        x = x .* c'
        if edge_weight !== nothing
            x = propagate(e_mul_xj, g, +, xj = x, e = edge_weight)
        elseif l.use_edge_weight
            x = propagate(w_mul_xj, g, +, xj = x)
        else
            x = propagate(copy_xj, g, +, xj = x)
        end
        x = x .* c'
    end
    if Dout >= Din
        x = l.weight * x
    end
    return (x .+ l.bias)
end

# when we also have edge_weight we need to convert the graph to COO
function sg_conv(l, g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix,
                     edge_weight::AbstractVector)
    g = GNNGraph(g; graph_type=:coo)
    return sg_conv(l, g, x, edge_weight)
end

######################## TransformerConv ######################################

function transformer_conv(l, g::GNNGraph, x::AbstractMatrix,  e::Union{AbstractMatrix, Nothing} = nothing)
    check_num_nodes(g, x)

    if l.add_self_loops
        g = add_self_loops(g)
    end

    out = l.out_dims
    heads = l.heads
    W1x = !isnothing(l.W1) ? l.W1(x) : nothing
    W2x = reshape(l.W2(x), out, heads, :)
    W3x = reshape(l.W3(x), out, heads, :)
    W4x = reshape(l.W4(x), out, heads, :)
    W6e = !isnothing(l.W6) ? reshape(l.W6(e), out, heads, :) : nothing

    message_uij = Fix1(transformer_message_uij, l)
    m = apply_edges(message_uij, g; xi = (; W3x), xj = (; W4x), e = (; W6e))
    α = softmax_edge_neighbors(g, m)
    α_val = propagate(transformer_message_main, g, +; 
                     xi = (; W3x), xj = (; W2x), e = (; W6e, α))

    h = α_val
    if l.concat
        h = reshape(h, out * heads, :)  # concatenate heads
    else
        h = mean(h, dims = 2)  # average heads
        h = reshape(h, out, :)
    end

    if !isnothing(W1x)  # root_weight
        if !isnothing(l.W5)  # gating
            β = l.W5(vcat(h, W1x, h .- W1x))
            h = β .* W1x + (1.0f0 .- β) .* h
        else
            h += W1x
        end
    end

    if l.skip_connection
        @assert size(h, 1)==size(x, 1) "In-channels must correspond to out-channels * heads if skip_connection is used"
        h += x
    end
    if !isnothing(l.BN1)
        h = l.BN1(h)
    end

    if !isnothing(l.FF)
        h1 = h
        h = l.FF(h)
        if l.skip_connection
            h += h1
        end
        if !isnothing(l.BN2)
            h = l.BN2(h)
        end
    end

    return h
end

# TODO remove l dependence
function transformer_message_uij(l, xi, xj, e)
    key = xj.W4x
    if !isnothing(e.W6e)
        key += e.W6e
    end
    uij = sum(xi.W3x .* key, dims = 1) ./ l.sqrt_out
    return uij
end

function transformer_message_main(xi, xj, e)
    val = xj.W2x
    if !isnothing(e.W6e)
        val += e.W6e
    end
    return e.α .* val
end


######################## TAGConv ######################################

function tag_conv(l, g::GNNGraph, x::AbstractMatrix{T},
                     edge_weight::EW = nothing) where
    {T, EW <: Union{Nothing, AbstractVector}}
    @assert !(g isa GNNGraph{<:ADJMAT_T} && edge_weight !== nothing) "Providing external edge_weight is not yet supported for adjacency matrix graphs"

    if edge_weight !== nothing
        @assert length(edge_weight)==g.num_edges "Wrong number of edge weights (expected $(g.num_edges) but given $(length(edge_weight)))"
    end

    if l.add_self_loops
        g = add_self_loops(g)
        if edge_weight !== nothing
            edge_weight = [edge_weight; ones_like(edge_weight, g.num_nodes)]
            @assert length(edge_weight) == g.num_edges
        end
    end
    Dout, Din = size(l.weight)
    if edge_weight !== nothing
        d = degree(g, T; dir = :in, edge_weight)
    else
        d = degree(g, T; dir = :in, edge_weight=l.use_edge_weight)
    end
    c = 1 ./ sqrt.(d)

    sum_pow = 0
    sum_total = 0
    for iter in 1:(l.k)
        x = x .* c'
        if edge_weight !== nothing
            x = propagate(e_mul_xj, g, +, xj = x, e = edge_weight)
        elseif l.use_edge_weight
            x = propagate(w_mul_xj, g, +, xj = x)
        else
            x = propagate(copy_xj, g, +, xj = x)
        end
        x = x .* c'

        # On the first iteration, initialize sum_pow with the first propagated features
        # On subsequent iterations, accumulate propagated features
        if iter == 1
            sum_pow = x
            sum_total = l.weight * sum_pow
        else
            sum_pow += x            
            # Weighted sum of features for each power of adjacency matrix
            # This applies the weight matrix to the accumulated sum of propagated features
            sum_total += l.weight * sum_pow
        end
    end

    return (sum_total .+ l.bias)
end

# when we also have edge_weight we need to convert the graph to COO
function tag_conv(l, g::GNNGraph{<:ADJMAT_T}, x::AbstractMatrix,
                     edge_weight::AbstractVector)
    g = GNNGraph(g; graph_type = :coo)
    return l(g, x, edge_weight)
end

######################## DConv ######################################

function d_conv(l, g::GNNGraph, x::AbstractMatrix)
    #A = adjacency_matrix(g, weighted = true)
    s, t = edge_index(g)
    gt = GNNGraph(t, s, get_edge_weight(g))
    deg_out = degree(g; dir = :out)
    deg_in = degree(g; dir = :in)
    deg_out = Diagonal(deg_out)
    deg_in = Diagonal(deg_in)
    
    h = l.weights[1,1,:,:] * x .+ l.weights[2,1,:,:] * x

    T0 = x
    if l.k > 1
        # T1_in = T0 * deg_in * A'
        #T1_out = T0 * deg_out' * A
        T1_out = propagate(w_mul_xj, g, +; xj = T0*deg_out')
        T1_in = propagate(w_mul_xj, gt, +; xj = T0*deg_in)
        h = h .+ l.weights[1,2,:,:] * T1_in .+ l.weights[2,2,:,:] * T1_out
    end
    for i in 2:l.k
        T2_in = propagate(w_mul_xj, gt, +; xj = T1_in*deg_in)
        T2_in = 2 * T2_in - T0
        T2_out =  propagate(w_mul_xj, g ,+; xj = T1_out*deg_out')
        T2_out = 2 * T2_out - T0
        h = h .+ l.weights[1,i,:,:] * T2_in .+ l.weights[2,i,:,:] * T2_out
        T1_in = T2_in
        T1_out = T2_out
    end
    return h .+ l.bias
end
