

function global_pool(l, g::GNNGraph, x::AbstractArray)
    return reduce_nodes(l.aggr, g, x)
end

function global_attention_pool(l, g::GNNGraph, x::AbstractArray)
    α = softmax_nodes(g, l.fgate(x))
    feats = α .* l.ffeat(x)
    u = reduce_nodes(+, g, feats)
    return u
end

function topk_pool(t, X::AbstractArray)
    y = t.p' * X / norm(t.p)
    idx = topk_index(y, t.k)
    t.Ã .= view(t.A, idx, idx)
    X_ = view(X, :, idx) .* σ.(view(y, idx)')
    return X_
end

function topk_index(y::AbstractVector, k::Int)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Int) = topk_index(y', k)

function set2set_pool(l, g::GNNGraph, x::AbstractMatrix)
    n_in = size(x, 1)    
    qstar = zeros_like(x, (2*n_in, g.num_graphs))
    h = zeros_like(l.Wh, size(l.Wh, 2))
    c = zeros_like(l.Wh, size(l.Wh, 2))
    state = (h, c)
    for t in 1:l.num_iters
        q, state = l.lstm(qstar, state)                     # [n_in, n_graphs]
        qn = broadcast_nodes(g, q)                    # [n_in, n_nodes]
        α = softmax_nodes(g, sum(qn .* x, dims = 1))  # [1, n_nodes]
        r = reduce_nodes(+, g, x .* α)               # [n_in, n_graphs]
        qstar = vcat(q, r)                           # [2*n_in, n_graphs]
    end
    return qstar
end
