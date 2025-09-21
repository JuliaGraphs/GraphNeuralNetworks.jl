### CONVERT_TO_COO REPRESENTATION ########

function to_coo(data::EDict; num_nodes = nothing, kws...)
    graph = EDict{COO_T}()
    _num_nodes = NDict{Int}()
    num_edges = EDict{Int}()
    for k in keys(data)
        d = data[k]
        @assert d isa Tuple
        if length(d) == 2
            d = (d..., nothing)
        end
        if num_nodes !== nothing
            n1 = get(num_nodes, k[1], nothing)
            n2 = get(num_nodes, k[3], nothing)
        else
            n1 = nothing
            n2 = nothing
        end
        g, nnodes, nedges = to_coo(d; hetero = true, num_nodes = (n1, n2), kws...)
        graph[k] = g
        num_edges[k] = nedges
        _num_nodes[k[1]] = max(get(_num_nodes, k[1], 0), nnodes[1])
        _num_nodes[k[3]] = max(get(_num_nodes, k[3], 0), nnodes[2])
    end
    return graph, _num_nodes, num_edges
end

function to_coo(coo::COO_T; dir = :out, num_nodes = nothing, weighted = true,
                hetero = false)
    s, t, val = coo

    if isnothing(num_nodes)
        ns = maximum(s)
        nt = maximum(t)
        num_nodes = hetero ? (ns, nt) : max(ns, nt)
    elseif num_nodes isa Integer
        ns = num_nodes
        nt = num_nodes
    elseif num_nodes isa Tuple
        ns = isnothing(num_nodes[1]) ? maximum(s) : num_nodes[1]
        nt = isnothing(num_nodes[2]) ? maximum(t) : num_nodes[2]
        num_nodes = (ns, nt)
    else
        error("Invalid num_nodes $num_nodes")
    end
    @assert isnothing(val) || length(val) == length(s)
    @assert length(s) == length(t)
    if !isempty(s)
        @assert minimum(s) >= 1
        @assert minimum(t) >= 1
        @assert maximum(s) <= ns
        @assert maximum(t) <= nt
    end
    num_edges = length(s)
    if !weighted
        coo = (s, t, nothing)
    end
    return coo, num_nodes, num_edges
end

function to_coo(A::SPARSE_T; dir = :out, num_nodes = nothing, weighted = true)
    s, t, v = findnz(A)
    if dir == :in
        s, t = t, s
    end
    num_nodes = isnothing(num_nodes) ? size(A, 1) : num_nodes
    num_edges = length(s)
    if !weighted
        v = nothing
    end
    return (s, t, v), num_nodes, num_edges
end

function _findnz_idx(A)
    nz = findall(!=(0), A) # vec of cartesian indexes
    s, t = ntuple(i -> map(t -> t[i], nz), 2)
    return s, t, nz
end

CRC.@non_differentiable _findnz_idx(A)

function to_coo(A::ADJMAT_T; dir = :out, num_nodes = nothing, weighted = true)
    s, t, nz = _findnz_idx(A)
    v = A[nz]
    if dir == :in
        s, t = t, s
    end
    num_nodes = isnothing(num_nodes) ? size(A, 1) : num_nodes
    num_edges = length(s)
    if !weighted
        v = nothing
    end
    return (s, t, v), num_nodes, num_edges
end

function to_coo(adj_list::ADJLIST_T; dir = :out, num_nodes = nothing, weighted = true)
    @assert dir ∈ [:out, :in]
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    @assert num_nodes > 0
    s = similar(adj_list[1], eltype(adj_list[1]), num_edges)
    t = similar(adj_list[1], eltype(adj_list[1]), num_edges)
    e = 0
    for i in 1:num_nodes
        for j in adj_list[i]
            e += 1
            s[e] = i
            t[e] = j
        end
    end
    @assert e == num_edges
    if dir == :in
        s, t = t, s
    end
    (s, t, nothing), num_nodes, num_edges
end

### CONVERT TO ADJACENCY MATRIX ################

### DENSE ####################

to_dense(A::AbstractSparseMatrix, x...; kws...) = to_dense(collect(A), x...; kws...)

function to_dense(A::ADJMAT_T, T = nothing; dir = :out, num_nodes = nothing,
                  weighted = true)
    @assert dir ∈ [:out, :in]
    T = T === nothing ? eltype(A) : T
    num_nodes = size(A, 1)
    @assert num_nodes == size(A, 2)
    # @assert all(x -> (x == 1) || (x == 0), A)
    num_edges = numnonzeros(A)
    if dir == :in
        A = A'
    end
    if T != eltype(A)
        A = T.(A)
    end
    if !weighted
        A = binarize(A, T)
    end
    return A, num_nodes, num_edges
end

function to_dense(adj_list::ADJLIST_T, T = nothing; dir = :out, num_nodes = nothing,
                  weighted = true)
    @assert dir ∈ [:out, :in]
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    @assert num_nodes > 0
    T = T === nothing ? eltype(adj_list[1]) : T
    A = fill!(similar(adj_list[1], T, (num_nodes, num_nodes)), 0)
    if dir == :out
        for (i, neigs) in enumerate(adj_list)
            A[i, neigs] .= 1
        end
    else
        for (i, neigs) in enumerate(adj_list)
            A[neigs, i] .= 1
        end
    end
    A, num_nodes, num_edges
end

function to_dense(coo::COO_T, T = nothing; dir = :out, num_nodes = nothing, weighted = true)
    # `dir` will be ignored since the input `coo` is always in source -> target format.
    # The output will always be a adjmat in :out format (e.g. A[i,j] denotes from i to j)
    s, t, val = coo
    n::Int = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes
    if T === nothing
        T = isnothing(val) ? eltype(s) : eltype(val)
    end
    if val === nothing || !weighted
        val = ones_like(s, T)
    end
    if eltype(val) != T
        val = T.(val)
    end

    idxs = s .+ n .* (t .- 1)

    ## using scatter instead of indexing since there could be multiple edges
    # A = fill!(similar(s, T, (n, n)), 0)
    # v = vec(A) # vec view of A
    # A[idxs] .= val # exploiting linear indexing
    v = NNlib.scatter(+, val, idxs, dstsize = n^2)
    A = reshape(v, (n, n))
    return A, n, length(s)
end

### SPARSE #############

function to_sparse(A::ADJMAT_T, T = nothing; dir = :out, num_nodes = nothing,
                   weighted = true)
    @assert dir ∈ [:out, :in]
    num_nodes = size(A, 1)
    @assert num_nodes == size(A, 2)
    T = T === nothing ? eltype(A) : T
    num_edges = A isa AbstractSparseMatrix ? nnz(A) : count(!=(0), A)
    if dir == :in
        A = A'
    end
    if T != eltype(A)
        A = T.(A)
    end
    if !(A isa AbstractSparseMatrix)
        A = sparse(A)
    end
    if !weighted
        A = binarize(A, T)
    end
    return A, num_nodes, num_edges
end

function to_sparse(adj_list::ADJLIST_T, T = nothing; dir = :out, num_nodes = nothing,
                   weighted = true)
    coo, num_nodes, num_edges = to_coo(adj_list; dir, num_nodes)
    return to_sparse(coo; num_nodes)
end

function to_sparse(coo::COO_T, T = nothing; dir = :out, num_nodes = nothing,
                   weighted = true)
    s, t, eweight = coo
    T = T === nothing ? (eweight === nothing ? eltype(s) : eltype(eweight)) : T

    if eweight === nothing || !weighted
        eweight = fill!(similar(s, T), 1)
    end

    num_nodes::Int = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes
    A = sparse(s, t, eweight, num_nodes, num_nodes)
    num_edges::Int = nnz(A)
    if eltype(A) != T
        A = T.(A)
    end
    return A, num_nodes, num_edges
end
