"""
    rand_heterograph([rng,] n, m; bidirected=false, kws...)

Construct an [`GNNHeteroGraph`](@ref) with random edges and with number of nodes and edges 
specified by `n` and `m` respectively. `n` and `m` can be any iterable of pairs
specifing node/edge types and their numbers.

Pass a random number generator as a first argument to make the generation reproducible.

Setting `bidirected=true` will generate a bidirected graph, i.e. each edge will have a reverse edge.
Therefore, for each edge type `(:A, :rel, :B)` a corresponding reverse edge type `(:B, :rel, :A)`
will be generated.

Additional keyword arguments will be passed to the [`GNNHeteroGraph`](@ref) constructor.

# Examples

```jldoctest
julia> g = rand_heterograph((:user => 10, :movie => 20),
                            (:user, :rate, :movie) => 30)
GNNHeteroGraph:
  num_nodes: Dict(:movie => 20, :user => 10)
  num_edges: Dict((:user, :rate, :movie) => 30)
```
"""
function rand_heterograph end

# for generic iterators of pairs
rand_heterograph(n, m; kws...) = rand_heterograph(Dict(n), Dict(m); kws...)
rand_heterograph(rng::AbstractRNG, n, m; kws...) = rand_heterograph(rng, Dict(n), Dict(m); kws...)

function  rand_heterograph(n::NDict, m::EDict; seed=-1, kws...)
    if seed != -1
        Base.depwarn("Keyword argument `seed` is deprecated, pass an rng as first argument instead.", :rand_heterograph)
        rng = MersenneTwister(seed)
    else
        rng = Random.default_rng()
    end
    return rand_heterograph(rng, n, m; kws...)
end

function rand_heterograph(rng::AbstractRNG, n::NDict, m::EDict; bidirected::Bool = false, kws...)
    if bidirected
        return _rand_bidirected_heterograph(rng, n, m; kws...)
    end
    graphs = Dict(k => _rand_edges(rng, (n[k[1]], n[k[3]]), m[k]) for k in keys(m))
    return GNNHeteroGraph(graphs; num_nodes = n, kws...)
end

function _rand_bidirected_heterograph(rng::AbstractRNG, n::NDict, m::EDict; kws...)
    for k in keys(m)
        if reverse(k) ∈ keys(m)
            @assert m[k] == m[reverse(k)] "Number of edges must be the same in reverse edge types for bidirected graphs."
        else
            m[reverse(k)] = m[k]
        end
    end
    graphs = Dict{EType, Tuple{Vector{Int}, Vector{Int}, Nothing}}()
    for k in keys(m)
        reverse(k) ∈ keys(graphs) && continue
        s, t, val =  _rand_edges(rng, (n[k[1]], n[k[3]]), m[k])
        graphs[k] = s, t, val
        graphs[reverse(k)] = t, s, val
    end
    return GNNHeteroGraph(graphs; num_nodes = n, kws...)
end


"""
    rand_bipartite_heterograph([rng,] 
                               (n1, n2), (m12, m21); 
                               bidirected = true, 
                               node_t = (:A, :B), 
                               edge_t = :to, 
                               kws...)

Construct an [`GNNHeteroGraph`](@ref) with random edges representing a bipartite graph.
The graph will have two types of nodes, and edges will only connect nodes of different types.

The first argument is a tuple `(n1, n2)` specifying the number of nodes of each type.
The second argument is a tuple `(m12, m21)` specifying the number of edges connecting nodes of type `1` to nodes of type `2` 
and vice versa.

The type of nodes and edges can be specified with the `node_t` and `edge_t` keyword arguments,
which default to `(:A, :B)` and `:to` respectively.

If `bidirected=true` (default), the reverse edge of each edge will be present. In this case
`m12 == m21` is required.

A random number generator can be passed as the first argument to make the generation reproducible.

Additional keyword arguments will be passed to the [`GNNHeteroGraph`](@ref) constructor.

See [`rand_heterograph`](@ref) for a more general version.

# Examples

```julia
julia> g = rand_bipartite_heterograph((10, 15), 20)
GNNHeteroGraph:
  num_nodes: (:A => 10, :B => 15)
  num_edges: ((:A, :to, :B) => 20, (:B, :to, :A) => 20)

julia> g = rand_bipartite_heterograph((10, 15), (20, 0), node_t=(:user, :item), edge_t=:-, bidirected=false)
GNNHeteroGraph:
  num_nodes: Dict(:item => 15, :user => 10)
  num_edges: Dict((:item, :-, :user) => 0, (:user, :-, :item) => 20)
```
"""
rand_bipartite_heterograph(n, m; kws...) = rand_bipartite_heterograph(Random.default_rng(), n, m; kws...)

function rand_bipartite_heterograph(rng::AbstractRNG, (n1, n2)::NTuple{2,Int}, m; bidirected=true, 
                        node_t = (:A, :B), edge_t::Symbol = :to, kws...)
    if m isa Integer
        m12 = m21 = m
    else
        m12, m21 = m
    end

    return rand_heterograph(rng, Dict(node_t[1] => n1, node_t[2] => n2), 
                            Dict((node_t[1], edge_t, node_t[2]) => m12, (node_t[2], edge_t, node_t[1]) => m21); 
                            bidirected, kws...)
end

