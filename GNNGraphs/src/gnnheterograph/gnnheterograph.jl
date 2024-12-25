
const EType = Tuple{Symbol, Symbol, Symbol} 
const NType = Symbol
const EDict{T} = Dict{EType, T}
const NDict{T} = Dict{NType, T}

"""
    GNNHeteroGraph(data; [ndata, edata, gdata, num_nodes])
    GNNHeteroGraph(pairs...; [ndata, edata, gdata, num_nodes])

A type representing a heterogeneous graph structure.
It is similar to [`GNNGraph`](@ref) but nodes and edges are of different types.

# Constructor Arguments

- `data`: A dictionary or an iterable object that maps `(source_type, edge_type, target_type)`
          triples to `(source, target)` index vectors (or to `(source, target, weight)` if also edge weights are present).
- `pairs`: Passing multiple relations as pairs is equivalent to passing `data=Dict(pairs...)`.
- `ndata`: Node features. A dictionary of arrays or named tuple of arrays.
           The size of the last dimension of each array must be given by `g.num_nodes`.
- `edata`: Edge features. A dictionary of arrays or named tuple of arrays. Default `nothing`.
           The size of the last dimension of each array must be given by `g.num_edges`. Default `nothing`.
- `gdata`: Graph features. An array or named tuple of arrays whose last dimension has size `num_graphs`. Default `nothing`.
- `num_nodes`: The number of nodes for each type. If not specified, inferred from `data`. Default `nothing`.

# Fields

- `graph`: A dictionary that maps `(source_type, edge_type, target_type)` triples to `(source, target)` index vectors.
- `num_nodes`: The number of nodes for each type.
- `num_edges`: The number of edges for each type.
- `ndata`: Node features.
- `edata`: Edge features.
- `gdata`: Graph features.
- `ntypes`: The node types.
- `etypes`: The edge types.

# Examples

```julia
julia> using GNNGraphs

julia> nA, nB = 10, 20;

julia> num_nodes = Dict(:A => nA, :B => nB);

julia> edges1 = (rand(1:nA, 20), rand(1:nB, 20))
([4, 8, 6, 3, 4, 7, 2, 7, 3, 2, 3, 4, 9, 4, 2, 9, 10, 1, 3, 9], [6, 4, 20, 8, 16, 7, 12, 16, 5, 4, 6, 20, 11, 19, 17, 9, 12, 2, 18, 12])

julia> edges2 = (rand(1:nB, 30), rand(1:nA, 30))
([17, 5, 2, 4, 5, 3, 8, 7, 9, 7  …  19, 8, 20, 7, 16, 2, 9, 15, 8, 13], [1, 1, 3, 1, 1, 3, 2, 7, 4, 4  …  7, 10, 6, 3, 4, 9, 1, 5, 8, 5])

julia> data = ((:A, :rel1, :B) => edges1, (:B, :rel2, :A) => edges2);

julia> hg = GNNHeteroGraph(data; num_nodes)
GNNHeteroGraph:
  num_nodes: (:A => 10, :B => 20)
  num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)

julia> hg.num_edges
Dict{Tuple{Symbol, Symbol, Symbol}, Int64} with 2 entries:
(:A, :rel1, :B) => 20
(:B, :rel2, :A) => 30

# Let's add some node features
julia> ndata = Dict(:A => (x = rand(2, nA), y = rand(3, num_nodes[:A])),
                    :B => rand(10, nB));

julia> hg = GNNHeteroGraph(data; num_nodes, ndata)
GNNHeteroGraph:
    num_nodes: (:A => 10, :B => 20)
    num_edges: ((:A, :rel1, :B) => 20, (:B, :rel2, :A) => 30)
    ndata:
    :A  =>  (x = 2×10 Matrix{Float64}, y = 3×10 Matrix{Float64})
    :B  =>  x = 10×20 Matrix{Float64}

# Access features of nodes of type :A
julia> hg.ndata[:A].x
2×10 Matrix{Float64}:
    0.825882  0.0797502  0.245813  0.142281  0.231253  0.685025  0.821457  0.888838  0.571347   0.53165
    0.631286  0.316292   0.705325  0.239211  0.533007  0.249233  0.473736  0.595475  0.0623298  0.159307
```

See also [`GNNGraph`](@ref) for a homogeneous graph type and [`rand_heterograph`](@ref) for a function to generate random heterographs.
"""
struct GNNHeteroGraph{T <: Union{COO_T, ADJMAT_T}} <: AbstractGNNGraph{T}
    graph::EDict{T}
    num_nodes::NDict{Int}
    num_edges::EDict{Int}
    num_graphs::Int
    graph_indicator::Union{Nothing, NDict}
    ndata::NDict{DataStore}
    edata::EDict{DataStore}
    gdata::DataStore
    ntypes::Vector{NType}
    etypes::Vector{EType}
end

GNNHeteroGraph(data; kws...) = GNNHeteroGraph(Dict(data); kws...)
GNNHeteroGraph(data::Pair...; kws...) = GNNHeteroGraph(Dict(data...); kws...)

GNNHeteroGraph() = GNNHeteroGraph(Dict{Tuple{Symbol,Symbol,Symbol}, Any}())

function GNNHeteroGraph(data::Dict; kws...)
    all(k -> k isa EType, keys(data)) || throw(ArgumentError("Keys of data must be tuples of the form `(source_type, edge_type, target_type)`"))
    return GNNHeteroGraph(Dict([k => v for (k, v) in pairs(data)]...); kws...)
end

function GNNHeteroGraph(data::EDict;
                        num_nodes = nothing,
                        graph_indicator = nothing,
                        graph_type = :coo,
                        dir = :out,
                        ndata = nothing,
                        edata = nothing,
                        gdata = (;))
    @assert graph_type ∈ [:coo, :dense, :sparse] "Invalid graph_type $graph_type requested"
    @assert dir ∈ [:in, :out]
    @assert graph_type==:coo "only :coo graph_type is supported for now"

    if num_nodes !== nothing
        num_nodes = Dict(num_nodes)
    end

    ntypes = union([[k[1] for k in keys(data)]; [k[3] for k in keys(data)]])
    etypes = collect(keys(data))

    if graph_type == :coo
        graph, num_nodes, num_edges = to_coo(data; num_nodes, dir)
    elseif graph_type == :dense
        graph, num_nodes, num_edges = to_dense(data; num_nodes, dir)
    elseif graph_type == :sparse
        graph, num_nodes, num_edges = to_sparse(data; num_nodes, dir)
    end

    num_graphs = !isnothing(graph_indicator) ?
                 maximum([maximum(gi) for gi in values(graph_indicator)]) : 1


    if length(keys(graph)) == 0
        ndata = Dict{Symbol, DataStore}()
        edata = Dict{Tuple{Symbol, Symbol, Symbol}, DataStore}()
        gdata = DataStore()
    else
        ndata = normalize_heterographdata(ndata, default_name = :x, ns = num_nodes)
        edata = normalize_heterographdata(edata, default_name = :e, ns = num_edges,
                                          duplicate_if_needed = true)
        gdata = normalize_graphdata(gdata, default_name = :u,
                                        n = num_graphs > 1 ? num_graphs : -1)
    end

    return GNNHeteroGraph(graph,
                          num_nodes, num_edges, num_graphs,
                          graph_indicator,
                          ndata, edata, gdata,
                          ntypes, etypes)
end

function show_sorted_dict(io::IO, d::Dict, compact::Bool)
    # if compact
        print(io, "Dict")
    # end
    print(io, "(")
    if !isempty(d)
        _keys = sort!(collect(keys(d)))
        for key in _keys[1:end-1]
            print(io, "$(_str(key)) => $(d[key]), ")
        end
        print(io, "$(_str(_keys[end])) => $(d[_keys[end]])")
    end
    # if length(d) == 1
    #     print(io, ",")
    # end
    print(io, ")")
end

function Base.show(io::IO, g::GNNHeteroGraph)
    print(io, "GNNHeteroGraph(")
    show_sorted_dict(io, g.num_nodes, true)
    print(io, ", ")
    show_sorted_dict(io, g.num_edges, true)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::GNNHeteroGraph)
    if get(io, :compact, false)
        print(io, "GNNHeteroGraph(")
        show_sorted_dict(io, g.num_nodes, true)
        print(io, ", ")
        show_sorted_dict(io, g.num_edges, true)
        print(io, ")")
    else
        print(io, "GNNHeteroGraph:\n  num_nodes: ")
        show_sorted_dict(io, g.num_nodes, false)
        print(io, "\n  num_edges: ")
        show_sorted_dict(io, g.num_edges, false)
        g.num_graphs > 1 && print(io, "\n  num_graphs: $(g.num_graphs)")
        if !isempty(g.ndata) && !all(isempty, values(g.ndata))
            print(io, "\n  ndata:")
            for k in sort(collect(keys(g.ndata)))
                isempty(g.ndata[k]) && continue    
                print(io, "\n\t", _str(k), "  =>  $(shortsummary(g.ndata[k]))")
            end
        end
        if !isempty(g.edata) && !all(isempty, values(g.edata))
            print(io, "\n  edata:")
            for k in sort(collect(keys(g.edata)))
                isempty(g.edata[k]) && continue
                print(io, "\n\t$k  =>  $(shortsummary(g.edata[k]))")
            end
        end
        if !isempty(g.gdata)
            print(io, "\n  gdata:\n\t")
            shortsummary(io, g.gdata)
        end
    end
end

_str(s::Symbol) = ":$s"
_str(s) = "$s"

MLUtils.numobs(g::GNNHeteroGraph) = g.num_graphs
# MLUtils.getobs(g::GNNHeteroGraph, i) = getgraph(g, i)


"""
    num_edge_types(g)

Return the number of edge types in the graph. For [`GNNGraph`](@ref)s, this is always 1.
For [`GNNHeteroGraph`](@ref)s, this is the number of unique edge types.
"""
num_edge_types(g::GNNGraph) = 1

num_edge_types(g::GNNHeteroGraph) = length(g.etypes)

"""
    num_node_types(g)

Return the number of node types in the graph. For [`GNNGraph`](@ref)s, this is always 1.
For [`GNNHeteroGraph`](@ref)s, this is the number of unique node types.
"""
num_node_types(g::GNNGraph) = 1

num_node_types(g::GNNHeteroGraph) = length(g.ntypes)

"""
    edge_type_subgraph(g::GNNHeteroGraph, edge_ts)

Return a subgraph of `g` that contains only the edges of type `edge_ts`.
Edge types can be specified as a single edge type (i.e. a tuple containing 3 symbols) or a vector of edge types.
"""
edge_type_subgraph(g::GNNHeteroGraph, edge_t::EType) = edge_type_subgraph(g, [edge_t])

function edge_type_subgraph(g::GNNHeteroGraph, edge_ts::AbstractVector{<:EType})
    for edge_t in edge_ts
        @assert edge_t in g.etypes "Edge type $(edge_t) not found in graph"
    end
    node_ts = _ntypes_from_edges(edge_ts)
    graph = Dict([edge_t => g.graph[edge_t] for edge_t in edge_ts]...)
    num_nodes = Dict([node_t => g.num_nodes[node_t] for node_t in node_ts]...)
    num_edges = Dict([edge_t => g.num_edges[edge_t] for edge_t in edge_ts]...)
    if g.graph_indicator === nothing
        graph_indicator = nothing
    else
        graph_indicator = Dict([node_t => g.graph_indicator[node_t] for node_t in node_ts]...)
    end
    ndata = Dict([node_t => g.ndata[node_t] for node_t in node_ts if node_t in keys(g.ndata)]...)
    edata = Dict([edge_t => g.edata[edge_t] for edge_t in edge_ts if edge_t in keys(g.edata)]...)
    
    return GNNHeteroGraph(graph, num_nodes, num_edges, g.num_graphs,
                          graph_indicator, ndata, edata, g.gdata,
                          node_ts, edge_ts)
end

# TODO this is not correct but Zygote cannot differentiate
# through dictionary generation
# @non_differentiable edge_type_subgraph(::Any...)

function _ntypes_from_edges(edge_ts::AbstractVector{<:EType})
    ntypes = Symbol[]
    for edge_t in edge_ts
        node1_t, _, node2_t = edge_t
        !in(node1_t, ntypes) && push!(ntypes, node1_t)
        !in(node2_t, ntypes) && push!(ntypes, node2_t)
    end
    return ntypes
end 

@non_differentiable _ntypes_from_edges(::Any...)

function Base.getindex(g::GNNHeteroGraph, node_t::NType)
    return g.ndata[node_t]
end

Base.getindex(g::GNNHeteroGraph, n1_t::Symbol, rel::Symbol, n2_t::Symbol) = g[(n1_t, rel, n2_t)]

function Base.getindex(g::GNNHeteroGraph, edge_t::EType)
    return g.edata[edge_t]
end
