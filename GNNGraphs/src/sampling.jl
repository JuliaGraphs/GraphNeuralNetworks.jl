"""
    NeighborLoader(graph; num_neighbors, input_nodes, num_layers, [batch_size])

A data structure for sampling neighbors from a graph for training Graph Neural Networks (GNNs). 
It supports multi-layer sampling of neighbors for a batch of input nodes, useful for mini-batch training
originally introduced in ["Inductive Representation Learning on Large Graphs"}(https://arxiv.org/abs/1706.02216) paper.

# Fields
- `graph::GNNGraph`: The input graph.
- `num_neighbors::Vector{Int}`: A vector specifying the number of neighbors to sample per node at each GNN layer.
- `input_nodes::Vector{Int}`: A vector containing the starting nodes for neighbor sampling.
- `num_layers::Int`: The number of layers for neighborhood expansion (how far to sample neighbors).
- `batch_size::Union{Int, Nothing}`: The size of the batch. If not specified, it defaults to the number of `input_nodes`.

# Examples

```julia
julia> loader = NeighborLoader(graph; num_neighbors=[10, 5], input_nodes=[1, 2, 3], num_layers=2)

julia> batch_counter = 0

julia> for mini_batch_gnn in loader
            batch_counter += 1
            println("Batch ", batch_counter, ": Nodes in mini-batch graph: ", nv(mini_batch_gnn))
        end
```
"""
struct NeighborLoader
    graph::GNNGraph             # The input GNNGraph (graph + features from GraphNeuralNetworks.jl)
    num_neighbors::Vector{Int}  # Number of neighbors to sample per node, for each layer
    input_nodes::Vector{Int}    # Set of input nodes (starting nodes for sampling)
    num_layers::Int             # Number of layers for neighborhood expansion
    batch_size::Union{Int, Nothing}  # Optional batch size, defaults to the length of input_nodes if not given
    neighbors_cache::Dict{Int, Vector{Int}}  # Cache neighbors to avoid recomputation
end

function NeighborLoader(graph::GNNGraph; num_neighbors::Vector{Int}, input_nodes::Vector{Int}=nothing, 
                        num_layers::Int, batch_size::Union{Int, Nothing}=nothing)
    return NeighborLoader(graph, num_neighbors, input_nodes === nothing ? collect(1:graph.num_nodes) : input_nodes, num_layers, 
                            batch_size === nothing ? length(input_nodes) : batch_size, Dict{Int, Vector{Int}}())
end

# Function to get cached neighbors or compute them
function get_neighbors(loader::NeighborLoader, node::Int)
    if haskey(loader.neighbors_cache, node)
        return loader.neighbors_cache[node]
    else
        neighbors = Graphs.neighbors(loader.graph, node, dir = :in)  # Get neighbors from graph
        loader.neighbors_cache[node] = neighbors
        return neighbors
    end
end

# Function to sample neighbors for a given node at a specific layer
function sample_nbrs(loader::NeighborLoader, node::Int, layer::Int)
    neighbors = get_neighbors(loader, node)
    if isempty(neighbors)
        return Int[]
    else
        num_samples = min(loader.num_neighbors[layer], length(neighbors))  # Limit to required samples for this layer
        return rand(neighbors, num_samples)  # Randomly sample neighbors
    end
end

# Iterator protocol for NeighborLoader with lazy batch loading
function Base.iterate(loader::NeighborLoader, state=1)
    if state > length(loader.input_nodes) 
        return nothing  # End of iteration if batches are exhausted (state larger than amount of input nodes or current batch no >= batch number)
    end

    # Determine the size of the current batch
    batch_size = min(loader.batch_size, length(loader.input_nodes) - state + 1) # Conditional in case there is not enough nodes to fill the last batch
    batch_nodes = loader.input_nodes[state:state + batch_size - 1] # Each mini-batch uses different set of input nodes 

    # Set for tracking the subgraph nodes
    subgraph_nodes = Set(batch_nodes)

    for node in batch_nodes
        # Initialize current layer of nodes (starting with the node itself)
        sampled_neighbors = Set([node])

        # For each GNN layer, sample the neighborhood
        for layer in 1:loader.num_layers
            new_neighbors = Set{Int}()
            for n in sampled_neighbors
                neighbors = sample_nbrs(loader, n, layer)  # Sample neighbors of the node for this layer
                new_neighbors = union(new_neighbors, neighbors)  # Avoid duplicates in the neighbor set
            end
            sampled_neighbors = new_neighbors
            subgraph_nodes = union(subgraph_nodes, sampled_neighbors)  # Expand the subgraph with the new neighbors
        end
    end

    # Collect subgraph nodes and their features
    subgraph_node_list = collect(subgraph_nodes)

    if isempty(subgraph_node_list)
        return GNNGraph(), state + batch_size
    end

    mini_batch_gnn = Graphs.induced_subgraph(loader.graph, subgraph_node_list)  # Create a subgraph of the nodes

    # Continue iteration for the next batch
    return mini_batch_gnn, state + batch_size
end


"""
    sample_neighbors(g, nodes, K=-1; dir=:in, replace=false, dropnodes=false)

Sample neighboring edges of the given nodes and return the induced subgraph.
For each node, a number of inbound (or outbound when `dir = :out``) edges will be randomly chosen. 
If `dropnodes=false`, the graph returned will then contain all the nodes in the original graph, 
but only the sampled edges.

The returned graph will contain an edge feature `EID` corresponding to the id of the edge
in the original graph. If `dropnodes=true`, it will also contain a node feature `NID` with
the node ids in the original graph.

# Arguments

- `g`. The graph.
- `nodes`. A list of node IDs to sample neighbors from.
- `K`. The maximum number of edges to be sampled for each node.
       If -1, all the neighboring edges will be selected.
- `dir`. Determines whether to sample inbound (`:in`) or outbound (``:out`) edges (Default `:in`).
- `replace`. If `true`, sample with replacement.
- `dropnodes`. If `true`, the resulting subgraph will contain only the nodes involved in the sampled edges.
     
# Examples

```julia
julia> g = rand_graph(20, 100)
GNNGraph:
    num_nodes = 20
    num_edges = 100

julia> sample_neighbors(g, 2:3)
GNNGraph:
    num_nodes = 20
    num_edges = 9
    edata:
        EID => (9,)

julia> sg = sample_neighbors(g, 2:3, dropnodes=true)
GNNGraph:
    num_nodes = 10
    num_edges = 9
    ndata:
        NID => (10,)
    edata:
        EID => (9,)

julia> sg.ndata.NID
10-element Vector{Int64}:
  2
  3
 17
 14
 18
 15
 16
 20
  7
 10

julia> sample_neighbors(g, 2:3, 5, replace=true)
GNNGraph:
    num_nodes = 20
    num_edges = 10
    edata:
        EID => (10,)
```
"""
function sample_neighbors(g::GNNGraph{<:COO_T}, nodes, K = -1;
                          dir = :in, replace = false, dropnodes = false)
    @assert dir ∈ (:in, :out)
    _, eidlist = adjacency_list(g, nodes; dir, with_eid = true)
    for i in 1:length(eidlist)
        if replace
            k = K > 0 ? K : length(eidlist[i])
        else
            k = K > 0 ? min(length(eidlist[i]), K) : length(eidlist[i])
        end
        eidlist[i] = StatsBase.sample(eidlist[i], k; replace)
    end
    eids = reduce(vcat, eidlist)
    s, t = edge_index(g)
    w = get_edge_weight(g)
    s = s[eids]
    t = t[eids]
    w = isnothing(w) ? nothing : w[eids]

    edata = getobs(g.edata, eids)
    edata.EID = eids

    num_edges = length(eids)

    if !dropnodes
        graph = (s, t, w)

        gnew = GNNGraph(graph,
                        g.num_nodes, num_edges, g.num_graphs,
                        g.graph_indicator,
                        g.ndata, edata, g.gdata)
    else
        nodes_other = dir == :in ? setdiff(s, nodes) : setdiff(t, nodes)
        nodes_all = [nodes; nodes_other]
        nodemap = Dict(n => i for (i, n) in enumerate(nodes_all))
        s = [nodemap[s] for s in s]
        t = [nodemap[t] for t in t]
        graph = (s, t, w)
        graph_indicator = g.graph_indicator !== nothing ? g.graph_indicator[nodes_all] :
                          nothing
        num_nodes = length(nodes_all)
        ndata = getobs(g.ndata, nodes_all)
        ndata.NID = nodes_all

        gnew = GNNGraph(graph,
                        num_nodes, num_edges, g.num_graphs,
                        graph_indicator,
                        ndata, edata, g.gdata)
    end
    return gnew
end


"""
    induced_subgraph(graph, nodes)

Generates a subgraph from the original graph using the provided `nodes`. 
The function includes the nodes' neighbors and creates edges between nodes that are connected in the original graph. 
If a node has no neighbors, an isolated node will be added to the subgraph. 
Returns A new `GNNGraph` containing the subgraph with the specified nodes and their features.

# Arguments

- `graph`. The original GNNGraph containing nodes, edges, and node features.
- `nodes``. A vector of node indices to include in the subgraph.
     
# Examples

```julia
julia> s = [1, 2]
2-element Vector{Int64}:
 1
 2

julia> t = [2, 3]
2-element Vector{Int64}:
 2
 3

julia> graph = GNNGraph((s, t), ndata = (; x=rand(Float32, 32, 3), y=rand(Float32, 3)), edata = rand(Float32, 2))
GNNGraph:
  num_nodes: 3
  num_edges: 2
  ndata:
        y = 3-element Vector{Float32}
        x = 32×3 Matrix{Float32}
  edata:
        e = 2-element Vector{Float32}

julia> nodes = [1, 2]
2-element Vector{Int64}:
 1
 2

julia> subgraph = Graphs.induced_subgraph(graph, nodes)
GNNGraph:
  num_nodes: 2
  num_edges: 1
  ndata:
        y = 2-element Vector{Float32}
        x = 32×2 Matrix{Float32}
  edata:
        e = 1-element Vector{Float32}
```
"""
function Graphs.induced_subgraph(graph::GNNGraph, nodes::Vector{Int})
    if isempty(nodes)
        return GNNGraph()  # Return empty graph if no nodes are provided
    end

    node_map = Dict(node => i for (i, node) in enumerate(nodes))

    edge_list = [collect(t) for t in zip(edge_index(graph)[1],edge_index(graph)[2])]

    # Collect edges to add
    source = Int[]
    target = Int[]
    eindices = Int[]
    for node in nodes
        neighbors = Graphs.neighbors(graph, node, dir = :in)
        for neighbor in neighbors
            if neighbor in keys(node_map)
                push!(target, node_map[node])
                push!(source, node_map[neighbor])
                eindex = findfirst(x -> x == [neighbor, node], edge_list)
                push!(eindices, eindex)
            end
        end
    end

    # Extract features for the new nodes
    new_ndata = getobs(graph.ndata, nodes)
    new_edata = getobs(graph.edata, eindices)

    return GNNGraph(source, target, num_nodes = length(node_map), ndata = new_ndata, edata = new_edata) 
end
