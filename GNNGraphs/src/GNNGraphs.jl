module GNNGraphs

using SparseArrays
import Graphs
using Graphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree, 
              has_self_loops, is_directed, induced_subgraph, has_edge
import NearestNeighbors
import NNlib
import StatsBase
import KrylovKit
import ChainRulesCore as CRC
using LinearAlgebra, Random, Statistics
import MLUtils
using MLUtils: getobs, numobs, ones_like, zeros_like, chunk, batch, rand_like
using MLDataDevices: get_device, cpu_device, CPUDevice
using Functors: @functor

include("chainrules.jl") # hacks for differentiability

include("datastore.jl")
export DataStore

include("abstracttypes.jl")
export AbstractGNNGraph

include("gnngraph.jl")
export GNNGraph,
       node_features,
       edge_features,
       graph_features

include("gnnheterograph/gnnheterograph.jl")
export GNNHeteroGraph,
       num_edge_types,
       num_node_types,
       edge_type_subgraph

include("temporalsnapshotsgnngraph.jl")
export TemporalSnapshotsGNNGraph,
       add_snapshot,
       # add_snapshot!,
       remove_snapshot
       # remove_snapshot!

include("query.jl")
include("gnnheterograph/query.jl")
export adjacency_list,
       edge_index,
       get_edge_weight,
       get_graph_type,
       graph_indicator,
       has_multi_edges,
       is_directed,
       is_bidirected,
       normalized_laplacian,
       scaled_laplacian,
       laplacian_lambda_max,
# from Graphs.jl
       adjacency_matrix,
       degree,
       has_edge, 
       has_isolated_nodes,
       has_self_loops,
       inneighbors,
       outneighbors,
       khop_adj

include("transform.jl")
include("gnnheterograph/transform.jl")
export add_nodes,
       add_edges,
       add_self_loops,
       getgraph,
       negative_sample,
       rand_edge_split,
       remove_self_loops,
       remove_edges, 
       remove_multi_edges,
       set_edge_weight,
       to_bidirected,
       to_unidirected,
       random_walk_pe,
       perturb_edges,
       remove_nodes,
       ppr_diffusion,
# from MLUtils.jl
       batch,
       unbatch,
# from SparseArrays
       blockdiag

include("generate.jl")
include("gnnheterograph/generate.jl")
export rand_graph,
       rand_heterograph,
       rand_bipartite_heterograph,
       knn_graph,
       radius_graph,
       rand_temporal_radius_graph,
       rand_temporal_hyperbolic_graph

include("operators.jl")
# Base.intersect

include("convert.jl")
include("utils.jl")
include("gnnheterograph/utils.jl")
export sort_edge_index, color_refinement

include("gatherscatter.jl")
# _gather, _scatter

include("mldatasets.jl")
export mldataset2gnngraph

include("deprecations.jl")

include("sampling.jl")
export NeighborLoader, sample_neighbors, 
      induced_subgraph # from Graphs.jl

end #module
