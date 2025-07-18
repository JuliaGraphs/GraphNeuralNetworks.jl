module GNNlib

using Statistics: mean
using LinearAlgebra, Random
using MLUtils: zeros_like, ones_like
using NNlib
using NNlib: scatter, gather
using DataStructures: nlargest
using ChainRulesCore: @non_differentiable
using GNNGraphs
using .GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
                  check_num_nodes, check_num_edges,
                  EType, NType # for heteroconvs

include("utils.jl")
export reduce_nodes,
       reduce_edges,
       softmax_nodes,
       softmax_edges,
       broadcast_nodes,
       broadcast_edges,
       softmax_edge_neighbors

include("msgpass.jl")
export apply_edges,
       aggregate_neighbors,
       propagate,
       copy_xj,
       copy_xi,
       xi_dot_xj,
       xi_sub_xj,
       xj_sub_xi,
       e_mul_xj,
       w_mul_xj
 
## The following methods are defined but not exported

include("layers/basic.jl")
export dot_decoder

include("layers/conv.jl")
export agnn_conv,
       cg_conv,
       cheb_conv,
       d_conv,
       edge_conv,
       egnn_conv,
       gat_conv,
       gatv2_conv,
       gated_graph_conv,
       gcn_conv,
       gin_conv,
       gmm_conv,
       graph_conv,
       megnet_conv,
       nn_conv,
       res_gated_graph_conv,
       sage_conv,
       sg_conv,
       tag_conv,
       transformer_conv

include("layers/temporalconv.jl")
export a3tgcn_conv,
       dcgrucell_frwd,
       evolvegcnocell_frwd,
       gconvgrucell_frwd,
       gconvlstmcell_frwd,
       tgcn_frwd

include("layers/pool.jl")
export global_pool,
       global_attention_pool,
       set2set_pool,
       topk_pool,
       topk_index

# include("layers/heteroconv.jl") # no functional part at the moment

end #module
 