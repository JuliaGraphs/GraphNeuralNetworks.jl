module GNNlibReactantExt

using Reactant: AnyTracedRMatrix
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, COO_T, get_edge_weight

###### PROPAGATE SPECIALIZATIONS ####################

# The generic fast paths multiply by a sparse adjacency matrix. Reactant cannot trace one whose
# eltype is traced (scalar indexing) or that is built from traced edge indices (stack overflow).
# Fall back to gather/scatter message passing, which lowers to XLA gather/scatter ops.

## COPY_XJ

function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
        xi, xj::AnyTracedRMatrix, e)
    propagate((xi, xj, e) -> copy_xj(xi, xj, e), g, +, xi, xj, e)
end

## E_MUL_XJ

function GNNlib.propagate(::typeof(e_mul_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
        xi, xj::AnyTracedRMatrix, e::AbstractVector)
    propagate((xi, xj, e) -> e_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

## W_MUL_XJ

# The weights come from the graph here, so they have to be passed on explicitly:
# `w_mul_xj` with `e = nothing` would return unweighted messages.
function GNNlib.propagate(::typeof(w_mul_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
        xi, xj::AnyTracedRMatrix, e::Nothing)
    propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +, xi, xj, get_edge_weight(g))
end

end #module
