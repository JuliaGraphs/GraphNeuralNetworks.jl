module GNNlibReactantExt

using Reactant: AnyTracedRMatrix
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, COO_T, get_edge_weight

# The generic fast paths construct a sparse adjacency matrix, which Reactant
# cannot trace when its indices are dynamic. Gather/scatter keeps graph indices
# as inputs to the compiled program.
function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
                          xi, xj::AnyTracedRMatrix, e)
    propagate((xi, xj, e) -> copy_xj(xi, xj, e), g, +, xi, xj, e)
end

function GNNlib.propagate(::typeof(e_mul_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
                          xi, xj::AnyTracedRMatrix, e::AbstractVector)
    propagate((xi, xj, e) -> e_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

function GNNlib.propagate(::typeof(w_mul_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
                          xi, xj::AnyTracedRMatrix, e::Nothing)
    propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +, xi, xj, get_edge_weight(g))
end

end # module
