module GNNlibMetalExt

using Metal
using Random, Statistics, LinearAlgebra
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, COO_T, SPARSE_T, adjacency_matrix
using ChainRulesCore: @non_differentiable

const METAL_COO_T = Tuple{T, T, V} where {T <: MtlVector{<:Integer}, V <: Union{Nothing, MtlVector}}

###### PROPAGATE SPECIALIZATIONS ####################

## COPY_XJ 

## Metal does not support sparse arrays yet and neither scater.
## Have to use dense adjacency matrix multiplication for now.
function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:METAL_COO_T}, ::typeof(+),
        xi, xj::AbstractMatrix, e)
    A = adjacency_matrix(g, eltype(xj), weighted=false, fmt=:dense)
    return xj * A
end

end #module
