module GNNlibCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNlib: GNNlib, propagate, copy_xj, e_mul_xj, w_mul_xj
using GNNGraphs: GNNGraph, COO_T, SPARSE_T, to_dense, to_sparse
using ChainRulesCore: @non_differentiable

const CUDA_COO_T = Tuple{T, T, V} where {T <: AnyCuArray{<:Integer}, V <: Union{Nothing, AnyCuArray}}

###### PROPAGATE SPECIALIZATIONS ####################

## COPY_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
        xi, xj::AnyCuMatrix, e)
    A = _adjacency_matrix(g, eltype(xj); weighted = false)

    return xj * A
end

## E_MUL_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(e_mul_xj), g::GNNGraph{<:Union{COO_T, SPARSE_T}}, ::typeof(+),
        xi, xj::AnyCuMatrix, e::AbstractVector)
    propagate((xi, xj, e) -> e_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

## W_MUL_XJ 

## avoid the fast path on gpu until we have better cuda support
function GNNlib.propagate(::typeof(w_mul_xj), g::GNNGraph{<:COO_T}, ::typeof(+),
        xi, xj::AnyCuMatrix, e::Nothing)
    propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +, xi, xj, e)
end

# function GNNlib.propagate(::typeof(copy_xj), g::GNNGraph, ::typeof(mean), xi, xj::AbstractMatrix, e)
#     A = adjacency_matrix(g, weighted=false)
#     D = compute_degree(A)
#     return xj * A * D
# end

# # Zygote bug. Error with sparse matrix without nograd
# compute_degree(A) = Diagonal(1f0 ./ vec(sum(A; dims=2)))

# Flux.Zygote.@nograd compute_degree

## CUSTOM ADJACENCY_MATRIX IMPLEMENTATION FOR CUDA COO GRAPHS, returning dense matrix when not coalesced, more efficient 

function _adjacency_matrix(g::GNNGraph{<:CUDA_COO_T}, T::DataType = eltype(g); dir = :out,
                                 weighted = true)
    if !g.is_coalesced
        # Revisit after 
        # https://github.com/JuliaGPU/CUDA.jl/issues/1113
        A, n, m = to_dense(g.graph, T; num_nodes = g.num_nodes, weighted) # if not coalesced, construction of sparse matrix is slow
    else
        A, n, m = to_sparse(g.graph, T; num_nodes = g.num_nodes, weighted, is_coalesced = true)
    end
    @assert size(A) == (n, n)
    return dir == :out ? A : A'
end

@non_differentiable _adjacency_matrix(x...)

end #module
