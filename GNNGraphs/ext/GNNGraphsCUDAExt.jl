module GNNGraphsCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNGraphs
using GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 
using SparseArrays
using Graphs

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}
const CUDA_COO_T = Tuple{T, T, V} where {T <: AnyCuArray{<:Integer}, V <: Union{Nothing, AnyCuArray}}

# Query 

GNNGraphs._rand_dense_vector(A::CUMAT_T) = CUDA.randn(size(A, 1))

# Transform

GNNGraphs.dense_zeros_like(a::CUMAT_T, T::Type, sz = size(a)) = CUDA.zeros(T, sz)


# Utils

GNNGraphs.iscuarray(x::AnyCuArray) = true

function GNNGraphs.binarize(Mat::CUSPARSE.CuSparseMatrixCSC, T::DataType = Bool)
    bin_vals = fill!(similar(nonzeros(Mat)), one(T))
    return CUSPARSE.CuSparseMatrixCSC(Mat.colPtr, rowvals(Mat), bin_vals, size(Mat))
end


function sort_edge_index(u::AnyCuArray, v::AnyCuArray)
    dev = get_device(u)
    cdev = cpu_device()
    u, v = u |> cdev, v |> cdev
    #TODO proper cuda friendly implementation
    sort_edge_index(u, v) |> dev
end

# Convert

function GNNGraphs.to_sparse(coo::CUDA_COO_T, T = nothing; dir = :out, num_nodes = nothing,
                   weighted = true, is_coalesced = false)
    s, t, eweight = coo
    T = T === nothing ? (eweight === nothing ? eltype(s) : eltype(eweight)) : T

    if eweight === nothing || !weighted
        eweight = fill!(similar(s, T), 1)
    end

    num_nodes::Int = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes
    
    # if coalesced build directly sparse coo matrix
    if is_coalesced
        A = CUDA.CUSPARSE.CuSparseMatrixCOO{T,eltype(s)}(s, t, eweight, (num_nodes, num_nodes)) 
    else
        A = sparse(s, t, eweight, num_nodes, num_nodes)
    end

    num_edges::Int = nnz(A)
    if eltype(A) != T
        A = T.(A)
    end
    return A, num_nodes, num_edges
end

end #module
