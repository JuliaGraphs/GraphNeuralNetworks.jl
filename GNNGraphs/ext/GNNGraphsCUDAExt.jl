module GNNGraphsCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNGraphs
using GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 
using SparseArrays

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

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


end #module
