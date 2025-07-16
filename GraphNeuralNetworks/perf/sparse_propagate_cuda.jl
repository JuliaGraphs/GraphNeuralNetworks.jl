# # Activate the perf environment
# using Pkg
# Pkg.activate(@__DIR__)
# Pkg.develop(path=joinpath(@__DIR__, "..", "..", "GNNGraphs"))
# Pkg.develop(path=joinpath(@__DIR__, "..", "..", "GNNlib"))
# Pkg.develop(path=joinpath(@__DIR__, ".."))
# Pkg.instantiate()
using SparseArrays
using GraphNeuralNetworks
using BenchmarkTools
import Random: seed!
using LinearAlgebra
using Flux, CUDA

# ENV["JULIA_DEBUG"] = "GraphNeuralNetworks,GNNlib,GNNlibCUDAExt,GNNGraphs,GNNGraphsCUDAExt,CUDA" # packages with debugging enabled, don't put a whitespace between the package names

function prop_copy_xj(graph_type, sp_p, n, feat_size)
    A = sprand(n, n, sp_p)
    b = rand(1, n)
    B = rand(feat_size, n)
    g = GNNGraph(A,
                 ndata = (; b = b, B = B),
                 edata = (; A = reshape(A.nzval, 1, :)),
                 graph_type = graph_type) |> dev
    printstyled("propagate copy_xj for graph type: $graph_type", "\n", color=:yellow)
    CUDA.@sync propagate(copy_xj, g, +; xj = g.ndata.B) # run once to compile before benchmarking
    # @profview for _ in 1:1000
    #     propagate(copy_xj, g, +; xj = g.ndata.B)
    # end
    @btime CUDA.@sync propagate($copy_xj, $g, +; xj = $g.ndata.B) # using spmm for :sparse
    printstyled("gather/scatter propagate copy_xj for graph type: $graph_type", "\n", color=:yellow)
    CUDA.@sync propagate((xi, xj, e) -> xj, g, +; xj = g.ndata.B) # run once to compile before benchmarking
    @btime CUDA.@sync propagate((xi, xj, e) -> xj, $g, +; xj = $g.ndata.B) # using gather/scatter
    return nothing
end

function prop_w_mul_xj(graph_type, sp_p, n, feat_size)
    A = sprand(n, n, sp_p)
    b = rand(1, n)
    B = rand(feat_size, n)
    g = GNNGraph(A,
                 ndata = (; b = b, B = B),
                 edata = (; A = reshape(A.nzval, 1, :)),
                 graph_type = graph_type) |> dev
    printstyled("propagate w_mul_xj for graph type: $graph_type", "\n", color=:yellow)
    CUDA.@sync propagate(w_mul_xj, g, +; xj = g.ndata.B) # run once to compile before benchmarking
    @btime CUDA.@sync propagate($w_mul_xj, $g, +; xj = $g.ndata.B) # using spmm for :sparse
    printstyled("gather/scatter propagate w_mul_xj for graph type: $graph_type", "\n", color=:yellow)
    CUDA.@sync propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), g, +; xj = g.ndata.B) # run once to compile before benchmarking
    @btime CUDA.@sync propagate((xi, xj, e) -> w_mul_xj(xi, xj, e), $g, +; xj = $g.ndata.B) # using gather/scatter
    return nothing
end

seed!(0)
dev = gpu_device()
println("Device: ", dev)
feat_size = 128
# test for :sparse graph_type
for n in (32, 128, 1024)
    for sp_p in (0.01, 0.1, 0.9)
        printstyled("n = $n, feat_size = $feat_size, sparsity = $sp_p\n", color=:blue)
        prop_copy_xj(:sparse, sp_p, n, feat_size)
        println()
    end
end
