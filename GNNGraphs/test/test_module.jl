@testmodule GraphsTestModule begin

using Pkg

## Uncomment below to change the default test settings
# ENV["GNN_TEST_CUDA"] = "true"
# ENV["GNN_TEST_AMDGPU"] = "true"
# ENV["GNN_TEST_Metal"] = "true"

to_test(backend) = get(ENV, "GNN_TEST_$(backend)", "false") == "true"
has_dependecies(pkgs) = all(pkg -> haskey(Pkg.project().dependencies, pkg), pkgs)
deps_dict = Dict(:CUDA => ["CUDA", "cuDNN"], :AMDGPU => ["AMDGPU"], :Metal => ["Metal"])

for (backend, deps) in deps_dict
    if to_test(backend)
        if !has_dependecies(deps)
            Pkg.add(deps)
        end
        @eval using $backend
        if backend == :CUDA
            @eval using cuDNN
        end
        @eval $backend.allowscalar(false)
    end
end
    
using FiniteDifferences: FiniteDifferences
using Reexport: @reexport
using MLUtils: MLUtils
using Zygote: gradient
using MLDataDevices: AbstractGPUDevice
@reexport using SparseArrays
@reexport using MLDataDevices
@reexport using Random
@reexport using Statistics
@reexport using LinearAlgebra
@reexport using GNNGraphs
@reexport using Test
@reexport using Graphs
export MLUtils, gradient, AbstractGPUDevice
export ngradient, GRAPH_TYPES


# Using this until https://github.com/JuliaDiff/FiniteDifferences.jl/issues/188 is fixed
function FiniteDifferences.to_vec(x::Integer)
    Integer_from_vec(v) = x
    return Int[x], Integer_from_vec
end

function ngradient(f, x...)
    fdm = FiniteDifferences.central_fdm(5, 1)
    return FiniteDifferences.grad(fdm, f, x...)
end

const GRAPH_TYPES = [:coo, :dense, :sparse]

end # module
