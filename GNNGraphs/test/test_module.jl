@testmodule GraphsTestModule begin
using FiniteDifferences: FiniteDifferences
using Reexport: @reexport
using MLUtils: MLUtils

@reexport using Random
@reexport using Statistics
@reexport using LinearAlgebra
@reexport using GNNGraphs
@reexport using Test
@reexport using Graphs
export MLUtils

export ngradient
export GRAPH_TYPES


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
