module GNNGraphsEnzymeCoreExt

using GNNGraphs
import EnzymeCore

# `scaled_laplacian` is `@non_differentiable` for ChainRules (see query.jl), but Enzyme
# does not read those declarations and would differentiate its Krylov eigensolve.
# It depends only on the graph, never on the node features, so no gradient is lost.
EnzymeCore.EnzymeRules.inactive(::typeof(GNNGraphs.scaled_laplacian), args...) = nothing

end # module
