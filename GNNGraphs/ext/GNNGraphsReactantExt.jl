module GNNGraphsReactantExt

using GNNGraphs
using Reactant

GNNGraphs._graph_scalar_eltype(::Type{Reactant.TracedRNumber{T}}) where {T} =
    Reactant.unwrapped_eltype(Reactant.TracedRNumber{T})

end
