module GNNGraphsReactantExt

using GNNGraphs
using Reactant: TracedRNumber

GNNGraphs._check_index_eltype(::Type{<:TracedRNumber{T}}) where {T <: Integer} = true

GNNGraphs._graph_scalar_eltype(::Type{TracedRNumber{T}}) where {T} = T

end
