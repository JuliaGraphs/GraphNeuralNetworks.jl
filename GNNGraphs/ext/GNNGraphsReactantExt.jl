module GNNGraphsReactantExt

using GNNGraphs
using Reactant: TracedRNumber

# Layers compute degrees with the feature eltype, which inside a Reactant trace is
# `TracedRNumber{T}`. Allocating with that eltype from host index vectors would create a
# `Vector{TracedRNumber{T}}`, so compute with the underlying number type instead.
function GNNGraphs._degree((s, t)::Tuple, ::Type{TracedRNumber{T}}, dir::Symbol,
                           edge_weight::Nothing, num_nodes::Int) where {T}
    return GNNGraphs._degree((s, t), T, dir, edge_weight, num_nodes)
end

function GNNGraphs._degree((s, t)::Tuple, ::Type{TracedRNumber{T}}, dir::Symbol,
                           edge_weight::AbstractVector, num_nodes::Int) where {T}
    return GNNGraphs._degree((s, t), T, dir, edge_weight, num_nodes)
end

end # module
