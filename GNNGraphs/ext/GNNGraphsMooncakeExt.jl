module GNNGraphsMooncakeExt

using GNNGraphs
import Mooncake

# Without these rules Mooncake traces into graph-structure code and fails on CUDA:
# host→device index copies in `add_self_loops`, `findall` on `CuArray` in
# `_findnz_idx`, bounds checks in `_edge_values` (Mooncake's CUDA `getindex` rule
# covers only float/complex arrays). No gradient is lost: all three yield integers,
# whose Mooncake tangent is `NoTangent` anyway. Float edge values stay on the AD path.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(add_self_loops), GNNGraph}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(GNNGraphs._findnz_idx), Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(GNNGraphs._edge_values), AbstractMatrix{<:Integer}, Any, Any}

end # module
