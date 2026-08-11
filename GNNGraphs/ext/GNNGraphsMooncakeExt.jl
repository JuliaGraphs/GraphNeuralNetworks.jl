module GNNGraphsMooncakeExt

using GNNGraphs
import Mooncake

# Integer graph-structure extraction is non-differentiable, mirroring the
# ChainRules `@non_differentiable` annotations used by Zygote. Without these
# rules Mooncake traces into the functions and fails on CUDA (CPU→GPU index
# copies in `add_self_loops`, `findall` on `CuArray` in `_findnz_idx`,
# bounds checks of integer-array indexing in `_edge_values`). Float edge
# values in `_edge_values` stay differentiable.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(add_self_loops), GNNGraph}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(GNNGraphs._findnz_idx), Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(GNNGraphs._edge_values), AbstractMatrix{<:Integer}, Any, Any}

end # module
