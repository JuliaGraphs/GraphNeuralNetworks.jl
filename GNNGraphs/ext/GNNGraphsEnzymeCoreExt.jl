module GNNGraphsEnzymeCoreExt

using GNNGraphs
using EnzymeCore: EnzymeRules

# Structure extraction returns integer indices only, hence inactive (matching the
# ChainRules `@non_differentiable` annotations). Edge weights and features are left
# to Enzyme's normal differentiation.
EnzymeRules.inactive(::typeof(GNNGraphs._findnz_idx), ::Any...) = nothing
EnzymeRules.inactive(::typeof(GNNGraphs._sparse_structure), ::Any...) = nothing

end # module
