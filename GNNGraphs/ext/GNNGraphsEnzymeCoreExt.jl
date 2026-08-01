module GNNGraphsEnzymeCoreExt

using GNNGraphs
using EnzymeCore: EnzymeRules

# Storage conversion is structural and non-differentiable, matching its ChainRules
# treatment; the rule also keeps Enzyme out of Union-heavy code it cannot compile.
EnzymeRules.inactive(::typeof(GNNGraphs._to_coo_graph), ::GNNGraph) = nothing

end # module
