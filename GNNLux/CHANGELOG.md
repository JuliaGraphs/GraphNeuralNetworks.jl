# Changelog

All notable changes to the GNNLux.jl package (the Lux frontend) will be
documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work in progress towards the `0.2.0` release.

### Added
- Added pooling layers ([#576]).
- `TGCN` now supports non-linear activation functions ([#596]).

### Changed
- Bumped the NNlib, OneHotArrays, StableRNGs, and DocumenterInterLinks compat bounds ([#687], [#686], [#684], [#685]).

## [0.1.1] - 2024-12-09

Documentation release: added the "Hands On" tutorial, a version selector, and
general docs improvements ([#549], [#543], [#542], [#539]). No API changes.

## [0.1.0] - 2024-12-02

Initial release of the Lux-based frontend for GraphNeuralNetworks.jl, providing
Lux implementations of the graph convolutional, pooling, and temporal layers
(e.g. `GCNConv`, `GraphConv`, `SAGEConv`, `GATConv`, `GMMConv`, `NNConv`,
`ResGatedGraphConv`, and the temporal layers `TGCN`, `GConvGRU`, `GConvLSTM`,
`DCGRU`, `EvolveGCNO`) that share the message-passing implementations in GNNlib.

[Unreleased]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/compare/GNNLux-v0.1.1...HEAD
[0.1.1]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNLux-v0.1.1
[0.1.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNLux-v0.1.0

[#539]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/539
[#542]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/542
[#543]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/543
[#549]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/549
[#576]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/576
[#596]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/596
[#684]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/684
[#685]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/685
[#686]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/686
[#687]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/687
