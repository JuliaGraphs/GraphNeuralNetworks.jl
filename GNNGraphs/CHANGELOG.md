# Changelog

All notable changes to the GNNGraphs.jl package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.1] - 2026-07-22

### Changed
- Bumped the NNlib and KrylovKit compat bounds ([#687], [#680]).

### Fixed
- Fixed `remove_self_loops` mutating the input for adjacency-matrix graphs ([#659]).
- Fixed `unbatch` for COO batches containing zero-edge graphs ([#652]).
- Fixed `sample_nbrs` when sampling without replacement ([#648]).

## [1.5.0] - 2025-12-24

### Added
- Added `coalesce` and `is_coalesced` to sort and merge duplicate edges of COO graphs, and to query coalesced state ([#613], [#624], [#625]).
- Added SpMM-based message passing with CUDA support for coalesced COO graphs ([#617]).
- Added a `fmt` keyword to `adjacency_matrix` to select the output format, plus a `copy_xj` `propagate` fast path for Metal ([#619]).
- Added CUDA support for the `binarize()` operation on sparse matrices ([#601]).

### Changed
- The monorepo now uses Julia workspaces for development ([#630]).

### Fixed
- Fixed the NNlib CUDA extension ([#621]).

## [1.4.2] - 2025-02-07

### Changed
- Improved type inference for `num_nodes` ([#588]).
- Bumped the KrylovKit compat bound ([#587]).

### Fixed
- Fixed a corner case in `normalize_graphdata` ([#589]).
- Fixed a Zygote differentiation error ([#579]).

## [1.4.1] - 2024-12-25

### Changed
- Removed the constraint requiring an equal number of features across node/edge types in the `gdata` of heterographs ([#570]).

## [1.4.0] - 2024-12-21

### Added
- Added `broadcast`, `iterate`, and `setindex!` support for `TemporalSnapshotsGNNGraph` ([#563]).

### Changed
- Updated for compatibility with Flux v0.15 ([#550]).

### Fixed
- Fixed `show` for graphs whose features are not arrays ([#564]).

[1.5.1]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNGraphs-v1.5.1
[1.5.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNGraphs-v1.5.0
[1.4.2]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNGraphs-v1.4.2
[1.4.1]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNGraphs-v1.4.1
[1.4.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNGraphs-v1.4.0

[#550]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/550
[#563]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/563
[#564]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/564
[#570]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/570
[#579]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/579
[#587]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/587
[#588]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/588
[#589]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/589
[#601]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/601
[#613]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/613
[#617]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/617
[#619]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/619
[#621]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/621
[#624]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/624
[#625]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/625
[#630]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/630
[#648]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/648
[#652]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/652
[#659]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/659
[#680]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/680
[#687]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/687
