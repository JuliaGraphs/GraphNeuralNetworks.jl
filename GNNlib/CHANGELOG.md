# Changelog

All notable changes to the GNNlib.jl package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2026-07-22

### Added
- Added Mooncake rules for the `propagate` `copy_xj`, `e_mul_xj`, and `w_mul_xj` fast paths ([#677], [#678], [#679]).

### Changed
- Relaxed the `CUDA` compat bound to `"5, 6"`, adding support for CUDA.jl v6 ([#690]).
- Bumped the NNlib compat bound ([#687]).

## [1.2.1] - 2026-04-29

### Changed
- Bumped the DataStructures compat bound ([#670]).

### Fixed
- Fixed the `gmm_conv` (`GMMConv`) implementation ([#645]).

## [1.2.0] - 2026-01-11

### Added
- Added support for reducing nodes over heterogeneous graphs ([#634]).

### Fixed
- Fixed empty-edge handling in `softmax_edge_neighbors` ([#636]).

## [1.1.0] - 2025-12-24

### Added
- Added SpMM-based message passing with CUDA support for coalesced COO graphs ([#617]).
- Added CUDA sparse support for the `propagate` `copy_xj` and `w_mul_xj` fast paths via matrix multiplication ([#605], [#610]).
- Added a `fmt` keyword to `adjacency_matrix` and a `copy_xj` `propagate` fast path for Metal ([#619]).

### Changed
- Refactored `propagate` signatures to accept COO subtypes for the `copy_xj` and `w_mul_xj` fast paths ([#611]).

### Fixed
- Fixed the NNlib CUDA extension ([#621]).

## [1.0.1] - 2025-01-12

Maintenance release: internal fixes and test-infrastructure updates, no user-facing API changes.

## [1.0.0] - 2024-12-21

First stable release of GNNlib.jl.

### Changed
- Rewrote the recurrent temporal layers for Flux v0.16 ([#560]).
- Updated for compatibility with Flux v0.15 ([#550]).

[1.2.2]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.2.2
[1.2.1]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.2.1
[1.2.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.2.0
[1.1.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.1.0
[1.0.1]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.0.1
[1.0.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GNNlib-v1.0.0

[#550]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/550
[#560]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/560
[#605]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/605
[#610]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/610
[#611]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/611
[#617]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/617
[#619]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/619
[#621]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/621
[#634]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/634
[#636]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/636
[#645]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/645
[#670]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/670
[#677]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/677
[#678]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/678
[#679]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/679
[#687]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/687
[#690]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/690
