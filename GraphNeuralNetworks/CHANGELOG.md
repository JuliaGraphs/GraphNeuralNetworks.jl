# Changelog

All notable changes to the GraphNeuralNetworks.jl package (the Flux frontend) will
be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-12-24

### Added
- `TGCN` now supports non-linear activation functions ([#596]).

### Fixed
- Fixed a Zygote differentiation error ([#579]).

## [1.0.0] - 2024-12-21

First stable release of the Flux frontend.

### Changed
- Rewrote the recurrent temporal layers for Flux v0.16 ([#560]).
- Updated for compatibility with Flux v0.15 ([#550]).

[1.1.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GraphNeuralNetworks-v1.1.0
[1.0.0]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/releases/tag/GraphNeuralNetworks-v1.0.0

[#550]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/550
[#560]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/560
[#579]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/579
[#596]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/596
