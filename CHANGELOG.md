# Changelog

All notable changes to the packages in this monorepo are documented in this file.

The repository hosts four independently versioned packages — **GNNGraphs.jl**,
**GNNlib.jl**, **GraphNeuralNetworks.jl** (Flux frontend), and **GNNLux.jl**
(Lux frontend) — each with its own section below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the packages adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries link to the pull request that introduced them.

---

## GNNGraphs.jl

### 1.5.1 — 2026-07-22

**Changed**
- Relaxed the `CUDA` compat bound to `"5, 6"`, adding support for CUDA.jl v6 ([#690]).
- Bumped the NNlib and KrylovKit compat bounds ([#687], [#680]).

**Fixed**
- Fixed `remove_self_loops` mutating the input for adjacency-matrix graphs ([#659]).
- Fixed `unbatch` for COO batches containing zero-edge graphs ([#652]).
- Fixed `sample_nbrs` when sampling without replacement ([#648]).

### 1.5.0 — 2025-12-24

**Added**
- Added `coalesce` and `is_coalesced` to sort and merge duplicate edges of COO graphs, and to query coalesced state ([#613], [#624], [#625]).
- Added SpMM-based message passing with CUDA support for coalesced COO graphs ([#617]).
- Added a `fmt` keyword to `adjacency_matrix` to select the output format, plus a `copy_xj` `propagate` fast path for Metal ([#619]).
- Added CUDA support for the `binarize()` operation on sparse matrices ([#601]).

**Changed**
- The monorepo now uses Julia workspaces for development ([#630]).

**Fixed**
- Fixed the NNlib CUDA extension ([#621]).

### 1.4.2 — 2025-02-07

**Changed**
- Improved type inference for `num_nodes` ([#588]).
- Bumped the KrylovKit compat bound ([#587]).

**Fixed**
- Fixed a corner case in `normalize_graphdata` ([#589]).
- Fixed a Zygote differentiation error ([#579]).

### 1.4.1 — 2024-12-25

**Changed**
- Removed the constraint requiring an equal number of features across node/edge types in the `gdata` of heterographs ([#570]).

### 1.4.0 — 2024-12-21

**Added**
- Added `broadcast`, `iterate`, and `setindex!` support for `TemporalSnapshotsGNNGraph` ([#563]).

**Changed**
- Updated for compatibility with Flux v0.15 ([#550]).

**Fixed**
- Fixed `show` for graphs whose features are not arrays ([#564]).

---

## GNNlib.jl

### 1.3.0 — 2026-07-22

**Added**
- Added Mooncake rules for the `propagate` `copy_xj`, `e_mul_xj`, and `w_mul_xj` fast paths ([#677], [#678], [#679]).

**Changed**
- Relaxed the `CUDA` compat bound to `"5, 6"`, adding support for CUDA.jl v6 ([#690]).
- Bumped the NNlib compat bound ([#687]).

### 1.2.1 — 2026-04-29

**Changed**
- Bumped the DataStructures compat bound ([#670]).

**Fixed**
- Fixed the `gmm_conv` (`GMMConv`) implementation ([#645]).

### 1.2.0 — 2026-01-11

**Added**
- Added support for reducing nodes over heterogeneous graphs ([#634]).

**Fixed**
- Fixed empty-edge handling in `softmax_edge_neighbors` ([#636]).

### 1.1.0 — 2025-12-24

**Added**
- Added SpMM-based message passing with CUDA support for coalesced COO graphs ([#617]).
- Added CUDA sparse support for the `propagate` `copy_xj` and `w_mul_xj` fast paths via matrix multiplication ([#605], [#610]).
- Added a `fmt` keyword to `adjacency_matrix` and a `copy_xj` `propagate` fast path for Metal ([#619]).

**Changed**
- Refactored `propagate` signatures to accept COO subtypes for the `copy_xj` and `w_mul_xj` fast paths ([#611]).

**Fixed**
- Fixed the NNlib CUDA extension ([#621]).

### 1.0.1 — 2025-01-12

Maintenance release: internal fixes and test-infrastructure updates, no user-facing API changes.

### 1.0.0 — 2024-12-21

First stable release of GNNlib.jl.

**Changed**
- Rewrote the recurrent temporal layers for Flux v0.16 ([#560]).
- Updated for compatibility with Flux v0.15 ([#550]).

---

## GraphNeuralNetworks.jl (Flux frontend)

### 1.1.0 — 2025-12-24

**Added**
- `TGCN` now supports non-linear activation functions ([#596]).

**Fixed**
- Fixed a Zygote differentiation error ([#579]).

### 1.0.0 — 2024-12-21

First stable release of the Flux frontend.

**Changed**
- Rewrote the recurrent temporal layers for Flux v0.16 ([#560]).
- Updated for compatibility with Flux v0.15 ([#550]).

---

## GNNLux.jl (Lux frontend)

### Unreleased (towards 0.2.0)

**Added**
- Added pooling layers (`GlobalPool`, `GlobalAttentionPool`, `TopKPool`) ([#576]).
- `TGCN` now supports non-linear activation functions ([#596]).

**Changed**
- Bumped the NNlib, OneHotArrays, StableRNGs, and DocumenterInterLinks compat bounds ([#687], [#686], [#684], [#685]).

### 0.1.1 — 2024-12-09

Documentation release: added the "Hands On" tutorial, a version selector, and general docs improvements ([#549], [#543], [#542], [#539]). No API changes.

### 0.1.0 — 2024-12-02

Initial release of the Lux-based frontend for GraphNeuralNetworks.jl, providing
Lux implementations of the graph convolutional, pooling, and temporal layers
(e.g. `GCNConv`, `GraphConv`, `SAGEConv`, `GATConv`, `GMMConv`, `NNConv`,
`ResGatedGraphConv`, and the temporal layers `TGCN`, `GConvGRU`, `GConvLSTM`,
`DCGRU`, `EvolveGCNO`) that share the message-passing implementations in GNNlib.

---

[#539]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/539
[#542]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/542
[#543]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/543
[#549]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/549
[#550]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/550
[#560]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/560
[#563]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/563
[#564]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/564
[#570]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/570
[#576]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/576
[#579]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/579
[#587]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/587
[#588]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/588
[#589]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/589
[#596]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/596
[#601]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/601
[#605]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/605
[#610]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/610
[#611]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/611
[#613]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/613
[#617]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/617
[#619]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/619
[#621]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/621
[#624]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/624
[#625]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/625
[#630]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/630
[#634]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/634
[#636]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/636
[#645]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/645
[#648]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/648
[#652]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/652
[#659]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/659
[#670]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/670
[#677]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/677
[#678]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/678
[#679]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/679
[#680]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/680
[#684]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/684
[#685]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/685
[#686]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/686
[#687]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/687
[#690]: https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/690
