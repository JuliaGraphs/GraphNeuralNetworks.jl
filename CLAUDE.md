# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a **monorepo of four registered Julia packages** wired together with a top-level Julia `[workspace]` (`Project.toml`). Dependencies flow one way:

```
GNNGraphs  ──►  GNNlib  ──►  GraphNeuralNetworks   (Flux frontend)
    │             │
    └─────────────┴──────►  GNNLux                 (Lux frontend)
```

- **GNNGraphs** — graph data structures (`GNNGraph`, `GNNHeteroGraph`, `TemporalSnapshotsGNNGraph`, `DataStore`) and graph ops (query/transform/generate/sampling/convert). No deep-learning framework dependency.
- **GNNlib** — framework-agnostic message-passing engine (`propagate`, `apply_edges`, `aggregate_neighbors`, gather/scatter) and the **functional** (stateless) implementation of every layer's forward pass (e.g. `gcn_conv`, `gat_conv`). Not meant for direct end-user use; re-exported by the frontends.
- **GraphNeuralNetworks** — Flux frontend. Stateful mutable-struct layers (`GCNConv`, `GATConv`, ...) whose forward pass calls into GNNlib. This is the package that shares the repo name.
- **GNNLux** — Lux frontend. Stateless `@concrete`/`AbstractLuxLayer` versions of the same layers, also calling into GNNlib.

The frontends `@reexport using GNNGraphs` and `GNNlib`, so users only install one frontend.

### Parallel file structure
Each package mirrors the same `src/layers/` split: `basic.jl`, `conv.jl`, `pool.jl`, `temporalconv.jl` (plus `heteroconv.jl` in the Flux frontend). A given layer therefore appears in up to three places — its math in `GNNlib/src/layers/conv.jl`, its Flux struct in `GraphNeuralNetworks/src/layers/conv.jl`, its Lux struct in `GNNLux/src/layers/conv.jl`.

## Common commands

Julia ≥ 1.10 is required (some tests, e.g. Mooncake, need ≥ 1.12). Thanks to the workspace, activating any package makes the sibling packages available in dev mode automatically — no manual `Pkg.dev` needed for local development.

```bash
# Run one package's full CPU test suite
julia --project=GraphNeuralNetworks -e 'using Pkg; Pkg.test("GraphNeuralNetworks")'
julia --project=GNNGraphs         -e 'using Pkg; Pkg.test("GNNGraphs")'

# Build docs for one package
julia --project=GNNGraphs/docs GNNGraphs/docs/make.jl

# Format code (SciML style, configured in .JuliaFormatter.toml)
julia -e 'using JuliaFormatter; format(".")'
```

### Tests use TestItemRunner, not plain `@testset`
Test files define independent `@testitem "..." setup=[TestModule] begin ... end` blocks. `runtests.jl` in each package just calls `@run_package_tests` with a tag filter. Shared fixtures and exports live in each package's `test/test_module.jl` (`TestModule`).

- **Run a single test item**: use the VS Code Julia extension's test-item UI, or filter in `runtests.jl`. Test items are self-contained, so you can also copy one into a REPL after `include("test/test_module.jl")`.
- **GPU tests** are tagged `:gpu`; untagged items are CPU tests. Backend selection is via env vars read in `runtests.jl` and `test_module.jl`: `GNN_TEST_CPU` (default `"true"`), `GNN_TEST_CUDA`, `GNN_TEST_AMDGPU`, `GNN_TEST_Metal`. CPU CI runs untagged items; GPU CI (Buildkite, `.buildkite/pipeline.yml`) sets `GNN_TEST_CPU=false` and a backend var to `true`.

## Adding a new convolutional layer

Per `GraphNeuralNetworks/docs/src/dev.md`, a layer must be added in all frontends:

1. Functional forward pass in **GNNlib** (`src/layers/conv.jl`, exported lowercase e.g. `my_conv`).
2. Stateful struct in **GraphNeuralNetworks** (Flux), calling the GNNlib function.
3. Stateless struct in **GNNLux** (Lux).
4. Add it to the table in the docs `api/conv.md`.

Recommended workflow: build a self-contained Flux layer first with tests, then move the forward-pass body into GNNlib, then add the Lux version. The `@structdef` macro from AutoStructs.jl can generate the struct + constructor together.

Layers typically implement their forward pass via `propagate(fmsg, g, aggr; xi, xj, e)` — see the extensive docstring at the top of `GNNlib/src/msgpass.jl`. `xi`/`xj` are target/source node features materialized onto edges; `e` is edge features.

## Conventions

- **GPU support** is provided through package extensions (`ext/`, e.g. `GNNGraphsCUDAExt`, `GNNlibCUDAExt`/`AMDGPUExt`/`MetalExt`) gated on weakdeps — don't add a hard GPU dependency.
- **Mooncake** and **SimpleWeightedGraphs** support are also extensions; keep them optional.
- **Versioning**: each PR should bump the `version` in the `Project.toml` of every affected package per semver, using a `-DEV` suffix for unreleased changes (e.g. `1.18.0-DEV`), and update inter-package `[compat]` bounds when a frontend needs newer GNNGraphs/GNNlib.
- Follow the `[workspace]` dev model; do not rewire packages to local paths outside the repo.
