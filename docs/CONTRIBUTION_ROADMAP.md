# 30 / 60 / 90-Day Contribution Roadmap

This plan assumes about 6–8 focused hours each week alongside study/work. Adjust the pace to your schedule. The goal is to learn by making small, reviewed changes; GSoC 2027 opportunities and mentor availability are not yet known.

## Days 1–30: Build a working mental model

### Week 1 — Julia and local workflow

- Read `AGENTS.md`, `README.md`, and `DEVELOPMENT.md`.
- Use `scripts/julia.ps1` to inspect `GNNGraph`, `DataStore`, and the Julia package environment.
- Learn multiple dispatch, parametric structs, keyword arguments, broadcasting, named tuples, and Julia package extensions from the Julia manual.
- Run one focused `@testitem`; then run the `GNNGraphs` CPU suite when data/network requirements are available.

**Deliverable:** personal notes explaining `features × nodes`, COO source/target arrays, and how a test item is launched.

### Week 2 — Graph representation and operations

- Trace `GNNGraph` construction from `GNNGraphs/src/gnngraph.jl` into `to_coo`, `to_sparse`, `to_dense`, and `normalize_graphdata`.
- Read `query.jl` for `edge_index`, `degree`, and adjacency semantics.
- Read `transform.jl` batch/unbatch, self-loop, remove-edge, and `getgraph` functions.
- Run the matching graph tests and inspect any `broken` annotations before interpreting results.

**Deliverable:** a one-page graph API map with one homogeneous and one heterogeneous example.

### Week 3 — Message passing and one layer

- Read `GNNlib/src/msgpass.jl`; step through `apply_edges` → message function → `aggregate_neighbors`.
- Compare a generic message closure with `copy_xj` and identify when GNNlib can use a fast path.
- Follow GCN from Flux and Lux frontend calls to `GNNlib.gcn_conv`, then to GNNGraphs `degree`.
- Compare the source code against layer docstrings and hand-calculate a tiny graph result.

**Deliverable:** a trace diagram for GCN and a small manual reference calculation.

### Week 4 — First small contribution

- Read an open issue page and check for current branches/PRs before choosing work.
- Recommended starting scope: a missing expected-value test, a documentation correction, or a reproducible error-message improvement.
- Ask maintainers on the issue if the task is still wanted; do not duplicate an existing PR.
- Write the regression test first, make the smallest change, run the package suite, and open a draft PR if appropriate.

**Deliverable:** one small reviewable PR or a maintainer-confirmed test plan.

## Days 31–60: Own one subsystem

### Weeks 5–6 — Learn the test and AD model

- Study `test_module.jl` for finite differences, Zygote, Mooncake and device selection.
- Pick one layer and inspect primal output tests plus gradient tests. Add a hand-computed reference to your local learning branch if a formula is otherwise only shape/gradient checked.
- Learn how `@testitem` setup modules and `GNN_TEST_*` flags select CPU/GPU execution.
- Read relevant Mooncake/Enzyme extensions and the changelog entries for prior AD fixes.

**Deliverable:** a backend/layer/graph-format test matrix for one operation.

### Weeks 7–8 — Reproduce and scope a real issue

- Check current issue #710 or #640 status and linked PRs; verify the versions/OS/devices in the issue before running it.
- Build a minimal script outside production source. Capture Julia/package versions and stack trace.
- Split graph data, message passing, frontend and backend responsibilities. Use `@which`, `methods`, and `@code_warntype` as appropriate; do not treat type instability output as proof of a bug by itself.
- Present reproduction and test plan to maintainers before proposing a backend-wide change.

**Deliverable:** a minimal reproducible example, root-cause evidence, and scoped implementation plan.

## Days 61–90: Deliver a medium contribution and prepare a proposal

### Weeks 9–10 — Implement with regression coverage

- Implement in the lowest package that owns the behavior: graph semantics in GNNGraphs, shared math in GNNlib, training/state wrapper in Flux/Lux frontend, backend-specialized behavior in an extension.
- Keep CPU behavior and optional backend behavior tested separately.
- Add exact outputs/gradients or reference results, including isolated node, empty-edge, directed graph and multiple representation cases when relevant.
- Benchmark only if performance is the stated goal; measure warm and steady-state behavior, allocations and device synchronization.

**Deliverable:** focused PR with reproduction, tests, review notes and accurate limitations.

### Weeks 11–12 — GSoC exploration

- Identify one problem theme and ask a maintainer whether it fits the project's likely priorities; do not assume a 2027 slot exists.
- Write a short design note: user impact, package ownership, API surface, baseline, test plan, milestones, risks, fallback and mentor touchpoints.
- Read current GSoC/Julia organization dates and application guidance once published.
- Contribute a second small PR or review an existing PR to demonstrate collaboration and follow-through.

**Deliverable:** proposal draft grounded in a merged/under-review contribution and current maintainer feedback.

## 90-day success criteria

- Can trace a graph operation from public API through the owning package and tests.
- Can explain the graph source/target orientation and feature axis convention.
- Has run a package CPU suite and at least one focused test item, recording any environment failures honestly.
- Has one or more small PRs or a clear maintainer-approved implementation plan.
- Can distinguish a confirmed bug from a report, a TODO, a backend gap, and a performance hypothesis.
- Has a GSoC project outline with scoped milestones, but no assumption of acceptance or project availability.

## Weekly routine

1. Sync your fork with upstream `master` and create a focused branch.
2. Read the owning package's source, matching test, docs and recent changelog/history.
3. Reproduce before changing code.
4. Make one minimal change and run the smallest relevant test first.
5. Run the full package CPU suite and any available optional backend suite.
6. Review the diff for accidental generated files, summarize evidence, then ask maintainers for review.

