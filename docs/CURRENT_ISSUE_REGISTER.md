# Current upstream issue register

**Checked:** 2026-11-04 (Asia/Calcutta)  
**Repository:** upstream `JuliaGraphs/GraphNeuralNetworks.jl`; local working copy is the fork `DHAKED440/GraphNeuralNetworks.jl`.  
**Local checkout:** `master` at `aa2926c`; the existing audit notes below are from 2026-11-03. The local checkout was not fetched during this pass.

This register reflects a current spot-check of public GitHub pages plus the repository audit in [`CONTRIBUTOR_RESEARCH_REPORT.md`](CONTRIBUTOR_RESEARCH_REPORT.md). It is not a complete record of all historical issues, PRs, comments, or commits: `gh` is unavailable, GitHub's list pages are dynamically loaded, and the API endpoints could not be read in this environment. As of the page snapshots, GitHub showed 54 open issues and 23 open PRs (444 closed PRs shown on the PR page). Treat counts and statuses as a snapshot, not a full export. Issue #710 was opened directly and its reproduction/details were readable. No source changes are included in this register.

## Findings

| ID | Category | Status | Location | Difficulty | Evidence and next step |
|---|---|---|---|---|---|
| BUG-001 | Backend | Reported, reproduction provided; not locally reproduced | `GNNGraphs/src/query.jl` (`degree`); called by `GNNlib/src/layers/conv.jl` | Advanced | [Upstream #710](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/710) reports `GCNConv` failing in `Reactant.@compile`: scatter receives `Vector{Vector{TracedRNumber}}` while available methods expect traced arrays. The issue is open, has `gnnlib` and `lux` labels, and no linked branch/PR on the issue page. A local code-path analysis is in the research report. Reproduce on the stated package/device versions before changing graph degree or adding a Reactant-specific extension. |
| BUG-002 | Backend / AD | Reported; current behavior not verified | `GNNGraphs/src/transform.jl` (`add_edges`) | Advanced | [Upstream #671](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/671) reports Zygote segfault when differentiating across `add_edges`. The current issue index shows it open. No new independent reproduction was run. Minimize the reporter's case and check current upstream implementation/linked work before proposing a fix. |
| BUG-003 | Backend / AD | Reported; current behavior not verified | Heterogeneous GAT path; `GraphNeuralNetworks/src/layers/heteroconv.jl`, `GNNlib/src/layers/conv.jl` | Advanced | [Upstream #637](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/637) reports empty edge arrays and backpropagation failures for `GATConv` on heterographs. A related open PR, [#638](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/638), is titled “Gate empty result in GATConv”. Inspect the PR diff/checks and run the minimal empty-relation case before selecting a contribution. |
| BUG-004 | Performance | Reported; not reproduced | `GNNlib/src/layers/conv.jl` (`SGConv`) | Intermediate | [Upstream #600](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/600) reports progressively increasing runtime in a training loop. The report is not proof of a leak. Profile repeated forward/backward iterations, allocations, and compilation separately before assigning cause. |
| BUG-005 | AD / compatibility | Reported; current compatibility not verified | `GNNGraphs/ext/`, `GNNGraphs/test/ext/` | Advanced | [Upstream #702](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/702) is “Mooncake on CUDA issue tracker”. Inspect the individual reports and current extension support; do not treat it as one confirmed defect without a reproducible case. |
| TEST-001 | Testing | Missing coverage | `GNNLux/test/` | Beginner | Existing audit found the Lux test runner states GPU coverage has not been added. Add backend-tagged coverage for supported operations once a reliable CI/device strategy is chosen; this is a coverage opportunity, not evidence of a runtime failure. |
| PERF-001 | Performance | Improvement opportunity | `GNNGraphs/src/query.jl` (`Graphs.degree(g, i)`) | Intermediate | Existing audit found this overload computes the full degree vector before selecting node `i` and has a TODO to make it efficient. Benchmark repeated single-node queries and preserve directedness/representation semantics before optimizing. |
| DOC-001 | Documentation | New opportunity | `GNNLux/docs/`, `GraphNeuralNetworks/docs/` | Beginner | Upstream has open documentation proposals including [#669](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/669) (link prediction tutorial) and [#603](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/issues/603) (reinstate TemporalBrains tutorial). Check current pages and issue discussion before taking either. |
| API-001 | Feature parity | Improvement opportunity | `GNNLux/src/GNNLux.jl`, `GNNLux/src/layers/` | Advanced | Existing audit noted Flux exports `TAGConv` and `TransformerConv` while GNNLux does not. Upstream has an open WIP TransformerConv PR [#501](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/501). Review its present state and design intent before duplicating or extending it. |

## Active PRs worth triaging

The public PR listing snapshot showed 23 open PRs. These are directly relevant to the findings/opportunities above; the list is not exhaustive.

| PR | Topic | Register relation | Recommended check |
|---|---|---|---|
| [#703](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/703) | Fix Enzyme crash on dense/sparse adjacency graphs | AD/backend | Inspect changed methods, review status and checks; compare with issue #609 and current Enzyme extension. |
| [#697](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/697) | Learnable epsilon in GIN | Feature | Check whether it is active and whether API/tests/docs are complete before proposing overlapping work. |
| [#638](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/638) | Gate empty result in GATConv | BUG-003 | Verify it resolves the heterograph empty-edge and gradient case, not just the reported forward path. |
| [#501](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/501) | WIP GNNLux TransformerConv | API-001 | Inspect latest activity and implementation/tests; avoid parallel implementation until its disposition is clear. |
| [#657](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/657), [#654](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/654), [#653](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/653), [#649](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl/pull/649) | Heterogeneous convolution support | Feature coverage | Check current review state and API consistency across Flux/Lux/GNNlib before choosing a narrowly scoped contribution. |

## Audit stages and limits

1. **Repository audit:** The prior research report maps the four-package workspace, source, tests, docs, examples, extensions, CI and benchmarks. It records CPU test results for all four package suites. No GPU suite or documentation build was run in that audit.
2. **GitHub history:** Current issue/PR counts and selected current pages were inspected. A complete upstream export and full comments/reviews/commit-by-commit review were not available because GitHub CLI is absent and the API could not be accessed through the browsing tool.
3. **Cross-reference:** The report traces #710 into the current local source path. Other issue/PR links above remain leads requiring their full discussion and diff to be checked against a fresh upstream checkout.
4. **Independent bug hunt:** The prior report found no separate bug established by static review. The listed performance and coverage items are not classified as bugs.
5. **Verification:** #710 has a reporter-provided minimal reproduction but was not reproduced locally; Reactant is not in the test environment. No bug in this register is newly marked “confirmed unfixed” based solely on its issue being open.

## Suggested next contribution

Start with TEST-001 or a narrowly scoped documentation improvement. For a bug fix, #710 has the clearest technical trace, but first reproduce it with the stated Reactant setup and establish whether the traced scalar materialization is caused by GNNGraphs or Reactant's scatter API. For all work, check the live issue/PR state and latest upstream source immediately before opening an issue or branch.
