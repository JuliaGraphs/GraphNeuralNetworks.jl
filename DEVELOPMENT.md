# Local development setup

Run PowerShell from the repository root:

```powershell
.\scripts\setup-dev.ps1
```

The setup installs Julia 1.13.3 as a portable copy under `.tools`, verifies its
SHA-256 checksum, then installs the workspace, package, test, docs, and
JuliaFormatter environments. Julia packages, compiled caches, and artifacts
stay in `.julia-depot`. Neither setup nor the launcher edits the machine PATH or
uses the global Julia depot. Both directories are ignored by Git.

Use the repo-local Julia launcher for subsequent work:

```powershell
.\scripts\julia.ps1 --version
.\scripts\julia.ps1 --project=GNNGraphs -e 'using Pkg; Pkg.test("GNNGraphs")'
.\scripts\julia.ps1 --project=GraphNeuralNetworks -e 'using Pkg; Pkg.test("GraphNeuralNetworks")'
.\scripts\julia.ps1 --project=GNNGraphs/docs GNNGraphs/docs/make.jl
.\scripts\julia.ps1 --project=.tools/devtools -e 'using JuliaFormatter; format(".")'
```

The full package test environments are installed, but setup does not execute
the test suites. GPU backends remain opt-in; install and select the backend
required by the machine before running GPU-tagged tests.
