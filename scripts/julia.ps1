$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$julia = Join-Path $repoRoot '.tools\julia-1.13.3\bin\julia.exe'
if (-not (Test-Path -LiteralPath $julia)) {
    throw "Repo-local Julia was not found at '$julia'. Run .\scripts\setup-dev.ps1 first."
}

$env:JULIA_DEPOT_PATH = Join-Path $repoRoot '.julia-depot'
& $julia @args
exit $LASTEXITCODE
