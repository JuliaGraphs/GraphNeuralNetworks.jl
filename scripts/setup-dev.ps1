$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $repoRoot '.tools'
$juliaDir = Join-Path $toolsDir 'julia-1.13.3'
$julia = Join-Path $juliaDir 'bin\julia.exe'
$archive = Join-Path $toolsDir 'julia-1.13.3-win64.zip'
$expectedSha256 = '1fae8fc95e6199b3d8f6d7a3791810e45e1b3c480eb4b9f5bb211789badfc729'

if (-not (Test-Path -LiteralPath $julia)) {
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host 'Downloading the official Julia 1.13.3 portable archive'
        Invoke-WebRequest -Uri 'https://julialang-s3.julialang.org/bin/winnt/x64/1.13/julia-1.13.3-win64.zip' -OutFile $archive
    }
    $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw "Julia archive SHA-256 mismatch: expected $expectedSha256, got $actualSha256. Remove '$archive' and retry."
    }
    Write-Host 'Extracting Julia into the ignored .tools folder'
    Expand-Archive -LiteralPath $archive -DestinationPath $toolsDir -Force
}
if (-not (Test-Path -LiteralPath $julia)) {
    throw "Julia executable not found at '$julia' after archive extraction."
}

$env:JULIA_DEPOT_PATH = Join-Path $repoRoot '.julia-depot'
$environments = @(
    '.',
    'GNNGraphs', 'GNNGraphs/test',
    'GNNlib', 'GNNlib/test',
    'GraphNeuralNetworks', 'GraphNeuralNetworks/test',
    'GNNLux', 'GNNLux/test',
    'docs'
)

foreach ($environment in $environments) {
    $project = Join-Path $repoRoot $environment
    Write-Host "Instantiating Julia environment: $environment"
    & $julia "--project=$project" -e 'using Pkg; Pkg.instantiate()'
    if ($LASTEXITCODE -ne 0) {
        throw "Pkg.instantiate failed for '$environment' (exit $LASTEXITCODE)."
    }
}

$toolingProject = Join-Path $repoRoot '.tools\devtools'
Write-Host 'Installing JuliaFormatter into the repo-local tooling environment'
& $julia "--project=$toolingProject" -e 'using Pkg; Pkg.add("JuliaFormatter")'
if ($LASTEXITCODE -ne 0) {
    throw "Installing JuliaFormatter failed (exit $LASTEXITCODE)."
}

Write-Host "Development environments are installed in $($env:JULIA_DEPOT_PATH)."
