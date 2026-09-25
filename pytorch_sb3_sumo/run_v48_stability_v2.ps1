param([switch]$CheckOnly, [switch]$ReportOnly)
$ErrorActionPreference = 'Stop'
$stabilityPython = 'D:\Programs\Anaconda\envs\llm_pipeline\python.exe'
if (-not (Test-Path -LiteralPath $stabilityPython)) { throw 'Configured Python environment is missing.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    $stabilityCommand = if ($CheckOnly) { 'check' } elseif ($ReportOnly) { 'report' } else { 'run' }
    & $stabilityPython -m tools.v48_stability_v2.cli $stabilityCommand
    if ($LASTEXITCODE -ne 0) { throw "Stability pipeline failed (exit $LASTEXITCODE). Check r48s2/logs and preserved status files." }
} finally {
    Pop-Location
}
