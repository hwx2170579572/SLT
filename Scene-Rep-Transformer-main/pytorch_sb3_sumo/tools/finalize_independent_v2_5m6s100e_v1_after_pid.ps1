param(
    [Parameter(Mandatory = $true)]
    [int]$ControllerPid,
    [string]$PythonExe = "D:\Programs\Anaconda\envs\llm_pipeline\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $projectRoot "tools\run_independent_v2_5m6s100e_v1.py"
$protocol = Join-Path $projectRoot "experiments\independent_v2_five_methods_six_scenarios_100ep_v1\protocol.json"
$resultRoot = Join-Path $projectRoot "results_iv2_5m6s100e_v1"

$controller = Get-Process -Id $ControllerPid -ErrorAction SilentlyContinue
if ($null -ne $controller) {
    Wait-Process -Id $ControllerPid
}

Set-Location -LiteralPath $projectRoot
& $PythonExe $runner status --protocol $protocol --result-root $resultRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $PythonExe $runner summarize --protocol $protocol --result-root $resultRoot --require-complete
exit $LASTEXITCODE
