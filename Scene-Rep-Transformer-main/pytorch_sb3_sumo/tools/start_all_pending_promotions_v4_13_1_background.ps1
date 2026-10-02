param(
    [string]$Device = "cuda",
    [switch]$RerunCompleted
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = "D:\Programs\Anaconda\envs\llm_pipeline\python.exe"
$runnerArgument = "tools\run_all_pending_promotions_v4_12.py"
$reportArgument = "results_promotion_automation\all_latest_family_promotions_v4_13_1.json"
$statePath = Join-Path $root "results_promotion_automation\all_pending_promotions_v4_13_1_background_state.json"
$logRoot = Join-Path $root "results_promotion_automation\detached_logs_v4_13_1"

if (Test-Path -LiteralPath $statePath) {
    $previous = Get-Content -Raw -Encoding UTF8 -LiteralPath $statePath | ConvertFrom-Json
    $previousProcess = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($null -ne $previousProcess) {
        $previous | ConvertTo-Json -Depth 8
        exit 0
    }
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutPath = Join-Path $logRoot "promotions_${timestamp}.stdout.log"
$stderrPath = Join-Path $logRoot "promotions_${timestamp}.stderr.log"
$arguments = @(
    $runnerArgument,
    "--device", $Device,
    "--report", $reportArgument
)
if ($RerunCompleted) {
    $arguments += "--rerun-completed"
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$state = [ordered]@{
    schema_version = "topo-scene.latest-family-background-launcher/v4.13.1"
    started_at_local = (Get-Date).ToString("o")
    pid = $process.Id
    command = @($python) + $arguments
    working_directory = $root
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    aggregate_report = (Join-Path $root $reportArgument)
    supersedes_launcher = "tools/start_all_pending_promotions_v4_13_background.ps1"
    recovery_reason = "quote-safe relative child arguments under a working directory containing spaces"
    hidden_window = $true
    background_process = $true
    terminal_session_independent = $true
    discovery_supports_current_and_future_versioned_pipelines = $true
    failure_does_not_cancel_remaining_versions_or_jobs = $true
    formal_stage_launched = $false
}
$temporary = "${statePath}.tmp"
$state | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -LiteralPath $temporary
Move-Item -Force -LiteralPath $temporary -Destination $statePath
$state | ConvertTo-Json -Depth 8
