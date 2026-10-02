$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$resultRoot = Join-Path $projectRoot "results_sb3_sumo_paper\paper_baselines_sac_ppo_v14_frozen"
$diagnosticRoot = Join-Path $resultRoot "_diagnostics"

function Read-Json([string]$Path) {
    Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}

function Evaluation-Row(
    [string]$Label,
    [string]$Path,
    [string]$Scenario,
    [string]$Protocol,
    [string]$Validity
) {
    $payload = Read-Json $Path
    [pscustomobject]@{
        case = $Label
        scenario = $Scenario
        protocol = $Protocol
        validity = $Validity
        episodes = [int]$payload.summary.episodes
        success_rate = [double]$payload.summary.success_rate
        collision_rate = [double]$payload.summary.collision_rate
        timeout_rate = [double]$payload.summary.timeout_rate
    }
}

$sacReconciliation = @(
    Evaluation-Row "1. Left turn - wrong generic env (invalid)" (Join-Path $resultRoot "paper__sac__left_turn__seed3\paper_evaluation_detailed.json") "left_turn" "generic_env" "invalid"
    Evaluation-Row "2. Left turn - paper env / frozen" (Join-Path $diagnosticRoot "sac_left_turn_seed3_frozen_80_20.json") "left_turn" "frozen_80_20" "valid"
    Evaluation-Row "3. Left turn - paper env / source-all" (Join-Path $diagnosticRoot "sac_left_turn_seed3_source_all.json") "left_turn" "source_all" "valid"
    [pscustomobject]@{ case = "4. Left turn - paper SAC"; scenario = "left_turn"; protocol = "paper_reference"; validity = "paper"; episodes = 50; success_rate = 0.68; collision_rate = 0.28; timeout_rate = 0.00 }
    Evaluation-Row "5. Roundabout-A - wrong generic env (invalid)" (Join-Path $resultRoot "paper__sac__roundabout_easy__seed0\paper_evaluation_detailed.json") "roundabout_a" "generic_env" "invalid"
    Evaluation-Row "6. Roundabout-A - paper env / frozen" (Join-Path $diagnosticRoot "sac_roundabout_easy_seed0_frozen_80_20.json") "roundabout_a" "frozen_80_20" "valid"
    Evaluation-Row "7. Roundabout-A - paper env / source-all" (Join-Path $diagnosticRoot "sac_roundabout_easy_seed0_source_all.json") "roundabout_a" "source_all" "valid"
    [pscustomobject]@{ case = "8. Roundabout-A - paper SAC"; scenario = "roundabout_a"; protocol = "paper_reference"; validity = "paper"; episodes = 50; success_rate = 0.76; collision_rate = 0.24; timeout_rate = 0.00 }
)

$ppoRows = @(0, 1, 2) | ForEach-Object {
    $seed = $_
    $payload = Read-Json (Join-Path $diagnosticRoot "ppo_left_turn_seed${seed}_source_all.json")
    [pscustomobject]@{
        order = $seed + 1
        policy = "seed$seed"
        episodes = [int]$payload.summary.episodes
        success_rate = [double]$payload.summary.success_rate
        collision_rate = [double]$payload.summary.collision_rate
        comparison = "source_all"
    }
}
$ppoRows += [pscustomobject]@{
    order = 4
    policy = "equal-weight mean of 3 seeds"
    episodes = 150
    success_rate = [double](($ppoRows.success_rate | Measure-Object -Average).Average)
    collision_rate = [double](($ppoRows.collision_rate | Measure-Object -Average).Average)
    comparison = "source_all_mean"
}
$ppoRows += [pscustomobject]@{
    order = 5
    policy = "paper PPO"
    episodes = 50
    success_rate = 0.36
    collision_rate = 0.50
    comparison = "paper_reference"
}

$state = Read-Json (Join-Path $resultRoot "orchestrator_state.json")
$detailedFiles = Get-ChildItem -LiteralPath $resultRoot -Directory |
    Where-Object { $_.Name -like "paper__*" } |
    ForEach-Object { Join-Path $_.FullName "paper_evaluation_detailed.json" } |
    Where-Object { Test-Path -LiteralPath $_ }
$validDetailed = @($detailedFiles | Where-Object {
    $payload = Read-Json $_
    @($payload.episode_records | Where-Object { $null -ne $_.traffic_variant }).Count -gt 0
}).Count
$contaminated = [int]$state.counts.complete - $validDetailed
$matrixQuality = @(
    [pscustomobject]@{ order = 1; status = "traceable paper-scenario evaluation"; jobs = $validDetailed; usable = "yes"; interpretation = "episode records contain released traffic XML identifiers" }
    [pscustomobject]@{ order = 2; status = "contaminated by generic environment"; jobs = $contaminated; usable = "no"; interpretation = "incorrectly counted as complete; traffic_variant is null" }
    [pscustomobject]@{ order = 3; status = "evaluation crash after training"; jobs = [int]$state.counts.partial; usable = "checkpoint recoverable"; interpretation = "PPO dict observation does not match Box space" }
    [pscustomobject]@{ order = 4; status = "not yet run"; jobs = [int]$state.counts.pending; usable = "no"; interpretation = "cannot enter paper-comparison aggregate" }
)

[pscustomobject]@{
    generated_at = (Get-Date).ToString("o")
    sac_reconciliation = $sacReconciliation
    ppo_source_all = $ppoRows
    matrix_quality = $matrixQuality
} | ConvertTo-Json -Depth 8
