[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int]$WaitForPid,

    [string]$PythonExe = 'D:\Programs\Anaconda\envs\llm_pipeline\python.exe',

    [string]$Device = 'cuda',

    [string]$ReportPath = 'results_promotion_automation\v4_11_2_pipeline_summary.json'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Runner = Join-Path $ProjectRoot 'tools\run_all_v4_11_2_pipeline.py'

try {
    Wait-Process -Id $WaitForPid -ErrorAction Stop
}
catch [Microsoft.PowerShell.Commands.ProcessCommandException] {
    # The predecessor already exited between scheduling and entering the wait.
}

Set-Location -LiteralPath $ProjectRoot
& $PythonExe $Runner --device $Device --report $ReportPath
exit $LASTEXITCODE
