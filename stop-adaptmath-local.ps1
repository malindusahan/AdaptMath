[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

if (-not (Test-Path -LiteralPath $script:AdaptMathStateFile)) {
    Write-Host 'No launcher-owned AdaptMath processes are recorded. Nothing was stopped.'
    exit 0
}

try {
    $state = Get-Content -Raw -LiteralPath $script:AdaptMathStateFile | ConvertFrom-Json
}
catch {
    Write-Error "Cannot safely parse $script:AdaptMathStateFile. No processes were stopped."
    exit 1
}

if ($state.frontend.startedByLauncher) {
    Stop-AdaptMathProcessTree -Record $state.frontend.process
    Write-Host 'Stopped launcher-owned Adaptive Tutor frontend process tree.'
}
if ($state.tutor.startedByLauncher) {
    Stop-AdaptMathProcessTree -Record $state.tutor.process
    Write-Host 'Stopped launcher-owned Tutor backend process tree.'
}
if ($state.memoryLog.startedByLauncher) {
    Stop-AdaptMathProcessTree -Record $state.memoryLog.process
    Write-Host 'Stopped launcher-owned Memory log follower.'
}

Push-Location $script:AdaptMathMemoryRoot
try {
    if ($state.memory.startedByLauncher) {
        & docker compose stop memory_service
        Write-Host 'Stopped launcher-owned Memory Compose service.'
    }
    if ($state.postgres.startedByLauncher) {
        & docker compose stop postgres
        Write-Host 'Stopped launcher-owned PostgreSQL Compose service.'
    }
}
finally {
    Pop-Location
}

$resolvedState = [System.IO.Path]::GetFullPath($script:AdaptMathStateFile)
$resolvedStateRoot = [System.IO.Path]::GetFullPath($script:AdaptMathStateDirectory).TrimEnd('\') + '\'
if (-not $resolvedState.StartsWith($resolvedStateRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error 'Refusing to remove a launcher state file outside the expected runtime-state directory.'
    exit 1
}
Remove-Item -LiteralPath $resolvedState -Force
Write-Host 'AdaptMath local launcher resources stopped. Existing Docker services and all persisted learner data were preserved.'

