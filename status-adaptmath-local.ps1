[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$MemoryPort = 8000,
    [ValidateRange(1, 65535)][int]$TutorPort = 8002,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

if (Test-Path -LiteralPath $script:AdaptMathStateFile) {
    try {
        $state = Get-Content -Raw -LiteralPath $script:AdaptMathStateFile | ConvertFrom-Json
        $MemoryPort = [int]$state.ports.memory
        $TutorPort = [int]$state.ports.tutor
        $FrontendPort = [int]$state.ports.frontend
    }
    catch {
        Write-Warning 'Launcher state could not be parsed; using default/requested ports.'
    }
}

$postgresUp = Test-AdaptMathComposeServiceRunning -Service 'postgres'
$memoryContainerUp = Test-AdaptMathComposeServiceRunning -Service 'memory_service'
$memoryHealth = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$MemoryPort/health"
$memoryReady = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$MemoryPort/ready"
$tutorHealth = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$TutorPort/health"
$tutorReady = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$TutorPort/ready"
$frontend = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$FrontendPort"
$frontendUp = $frontend.Success -and $frontend.Content -match '<title>AdaptMath</title>'

$databaseOk = $false
if ($postgresUp) {
    Push-Location $script:AdaptMathMemoryRoot
    try { & docker compose exec -T postgres pg_isready -U memory_app -d adaptmath_memory_integration *> $null }
    finally { Pop-Location }
    $databaseOk = $LASTEXITCODE -eq 0
}

$llmCredential = if (Test-AdaptMathCredentialPresent) { 'PRESENT' } else { 'MISSING' }

Write-Host "PostgreSQL: $(if ($postgresUp) { 'UP' } else { 'DOWN' })"
Write-Host ''
Write-Host "Memory: $(if ($memoryContainerUp) { 'UP' } else { 'DOWN' })"
Write-Host "health: $(if ($memoryHealth.Success) { '200' } else { 'FAIL' })"
Write-Host "ready:  $(if ($memoryReady.Success) { 'READY' } else { 'NOT READY' })"
Write-Host ''
Write-Host "Adaptive Tutor backend: $(if ($tutorHealth.Success) { 'UP' } else { 'DOWN' })"
Write-Host "health: $(if ($tutorHealth.Success) { '200' } else { 'FAIL' })"
Write-Host "ready:  $(if ($tutorReady.Success) { 'READY' } else { 'NOT READY' })"
Write-Host ''
Write-Host "Adaptive Tutor frontend: $(if ($frontendUp) { 'UP' } else { 'DOWN' })"
Write-Host ''
Write-Host "Memory DB connection: $(if ($databaseOk) { 'OK' } else { 'FAIL' })"
Write-Host "LLM credential: $llmCredential"
Write-Host ''
Write-Host "Frontend URL:       http://127.0.0.1:$FrontendPort"
Write-Host "Tutor backend URL:  http://127.0.0.1:$TutorPort"
Write-Host "Memory backend URL: http://127.0.0.1:$MemoryPort"

$allReady =
    $postgresUp -and $memoryContainerUp -and $databaseOk -and
    $memoryHealth.Success -and $memoryReady.Success -and
    $tutorHealth.Success -and $tutorReady.Success -and $frontendUp
if (-not $allReady) { exit 1 }

