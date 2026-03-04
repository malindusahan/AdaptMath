[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$MemoryPort = 8400,
    [ValidateRange(1, 65535)][int]$TutorPort = 8402,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173,
    [switch]$FinalSKLWarmstart,
    [switch]$FinalSKLLive,
    [switch]$ManualControlledLive,
    [switch]$MD7R1Rollback,
    [switch]$MD6Rollback
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

if (@($FinalSKLWarmstart, $FinalSKLLive, $ManualControlledLive, $MD7R1Rollback, $MD6Rollback).Where({ $_.IsPresent }).Count -gt 1) {
    throw '-FinalSKLWarmstart, -FinalSKLLive, -ManualControlledLive, -MD7R1Rollback, and -MD6Rollback are mutually exclusive.'
}
$finalSKLExplicit = $FinalSKLWarmstart.IsPresent -or $FinalSKLLive.IsPresent

$desiredSelectorMode = if ($ManualControlledLive.IsPresent) {
    'manual-controlled-live-md7r1-v1'
}
elseif ($MD7R1Rollback.IsPresent) {
    'md7r1-rollback-v1'
}
elseif ($MD6Rollback.IsPresent) {
    'md6-rollback-v1'
}
else {
    'ordinary-md7-r2-tell-c1-v1'
}
$desiredPolicyLineage = if ($MD6Rollback.IsPresent) {
    'preserved-md6'
}
elseif ($MD7R1Rollback.IsPresent -or $ManualControlledLive.IsPresent) {
    'md7r1-fresh-v1'
}
else {
    'direct_disjoint_turn_lints_v1'
}
$desiredTurnLinTSMode = if ($FinalSKLLive.IsPresent) {
    'LIVE'
}
elseif ($FinalSKLWarmstart.IsPresent) {
    'RANDOMIZED_WARMSTART'
}
elseif (Test-Path Env:\ADAPTIVE_TURN_LINTS_MODE) {
    ([string]$env:ADAPTIVE_TURN_LINTS_MODE).Trim().ToUpperInvariant()
}
else { 'SHADOW' }
if ($desiredTurnLinTSMode -notin @('OFF', 'SHADOW', 'RANDOMIZED_WARMSTART', 'LIVE')) {
    throw 'ADAPTIVE_TURN_LINTS_MODE must be OFF, SHADOW, RANDOMIZED_WARMSTART, or LIVE.'
}
$desiredTurnLinTSBlocks = if ($finalSKLExplicit) {
    'S+K+L'
}
elseif (Test-Path Env:\ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS) {
    ([string]$env:ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS).Trim().ToUpperInvariant()
}
else { 'S+K+L' }
$desiredTurnLinTSReward = if ($finalSKLExplicit) {
    'headroom_normalized'
}
elseif (Test-Path Env:\ADAPTIVE_TURN_LINTS_REWARD_MODE) {
    ([string]$env:ADAPTIVE_TURN_LINTS_REWARD_MODE).Trim().ToLowerInvariant()
}
else { 'headroom_normalized' }
$desiredTurnLinTSAnchorMode = if ($finalSKLExplicit) {
    'none'
}
elseif (Test-Path Env:\ADAPTIVE_TURN_LINTS_ANCHOR_MODE) {
    ([string]$env:ADAPTIVE_TURN_LINTS_ANCHOR_MODE).Trim().ToLowerInvariant()
}
else { 'none' }
if ($desiredTurnLinTSAnchorMode -notin @('none', 'md7_logprob_anchor')) {
    throw 'ADAPTIVE_TURN_LINTS_ANCHOR_MODE must be none or md7_logprob_anchor.'
}
$desiredTurnLinTSAnchorGamma = if ($finalSKLExplicit) {
    0.0
}
elseif (Test-Path Env:\ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA) {
    [double]$env:ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA
}
else { 0.0 }
if ([double]::IsNaN($desiredTurnLinTSAnchorGamma) -or [double]::IsInfinity($desiredTurnLinTSAnchorGamma) -or $desiredTurnLinTSAnchorGamma -lt 0.0) {
    throw 'ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA must be finite and nonnegative.'
}
if ($desiredTurnLinTSAnchorMode -eq 'none' -and $desiredTurnLinTSAnchorGamma -ne 0.0) {
    throw 'ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA must be zero when anchor mode is none.'
}
$defaultTurnLinTSRootName = if ($desiredTurnLinTSMode -eq 'RANDOMIZED_WARMSTART') {
    'runtime\turn_lints_randomized_warmstart_skl_final_v1'
}
elseif ($desiredTurnLinTSMode -eq 'LIVE') {
    'runtime\turn_lints_live_skl_final_v1'
}
else { 'runtime\turn_lints_shadow_skl_final_v1' }
$defaultTurnLinTSRoot = Join-Path $script:AdaptMathTutorBackend $defaultTurnLinTSRootName
$desiredTurnLinTSStatePath = if (-not $finalSKLExplicit -and (Test-Path Env:\ADAPTIVE_TURN_LINTS_STATE_PATH)) {
    [System.IO.Path]::GetFullPath([string]$env:ADAPTIVE_TURN_LINTS_STATE_PATH)
}
else { Join-Path $defaultTurnLinTSRoot 'policy_state.json' }
$desiredTurnLinTSEventRoot = if (-not $finalSKLExplicit -and (Test-Path Env:\ADAPTIVE_TURN_LINTS_EVENT_ROOT)) {
    [System.IO.Path]::GetFullPath([string]$env:ADAPTIVE_TURN_LINTS_EVENT_ROOT)
}
else { $defaultTurnLinTSRoot }
$desiredTurnLinTSLiveManifestPath = Join-Path $defaultTurnLinTSRoot 'lineage_manifest.json'
if ($FinalSKLLive.IsPresent) {
    if (-not (Test-Path -LiteralPath $desiredTurnLinTSStatePath -PathType Leaf)) {
        throw "Audited S+K+L LIVE posterior is missing: $desiredTurnLinTSStatePath"
    }
    if (-not (Test-Path -LiteralPath $desiredTurnLinTSLiveManifestPath -PathType Leaf)) {
        throw "Audited S+K+L LIVE lineage manifest is missing: $desiredTurnLinTSLiveManifestPath"
    }
}

New-Item -ItemType Directory -Force -Path $script:AdaptMathLogs | Out-Null
New-Item -ItemType Directory -Force -Path $script:AdaptMathStateDirectory | Out-Null

# Keep the public Memory port selected by this launcher aligned with the
# Docker Compose publication. The service itself still listens on port 8000
# inside its container.
$env:ADAPTMATH_MEMORY_HOST_PORT = [string]$MemoryPort

if (Test-Path -LiteralPath $script:AdaptMathStateFile) {
    $existingState = Get-Content -Raw -LiteralPath $script:AdaptMathStateFile | ConvertFrom-Json
    $trackedIsRunning =
        (Test-AdaptMathTrackedProcess -Record $existingState.tutor.process) -or
        (Test-AdaptMathTrackedProcess -Record $existingState.frontend.process) -or
        (Test-AdaptMathTrackedProcess -Record $existingState.memoryLog.process)
    if ($trackedIsRunning) {
        $trackedMemoryPort = [int]$existingState.ports.memory
        $trackedTutorPort = [int]$existingState.ports.tutor
        $trackedFrontendPort = [int]$existingState.ports.frontend
        $trackedMemoryHealth = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$trackedMemoryPort/health"
        $trackedMemoryReady = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$trackedMemoryPort/ready"
        $trackedTutorHealth = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$trackedTutorPort/health"
        $trackedTutorReady = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$trackedTutorPort/ready"
        $trackedFrontend = Get-AdaptMathHttpResult -Url "http://127.0.0.1:$trackedFrontendPort"
        $trackedFrontendHealthy =
            $trackedFrontend.Success -and
            $trackedFrontend.Content -match '<title>AdaptMath</title>'
        $trackedStackHealthy =
            $trackedMemoryHealth.Success -and $trackedMemoryReady.Success -and
            $trackedTutorHealth.Success -and $trackedTutorReady.Success -and
            $trackedFrontendHealthy
        if ($trackedStackHealthy) {
            $trackedSelectorMode = if (
                $existingState.PSObject.Properties.Name -contains 'selectorMode'
            ) {
                [string]$existingState.selectorMode
            }
            else {
                'ordinary-md7-r2-tell-c1-v1'
            }
            if ($trackedSelectorMode -ne $desiredSelectorMode) {
                throw (
                    "The healthy stack is running selector mode '$trackedSelectorMode', " +
                    "but '$desiredSelectorMode' was requested. Run " +
                    "stop-adaptmath-local.ps1, then start the requested mode."
                )
            }
            $trackedTurnMode = [string]$existingState.turnLinTSMode
            $trackedTurnBlocks = [string]$existingState.turnLinTSContextBlocks
            $trackedTurnReward = [string]$existingState.turnLinTSRewardMode
            if (
                $trackedTurnMode -ne $desiredTurnLinTSMode -or
                $trackedTurnBlocks -ne $desiredTurnLinTSBlocks -or
                $trackedTurnReward -ne $desiredTurnLinTSReward
            ) {
                throw (
                    "The healthy stack Turn-LinTS configuration is " +
                    "$trackedTurnMode/$trackedTurnBlocks/$trackedTurnReward, but " +
                    "$desiredTurnLinTSMode/$desiredTurnLinTSBlocks/$desiredTurnLinTSReward was requested. " +
                    "Run stop-adaptmath-local.ps1, then start the requested configuration."
                )
            }
            Write-Host 'AdaptMath local stack is already tracked and healthy.'
            Write-Host "Selector mode: $trackedSelectorMode"
            Write-Host "Frontend: http://127.0.0.1:$trackedFrontendPort"
            Write-Host "Status:   $PSScriptRoot\status-adaptmath-local.ps1"
            exit 0
        }
        throw (
            'Launcher-owned processes are still tracked, but the full stack is not healthy. ' +
            'Run stop-adaptmath-local.ps1 to stop only those tracked resources, then start again.'
        )
    }
    throw "A stale launcher state file exists at $script:AdaptMathStateFile. Run stop-adaptmath-local.ps1 once before starting."
}

$state = [ordered]@{
    version = 3
    createdAtUtc = [DateTime]::UtcNow.ToString('o')
    selectorMode = $desiredSelectorMode
    policyLineage = $desiredPolicyLineage
    dataMode = 'real'
    tutorPersistence = 'postgres'
    studentModelPersistence = 'postgres'
    researchPersistence = 'postgres'
    authSessionPersistence = 'postgres'
    turnLinTSMode = $desiredTurnLinTSMode
    turnLinTSContextBlocks = $desiredTurnLinTSBlocks
    turnLinTSDiagnosticContextBlocks = 'S+K+L+H+Q'
    turnLinTSRewardMode = $desiredTurnLinTSReward
    turnLinTSAnchorMode = $desiredTurnLinTSAnchorMode
    turnLinTSAnchorGamma = $desiredTurnLinTSAnchorGamma
    turnLinTSStatePath = $desiredTurnLinTSStatePath
    turnLinTSEventRoot = $desiredTurnLinTSEventRoot
    turnLinTSLiveManifestPath = $desiredTurnLinTSLiveManifestPath
    ports = [ordered]@{ memory = $MemoryPort; tutor = $TutorPort; frontend = $FrontendPort }
    postgres = [ordered]@{ startedByLauncher = $false }
    memory = [ordered]@{ startedByLauncher = $false }
    memoryLog = [ordered]@{ startedByLauncher = $false; process = $null }
    tutor = [ordered]@{ startedByLauncher = $false; process = $null }
    frontend = [ordered]@{ startedByLauncher = $false; process = $null }
}
Save-AdaptMathRuntimeState -State $state

if ($ManualControlledLive.IsPresent) {
    $env:ADAPTIVE_SELECTOR_MODE = $desiredSelectorMode
    $env:ADAPTIVE_MANUAL_CONTROLLED_LIVE = 'enabled-v1'
    Write-Host ('=' * 60)
    Write-Host 'MANUAL CONTROLLED-LIVE SELECTOR TEST'
    Write-Host 'ACTIVE: MD7-R1 epoch 3'
    Write-Host 'SHADOW: frozen MD6'
    Write-Host 'LinTS posterior updates: DISABLED'
    Write-Host 'Real adaptive persistence: DISABLED'
    Write-Host ('=' * 60)
}
elseif ($MD6Rollback.IsPresent) {
    $env:ADAPTIVE_SELECTOR_MODE = $desiredSelectorMode
    Remove-Item Env:\ADAPTIVE_MANUAL_CONTROLLED_LIVE -ErrorAction SilentlyContinue
    Write-Host ('=' * 60)
    Write-Host 'ROLLBACK MODE'
    Write-Host 'ACTIVE SELECTOR: frozen MD6'
    Write-Host 'POLICY LINEAGE: preserved MD6'
    Write-Host 'LinTS learning: ENABLED'
    Write-Host ('=' * 60)
}
elseif ($MD7R1Rollback.IsPresent) {
    $env:ADAPTIVE_SELECTOR_MODE = $desiredSelectorMode
    Remove-Item Env:\ADAPTIVE_MANUAL_CONTROLLED_LIVE -ErrorAction SilentlyContinue
    Write-Host ('=' * 60)
    Write-Host 'ROLLBACK MODE'
    Write-Host 'ACTIVE SELECTOR: MD7-R1 epoch 3'
    Write-Host 'POLICY LINEAGE: MD7-R1 fresh LinTS v1'
    Write-Host 'LEGACY TAU/BIAS/DELAYED-CREDIT SEMANTICS: ENABLED'
    Write-Host ('=' * 60)
}
else {
    $env:ADAPTIVE_SELECTOR_MODE = $desiredSelectorMode
    Remove-Item Env:\ADAPTIVE_MANUAL_CONTROLLED_LIVE -ErrorAction SilentlyContinue
    $env:ADAPTIVE_TURN_LINTS_MODE = $desiredTurnLinTSMode
    $env:ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS = $desiredTurnLinTSBlocks
    $env:ADAPTIVE_TURN_LINTS_REWARD_MODE = $desiredTurnLinTSReward
    $env:ADAPTIVE_TURN_LINTS_ANCHOR_MODE = $desiredTurnLinTSAnchorMode
    $env:ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA = [string]$desiredTurnLinTSAnchorGamma
    Write-Host ('=' * 60)
    Write-Host 'ACTIVE SELECTOR: MD7-R2-TELL-C1 Epoch 2'
    Write-Host 'SELECTOR SHA256: ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5'
    Write-Host 'POLICY LINEAGE: direct_disjoint_turn_lints_v1'
    Write-Host "TURN-LINTS MODE: $env:ADAPTIVE_TURN_LINTS_MODE"
    Write-Host "CONTEXT BLOCKS: $env:ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS"
    Write-Host "REWARD MODE: $env:ADAPTIVE_TURN_LINTS_REWARD_MODE"
    Write-Host "ACTION ANCHOR: $env:ADAPTIVE_TURN_LINTS_ANCHOR_MODE"
    Write-Host "ANCHOR GAMMA: $env:ADAPTIVE_TURN_LINTS_ANCHOR_GAMMA"
    Write-Host ('=' * 60)
}

try {
    & docker version --format '{{.Server.Version}}' *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'PostgreSQL unavailable: Docker Desktop is not reachable.'
    }

    $postgresWasRunning = Test-AdaptMathComposeServiceRunning -Service 'postgres'
    if (-not $postgresWasRunning) {
        if (Test-AdaptMathPortInUse -Port 5433) {
            throw 'PostgreSQL port conflict: port 5433 is occupied by a service outside this Compose stack.'
        }
        Push-Location $script:AdaptMathMemoryRoot
        try { & docker compose up -d postgres }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) {
            throw 'PostgreSQL unavailable: Docker Compose could not start the scoped postgres service.'
        }
        $state.postgres.startedByLauncher = $true
        Save-AdaptMathRuntimeState -State $state
    }

    $databaseReady = $false
    $databaseDeadline = [DateTime]::UtcNow.AddSeconds(60)
    while ([DateTime]::UtcNow -lt $databaseDeadline) {
        Push-Location $script:AdaptMathMemoryRoot
        try { & docker compose exec -T postgres pg_isready -U memory_app -d adaptmath_memory_integration *> $null }
        finally { Pop-Location }
        if ($LASTEXITCODE -eq 0) { $databaseReady = $true; break }
        Start-Sleep -Milliseconds 750
    }
    if (-not $databaseReady) {
        throw 'Memory DB connection failure: adaptmath_memory_integration did not become reachable.'
    }

    # Schema creation is an explicit provisioning step. Application startup
    # itself never creates Tutor/BKT/research tables.
    Push-Location $script:AdaptMathMemoryRoot
    try { & docker compose run --rm --build memory_provision }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        throw 'PostgreSQL schema provisioning failed.'
    }

    $memoryWasRunning = Test-AdaptMathComposeServiceRunning -Service 'memory_service'
    if (-not $memoryWasRunning -and (Test-AdaptMathPortInUse -Port $MemoryPort)) {
            throw "port conflict: Memory requires 127.0.0.1:$MemoryPort."
    }
    Push-Location $script:AdaptMathMemoryRoot
    try { & docker compose up -d --build --force-recreate memory_service }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        throw 'Memory startup failure: Docker Compose could not start memory_service.'
    }
    if (-not $memoryWasRunning) {
        $state.memory.startedByLauncher = $true
        Save-AdaptMathRuntimeState -State $state
    }

    Wait-AdaptMathHttp200 -Url "http://127.0.0.1:$MemoryPort/health" -Label 'Memory /health' | Out-Null
    try {
        Wait-AdaptMathHttp200 -Url "http://127.0.0.1:$MemoryPort/ready" -Label 'Memory /ready' | Out-Null
    }
    catch {
        throw "Memory /ready failure (possible database or migration problem): $($_.Exception.Message)"
    }

    $dockerPath = (Get-Command docker).Source
    $memoryStdout = Join-Path $script:AdaptMathLogs 'memory.stdout.log'
    $memoryStderr = Join-Path $script:AdaptMathLogs 'memory.stderr.log'
    $memoryLogProcess = Start-Process -FilePath $dockerPath `
        -ArgumentList @('compose', 'logs', '--follow', '--no-color', '--tail', '200', 'memory_service') `
        -WorkingDirectory $script:AdaptMathMemoryRoot `
        -RedirectStandardOutput $memoryStdout `
        -RedirectStandardError $memoryStderr `
        -WindowStyle Hidden `
        -PassThru
    $state.memoryLog.startedByLauncher = $true
    $state.memoryLog.process = New-AdaptMathProcessRecord -Process $memoryLogProcess
    Save-AdaptMathRuntimeState -State $state

    Push-Location $script:AdaptMathMemoryRoot
    try {
        $memoryKeyLines = @(& docker compose exec -T memory_service printenv MEMORY_SERVICE_API_KEY 2>$null)
    }
    finally { Pop-Location }
    $memoryKey = if ($memoryKeyLines.Count -gt 0) { [string]$memoryKeyLines[-1] } else { '' }
    if ([string]::IsNullOrWhiteSpace($memoryKey)) {
        throw 'Memory service API key missing: MEMORY_SERVICE_API_KEY is not configured in the running container.'
    }

    if (-not (Test-AdaptMathCredentialPresent)) {
        throw 'LLM credential missing: GEMINI_API_KEY is required by the active Tutor Agent provider.'
    }

    $tutorPython = Join-Path $script:AdaptMathTutorBackend '.venv-integration\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $tutorPython)) {
        throw "Tutor import failure: the verified environment is missing at $tutorPython."
    }

    $demoRoot = Join-Path $script:AdaptMathTutorBackend 'runtime\adaptive_live_real_v1'
    $md6PolicyRoot = Join-Path $script:AdaptMathTutorBackend 'runtime\adaptive_demo_md6_real_v1'
    $md7PolicyRoot = Join-Path $script:AdaptMathTutorBackend 'runtime\adaptive_demo_md7r1_real_v1'
    # This launcher is exclusively for genuine interactive tutoring. Synthetic
    # fixtures use their own test runners and must never share this provenance.
    $env:ADAPTIVE_DATA_MODE = 'real'
    $env:ADAPTIVE_LOCAL_RUNTIME_ROOT = $demoRoot
    $env:ADAPTIVE_MD7_RUNTIME_ROOT = $md7PolicyRoot
    Push-Location $script:AdaptMathTutorBackend
    try { & $tutorPython 'scripts\prepare_local_demo.py' }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        throw 'BKT artifact missing: isolated real runtime persistence could not be prepared.'
    }
    if ($ManualControlledLive.IsPresent -or $MD7R1Rollback.IsPresent) {
        Push-Location $script:AdaptMathTutorBackend
        try { & $tutorPython 'scripts\prepare_md7r1_policy_lineage.py' }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) {
            throw 'Legacy MD7-R1 policy lineage could not be prepared or validated.'
        }
    }

    $env:MEMORY_API_URL = "http://127.0.0.1:$MemoryPort"
    $env:MEMORY_SERVICE_API_KEY = $memoryKey
    $env:MEMORY_TIMEOUT_SECONDS = '2'
    $env:MEMORY_TOPIC_TIMEOUT_SECONDS = '15'
    $env:MEMORY_ENABLED = 'true'
    $memoryEnvPath = Join-Path $script:AdaptMathMemoryRoot '.env'
    $databasePassword = Get-AdaptMathDotEnvValue `
        -Path $memoryEnvPath `
        -Name 'MEMORY_POSTGRES_PASSWORD'
    if ([string]::IsNullOrWhiteSpace($databasePassword)) {
        $databasePassword = 'memory-local-dev-only'
    }
    $encodedDatabasePassword = [System.Uri]::EscapeDataString($databasePassword)
    $databaseUrl = "postgresql://memory_app:$encodedDatabasePassword@127.0.0.1:5433/adaptmath_memory_integration"
    $env:ADAPTMATH_DATABASE_URL = $databaseUrl
    $env:TUTOR_PERSISTENCE = 'postgres'
    $env:TUTOR_PERSISTENCE_SCHEMA = 'tutor'
    $env:STUDENT_MODEL_PERSISTENCE = 'postgres'
    $env:STUDENT_MODEL_DATABASE_URL = $databaseUrl
    $env:STUDENT_MODEL_DATABASE_SCHEMA = 'student_model'
    $env:RESEARCH_PERSISTENCE = 'postgres'
    $env:RESEARCH_DATABASE_URL = $databaseUrl
    $env:RESEARCH_DATABASE_SCHEMA = 'research'
    $env:AUTH_DATABASE_SCHEMA = 'auth'
    $env:CHECKPOINT_DB_PATH = Join-Path $demoRoot 'checkpoints.sqlite3'
    $env:ADAPTIVE_MD6_POLICY_STATE_PATH = Join-Path $md6PolicyRoot 'policy_state.json'
    $env:ADAPTIVE_MD6_EXPERIENCE_LOG_PATH = Join-Path $md6PolicyRoot 'attempts.jsonl'
    $env:ADAPTIVE_MD7_POLICY_STATE_PATH = Join-Path $md7PolicyRoot 'policy_state.json'
    $env:ADAPTIVE_MD7_EXPERIENCE_LOG_PATH = Join-Path $md7PolicyRoot 'attempts.jsonl'
    $env:ADAPTIVE_TURN_LINTS_STATE_PATH = $desiredTurnLinTSStatePath
    $env:ADAPTIVE_TURN_LINTS_EVENT_ROOT = $desiredTurnLinTSEventRoot
    $env:ADAPTIVE_TURN_LINTS_LIVE_MANIFEST_PATH = $desiredTurnLinTSLiveManifestPath
    if ($MD6Rollback.IsPresent) {
        $env:ADAPTIVE_POLICY_STATE_PATH = $env:ADAPTIVE_MD6_POLICY_STATE_PATH
        $env:ADAPTIVE_EXPERIENCE_LOG_PATH = $env:ADAPTIVE_MD6_EXPERIENCE_LOG_PATH
    }
    elseif ($ManualControlledLive.IsPresent -or $MD7R1Rollback.IsPresent) {
        $env:ADAPTIVE_POLICY_STATE_PATH = $env:ADAPTIVE_MD7_POLICY_STATE_PATH
        $env:ADAPTIVE_EXPERIENCE_LOG_PATH = $env:ADAPTIVE_MD7_EXPERIENCE_LOG_PATH
    }
    $env:STUDENT_MODEL_REPO = Join-Path $demoRoot 'student_model'
    $env:PEDAGOGICAL_MOVE_REPO = $script:AdaptMathMoveSelector
    $env:CORS_ORIGINS = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"

    Push-Location $script:AdaptMathTutorBackend
    try { & $tutorPython 'scripts\check_local_runtime_artifacts.py' }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        throw 'BKT/MD6/MRB1 artifact preflight failed. Review the preceding named artifact error.'
    }

    $tutorHealthUrl = "http://127.0.0.1:$TutorPort/health"
    $existingTutor = Get-AdaptMathHttpResult -Url $tutorHealthUrl
    if (Test-AdaptMathPortInUse -Port $TutorPort) {
        $isExpectedTutor = $existingTutor.Success -and $existingTutor.Content -match 'adaptmath-backend'
        if (-not $isExpectedTutor) {
            throw "port conflict: Tutor backend requires 127.0.0.1:$TutorPort."
        }
    }
    else {
        $tutorStdout = Join-Path $script:AdaptMathLogs 'tutor.stdout.log'
        $tutorStderr = Join-Path $script:AdaptMathLogs 'tutor.stderr.log'
        $tutorProcess = Start-Process -FilePath $tutorPython `
            -ArgumentList @('-m', 'uvicorn', 'app.local_runtime:app', '--host', '127.0.0.1', '--port', [string]$TutorPort) `
            -WorkingDirectory $script:AdaptMathTutorBackend `
            -RedirectStandardOutput $tutorStdout `
            -RedirectStandardError $tutorStderr `
            -WindowStyle Hidden `
            -PassThru
        $state.tutor.startedByLauncher = $true
        $state.tutor.process = New-AdaptMathProcessRecord -Process $tutorProcess
        Save-AdaptMathRuntimeState -State $state
    }

    try {
        Wait-AdaptMathHttp200 -Url $tutorHealthUrl -Label 'Tutor /health' -TimeoutSeconds 120 | Out-Null
        Wait-AdaptMathHttp200 -Url "http://127.0.0.1:$TutorPort/ready" -Label 'Tutor /ready' -TimeoutSeconds 120 | Out-Null
    }
    catch {
        throw "Tutor import or artifact failure: $($_.Exception.Message) Review tutor.stderr.log."
    }

    $frontendUrl = "http://127.0.0.1:$FrontendPort"
    $existingFrontend = Get-AdaptMathHttpResult -Url $frontendUrl
    if (Test-AdaptMathPortInUse -Port $FrontendPort) {
        $isExpectedFrontend = $existingFrontend.Success -and $existingFrontend.Content -match '<title>AdaptMath</title>'
        if (-not $isExpectedFrontend) {
            throw "port conflict: Adaptive Tutor frontend requires 127.0.0.1:$FrontendPort."
        }
    }
    else {
        if (-not (Test-Path -LiteralPath (Join-Path $script:AdaptMathTutorFrontend 'node_modules'))) {
            throw 'Frontend dependency missing: run npm install in adaptive-math-tutor\frontend.'
        }
        $npmPath = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
        if ([string]::IsNullOrWhiteSpace($npmPath)) {
            throw 'Frontend dependency missing: npm.cmd is not available.'
        }
        $env:VITE_API_BASE_URL = "http://127.0.0.1:$TutorPort"
        $env:VITE_MEMORY_API_BASE_URL = "http://127.0.0.1:$MemoryPort"
        $frontendStdout = Join-Path $script:AdaptMathLogs 'frontend.stdout.log'
        $frontendStderr = Join-Path $script:AdaptMathLogs 'frontend.stderr.log'
        $frontendProcess = Start-Process -FilePath $npmPath `
            -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', [string]$FrontendPort, '--strictPort') `
            -WorkingDirectory $script:AdaptMathTutorFrontend `
            -RedirectStandardOutput $frontendStdout `
            -RedirectStandardError $frontendStderr `
            -WindowStyle Hidden `
            -PassThru
        $state.frontend.startedByLauncher = $true
        $state.frontend.process = New-AdaptMathProcessRecord -Process $frontendProcess
        Save-AdaptMathRuntimeState -State $state
    }

    $frontendResult = Wait-AdaptMathHttp200 -Url $frontendUrl -Label 'Adaptive Tutor frontend' -TimeoutSeconds 60
    if ($frontendResult.Content -notmatch '<title>AdaptMath</title>') {
        throw 'Frontend startup failure: the responding page is not the Adaptive Tutor frontend.'
    }

    Write-Host ''
    Write-Host 'ADAPTMATH LOCAL STACK READY'
    Write-Host "Selector mode:   $desiredSelectorMode"
    Write-Host "Frontend:       $frontendUrl"
    Write-Host "Tutor backend:  http://127.0.0.1:$TutorPort"
    Write-Host "Memory backend: http://127.0.0.1:$MemoryPort"
    Write-Host "Logs:           $script:AdaptMathLogs"
    if ($desiredSelectorMode -eq 'manual-controlled-live-md7r1-v1') {
        $manualResults = Join-Path $script:AdaptMathMoveSelector 'results\md7r1_manual_controlled_live_v1'
        Write-Host "Manual turns:   $(Join-Path $manualResults 'manual_live_turns.jsonl')"
        Write-Host "BKT activity:   $(Join-Path $manualResults 'bkt_activity.jsonl')"
    }
    Write-Host "Status:         $PSScriptRoot\status-adaptmath-local.ps1"
    Write-Host "Stop:           $PSScriptRoot\stop-adaptmath-local.ps1"
}
catch {
    Save-AdaptMathRuntimeState -State $state
    Write-Error $_.Exception.Message
    Write-Host "Use $PSScriptRoot\stop-adaptmath-local.ps1 to stop only resources started by this failed launch."
    exit 1
}
