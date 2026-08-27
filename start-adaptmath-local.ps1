[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$MemoryPort = 8000,
    [ValidateRange(1, 65535)][int]$TutorPort = 8002,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

New-Item -ItemType Directory -Force -Path $script:AdaptMathLogs | Out-Null
New-Item -ItemType Directory -Force -Path $script:AdaptMathStateDirectory | Out-Null

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
            Write-Host 'AdaptMath local stack is already tracked and healthy.'
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
    version = 1
    createdAtUtc = [DateTime]::UtcNow.ToString('o')
    ports = [ordered]@{ memory = $MemoryPort; tutor = $TutorPort; frontend = $FrontendPort }
    postgres = [ordered]@{ startedByLauncher = $false }
    memory = [ordered]@{ startedByLauncher = $false }
    memoryLog = [ordered]@{ startedByLauncher = $false; process = $null }
    tutor = [ordered]@{ startedByLauncher = $false; process = $null }
    frontend = [ordered]@{ startedByLauncher = $false; process = $null }
}
Save-AdaptMathRuntimeState -State $state

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

    $memoryWasRunning = Test-AdaptMathComposeServiceRunning -Service 'memory_service'
    if (-not $memoryWasRunning) {
        if (Test-AdaptMathPortInUse -Port $MemoryPort) {
            throw "port conflict: Memory requires 127.0.0.1:$MemoryPort."
        }
        if ($MemoryPort -ne 8000) {
            throw 'Memory port configuration mismatch: the current Compose service publishes port 8000. Use -MemoryPort 8000.'
        }
        Push-Location $script:AdaptMathMemoryRoot
        try { & docker compose up -d memory_service }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) {
            throw 'Memory startup failure: Docker Compose could not start memory_service.'
        }
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

    $demoRoot = Join-Path $script:AdaptMathTutorBackend 'runtime\adaptive_demo'
    if (-not (Test-Path -LiteralPath $demoRoot)) {
        Push-Location $script:AdaptMathTutorBackend
        try { & $tutorPython 'scripts\prepare_local_demo.py' }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) {
            throw 'BKT artifact missing: isolated local-demo persistence could not be prepared.'
        }
    }

    $env:MEMORY_API_URL = "http://127.0.0.1:$MemoryPort"
    $env:MEMORY_SERVICE_API_KEY = $memoryKey
    $env:MEMORY_TIMEOUT_SECONDS = '2'
    $env:MEMORY_ENABLED = 'true'
    $env:ADAPTIVE_DATA_MODE = 'synthetic'
    $env:CHECKPOINT_DB_PATH = Join-Path $demoRoot 'checkpoints.sqlite3'
    $env:ADAPTIVE_POLICY_STATE_PATH = Join-Path $demoRoot 'policy_state.json'
    $env:ADAPTIVE_EXPERIENCE_LOG_PATH = Join-Path $demoRoot 'attempts.jsonl'
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
    Write-Host "Frontend:       $frontendUrl"
    Write-Host "Tutor backend:  http://127.0.0.1:$TutorPort"
    Write-Host "Memory backend: http://127.0.0.1:$MemoryPort"
    Write-Host "Logs:           $script:AdaptMathLogs"
    Write-Host "Status:         $PSScriptRoot\status-adaptmath-local.ps1"
    Write-Host "Stop:           $PSScriptRoot\stop-adaptmath-local.ps1"
}
catch {
    Save-AdaptMathRuntimeState -State $state
    Write-Error $_.Exception.Message
    Write-Host "Use $PSScriptRoot\stop-adaptmath-local.ps1 to stop only resources started by this failed launch."
    exit 1
}
