Set-StrictMode -Version Latest

$script:AdaptMathWorkspaceRoot = $PSScriptRoot
$script:AdaptMathMemoryRoot = Join-Path $script:AdaptMathWorkspaceRoot 'student-memory-personalization'
$script:AdaptMathTutorBackend = Join-Path $script:AdaptMathWorkspaceRoot 'adaptive-math-tutor\backend'
$script:AdaptMathTutorFrontend = Join-Path $script:AdaptMathWorkspaceRoot 'adaptive-math-tutor\frontend'
$script:AdaptMathStudentModel = Join-Path $script:AdaptMathWorkspaceRoot 'student-modeling'
$script:AdaptMathMoveSelector = Join-Path $script:AdaptMathWorkspaceRoot 'pedagogical-move-selection'
$script:AdaptMathLogs = Join-Path $script:AdaptMathWorkspaceRoot '_local_runtime_logs'
$script:AdaptMathStateDirectory = Join-Path $script:AdaptMathWorkspaceRoot '_local_runtime_state'
$script:AdaptMathStateFile = Join-Path $script:AdaptMathStateDirectory 'adaptmath-local.json'

function Get-AdaptMathHttpResult {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 3
    )

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return [pscustomobject]@{
            Success = $true
            StatusCode = [int]$response.StatusCode
            Content = [string]$response.Content
        }
    }
    catch {
        $statusCode = 0
        if ($null -ne $_.Exception.Response) {
            try { $statusCode = [int]$_.Exception.Response.StatusCode.value__ } catch { }
        }
        return [pscustomobject]@{
            Success = $false
            StatusCode = $statusCode
            Content = ''
        }
    }
}

function Wait-AdaptMathHttp200 {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$TimeoutSeconds = 90
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $result = Get-AdaptMathHttpResult -Url $Url
        if ($result.Success -and $result.StatusCode -eq 200) {
            return $result
        }
        Start-Sleep -Milliseconds 750
    }
    throw "$Label did not return HTTP 200 within $TimeoutSeconds seconds."
}

function Test-AdaptMathPortInUse {
    param([Parameter(Mandatory = $true)][int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(300)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Test-AdaptMathComposeServiceRunning {
    param([Parameter(Mandatory = $true)][string]$Service)

    Push-Location $script:AdaptMathMemoryRoot
    try {
        $containerId = & docker compose ps --status running -q $Service 2>$null
        return ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace(($containerId -join '')))
    }
    finally {
        Pop-Location
    }
}

function Test-AdaptMathCredentialPresent {
    $environmentValue = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY', 'Process')
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) {
        return $true
    }

    $environmentFile = Join-Path $script:AdaptMathStudentModel '.env'
    if (-not (Test-Path -LiteralPath $environmentFile)) {
        return $false
    }
    foreach ($line in Get-Content -LiteralPath $environmentFile) {
        if ($line -match '^\s*GEMINI_API_KEY\s*=\s*(.+?)\s*$') {
            $value = $Matches[1].Trim().Trim('"').Trim("'")
            return (-not [string]::IsNullOrWhiteSpace($value))
        }
    }
    return $false
}

function Save-AdaptMathRuntimeState {
    param([Parameter(Mandatory = $true)]$State)

    New-Item -ItemType Directory -Force -Path $script:AdaptMathStateDirectory | Out-Null
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $script:AdaptMathStateFile -Encoding UTF8
}

function New-AdaptMathProcessRecord {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process)

    return [ordered]@{
        pid = $Process.Id
        startedAtUtc = $Process.StartTime.ToUniversalTime().ToString('o')
    }
}

function Test-AdaptMathTrackedProcess {
    param($Record)

    if ($null -eq $Record -or $null -eq $Record.pid) {
        return $false
    }
    $process = Get-Process -Id ([int]$Record.pid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $false
    }
    try {
        $actual = $process.StartTime.ToUniversalTime()
        $expected = [DateTime]::Parse([string]$Record.startedAtUtc).ToUniversalTime()
        return ([Math]::Abs(($actual - $expected).TotalSeconds) -lt 2)
    }
    catch {
        return $false
    }
}

function Stop-AdaptMathProcessTree {
    param($Record)

    if (-not (Test-AdaptMathTrackedProcess -Record $Record)) {
        return
    }

    $rootPid = [int]$Record.pid
    $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)

    function Stop-Descendants {
        param([int]$ParentPid)
        foreach ($child in @($allProcesses | Where-Object { $_.ParentProcessId -eq $ParentPid })) {
            Stop-Descendants -ParentPid ([int]$child.ProcessId)
            Stop-Process -Id ([int]$child.ProcessId) -Force -ErrorAction SilentlyContinue
        }
    }

    Stop-Descendants -ParentPid $rootPid
    Stop-Process -Id $rootPid -Force -ErrorAction SilentlyContinue
}

