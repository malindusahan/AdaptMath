[CmdletBinding()]
param(
    [switch]$Compact,

    [switch]$Policy,

    [string]$StudentId,

    [string]$ThreadId,

    [switch]$SaveSession,

    [switch]$Once,

    [ValidateRange(100, 10000)]
    [int]$PollMilliseconds = 250
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$streamPath = Join-Path $workspaceRoot '_local_runtime_logs\move_selector_live.jsonl'
$memoryRoot = Join-Path $workspaceRoot 'apps\student-memory'
$fallbackPolicyPath = Join-Path $workspaceRoot `
    'apps\tutor\runtime\turn_lints_live_skl_final_v1\policy_state.json'
$script:SessionPath = $null

if ($SaveSession) {
    $sessionRoot = Join-Path $workspaceRoot `
        '_local_runtime_logs\move-selector-sessions'
    New-Item -ItemType Directory -Path $sessionRoot -Force | Out-Null
    $sessionName = 'move-selector-session-{0}.log' -f `
        (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    $script:SessionPath = Join-Path $sessionRoot $sessionName
}

function Write-MonitorLine {
    param(
        [AllowEmptyString()]
        [string]$Text = '',

        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    Write-Host $Text -ForegroundColor $Color
    if ($null -ne $script:SessionPath) {
        Add-Content -LiteralPath $script:SessionPath -Value $Text -Encoding UTF8
    }
}

function Format-MonitorValue {
    param([object]$Value)

    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return '-'
    }
    return [string]$Value
}

function Format-MonitorNumber {
    param([object]$Value)

    if ($null -eq $Value -or $Value -is [bool]) {
        return '-'
    }
    try {
        return ([double]$Value).ToString(
            '0.0000',
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        return '-'
    }
}

function ConvertTo-MonitorStudentId {
    param([string]$Value)

    $normalized = $Value.Trim()
    if ($normalized.StartsWith('psn_')) {
        return $normalized
    }
    $inputBytes = [System.Text.Encoding]::UTF8.GetBytes(
        'adaptmath_turn_outcome_v1:student:{0}' -f $normalized
    )
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash($inputBytes)
    }
    finally {
        $sha256.Dispose()
    }
    $hex = ($digest | ForEach-Object { $_.ToString('x2') }) -join ''
    return 'psn_{0}' -f $hex.Substring(0, 24)
}

function Get-PolicySummary {
    $sql = @'
SELECT json_build_object(
    'source', 'PostgreSQL research.policy_states',
    'policy_key', policy_key,
    'context_dimension', context_dimension,
    'total_updates', total_updates,
    'state', state_json
)::text
FROM research.policy_states
WHERE policy_class = 'DirectTurnLinTS'
  AND context_dimension = 11
  AND data_mode = 'real'
ORDER BY updated_at DESC
LIMIT 1;
'@

    if (Test-Path -LiteralPath $memoryRoot -PathType Container) {
        try {
            Push-Location $memoryRoot
            try {
                $databaseOutput = @(& docker compose exec -T postgres psql `
                    -U memory_app `
                    -d adaptmath_memory_integration `
                    -X `
                    -t `
                    -A `
                    -c $sql 2>$null)
                $databaseExitCode = $LASTEXITCODE
            }
            finally {
                Pop-Location
            }
            if ($databaseExitCode -eq 0) {
                $jsonLine = $databaseOutput |
                    Where-Object { ([string]$_).TrimStart().StartsWith('{') } |
                    Select-Object -Last 1
                if ($null -ne $jsonLine) {
                    return ([string]$jsonLine | ConvertFrom-Json)
                }
            }
        }
        catch {
            # The file below is the validated bootstrap/fallback snapshot.
        }
    }

    if (Test-Path -LiteralPath $fallbackPolicyPath -PathType Leaf) {
        $state = Get-Content -LiteralPath $fallbackPolicyPath -Raw |
            ConvertFrom-Json
        return [pscustomobject]@{
            source = 'Runtime JSON fallback (PostgreSQL unavailable)'
            policy_key = $fallbackPolicyPath
            context_dimension = $state.context_dimension
            total_updates = $state.update_count
            state = $state
        }
    }
    return $null
}

function Show-PolicySummary {
    $summary = Get-PolicySummary
    Write-MonitorLine ''
    Write-MonitorLine 'CURRENT LinTS POLICY' Cyan
    Write-MonitorLine '--------------------' Cyan
    if ($null -eq $summary) {
        Write-MonitorLine 'Policy state is unavailable (read-only lookup failed).' Yellow
        return
    }

    $state = $summary.state
    if ($null -ne $state.policy_state) {
        $state = $state.policy_state
    }
    Write-MonitorLine ('Source        : {0}' -f $summary.source)
    Write-MonitorLine ('Policy key    : {0}' -f $summary.policy_key)
    Write-MonitorLine ('Total updates : {0}' -f $summary.total_updates)
    Write-MonitorLine ''
    foreach ($arm in @('generic', 'probing', 'focus', 'telling')) {
        $count = $state.arm_update_counts.$arm
        Write-MonitorLine ('{0,-13} : {1}' -f $arm, $count)
    }

    $aRows = 0
    $aColumns = 0
    $bLength = 0
    if ($null -ne $state.A.generic) {
        $aRows = @($state.A.generic).Count
        if ($aRows -gt 0) {
            $aColumns = @($state.A.generic[0]).Count
        }
    }
    if ($null -ne $state.b.generic) {
        $bLength = @($state.b.generic).Count
    }
    Write-MonitorLine ''
    Write-MonitorLine ('Context       : {0} dimensions' -f `
        $summary.context_dimension)
    Write-MonitorLine ('Posterior/arm : A={0}x{1}, b={2}' -f `
        $aRows, $aColumns, $bLength)
    Write-MonitorLine ''
}

function Test-EventFilter {
    param([object]$Event)

    if (-not [string]::IsNullOrWhiteSpace($StudentId)) {
        $studentFilter = ConvertTo-MonitorStudentId $StudentId
        if ([string]$Event.student_id -ne $studentFilter) {
            return $false
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($ThreadId) -and
        [string]$Event.thread_id -ne $ThreadId) {
        return $false
    }
    return $true
}

function Show-CompactEvent {
    param([object]$Event)

    $clock = '-'
    try { $clock = ([datetimeoffset]$Event.timestamp).ToLocalTime().ToString('HH:mm:ss') }
    catch { }

    if ($Event.event_type -eq 'PRE_ACTION') {
        $p = $Event.md7_probabilities
        $probabilities = @(
            (Format-MonitorNumber $p.generic),
            (Format-MonitorNumber $p.probing),
            (Format-MonitorNumber $p.focus),
            (Format-MonitorNumber $p.telling)
        ) -join ','
        Write-MonitorLine (
            '[{0}] PRE  turn={1} skill={2} MD7=[{3}] selected={4}' -f `
                $clock,
                $Event.turn_index,
                (Format-MonitorValue $Event.target_skill),
                $probabilities,
                $Event.selected_move
        ) Green
        return
    }

    if ($Event.event_type -eq 'POST_UPDATE') {
        Write-MonitorLine (
            '[{0}] POST turn={1} selected={2} mastery={3}->{4} reward={5} updates={6}' -f `
                $clock,
                $Event.turn_index,
                $Event.selected_move,
                (Format-MonitorNumber $Event.mastery_before),
                (Format-MonitorNumber $Event.mastery_after),
                (Format-MonitorNumber $Event.final_reward),
                $Event.policy_update_count_after
        ) Magenta
    }
}

function Show-PreAction {
    param([object]$Event)

    $p = $Event.md7_probabilities
    $k = $Event.knowledge_state
    $l = $Event.learner_response
    $context = $Event.context
    $scores = $Event.sampled_scores

    Write-MonitorLine ('=' * 60) Cyan
    Write-MonitorLine 'MOVE SELECTOR - PRE ACTION' Green
    Write-MonitorLine ('=' * 60) Cyan
    Write-MonitorLine ''
    Write-MonitorLine ('Time         : {0}' -f (Format-MonitorValue $Event.timestamp))
    Write-MonitorLine ('Student      : {0}' -f (Format-MonitorValue $Event.student_id))
    Write-MonitorLine ('Thread       : {0}' -f (Format-MonitorValue $Event.thread_id))
    Write-MonitorLine ('Attempt      : {0}' -f (Format-MonitorValue $Event.attempt_id))
    Write-MonitorLine ('Turn         : {0}' -f (Format-MonitorValue $Event.turn_index))
    Write-MonitorLine ('Target Skill : {0}' -f (Format-MonitorValue $Event.target_skill))
    Write-MonitorLine ''
    Write-MonitorLine 'SUPERVISED MODEL' Yellow
    Write-MonitorLine ('Model        : {0}' -f (Format-MonitorValue $Event.supervised_model))
    Write-MonitorLine ('Checkpoint   : {0}' -f (Format-MonitorValue $Event.checkpoint_identifier))
    Write-MonitorLine ('SHA256       : {0}' -f (Format-MonitorValue $Event.selector_sha256))
    Write-MonitorLine ''
    Write-MonitorLine 'MD7 PRIOR (S)' Yellow
    Write-MonitorLine ('Generic      : {0}' -f (Format-MonitorNumber $p.generic))
    Write-MonitorLine ('Probing      : {0}' -f (Format-MonitorNumber $p.probing))
    Write-MonitorLine ('Focus        : {0}' -f (Format-MonitorNumber $p.focus))
    Write-MonitorLine ('Telling      : {0}' -f (Format-MonitorNumber $p.telling))
    Write-MonitorLine ''
    Write-MonitorLine 'KNOWLEDGE STATE (K)' Yellow
    Write-MonitorLine ('Mastery Before : {0}' -f (Format-MonitorNumber $k.mastery_before))
    Write-MonitorLine ('Previous Delta : {0}' -f (Format-MonitorNumber $k.previous_mastery_delta))
    Write-MonitorLine ('Missing        : {0}' -f (Format-MonitorNumber $k.missing))
    Write-MonitorLine ''
    Write-MonitorLine 'LEARNER RESPONSE (L)' Yellow
    Write-MonitorLine ('Reasoning      : {0}' -f (Format-MonitorNumber $l.reasoning))
    Write-MonitorLine ('Uncertainty    : {0}' -f (Format-MonitorNumber $l.uncertainty))
    Write-MonitorLine ('Clarification  : {0}' -f (Format-MonitorNumber $l.clarification))
    Write-MonitorLine ('Missing        : {0}' -f (Format-MonitorNumber $l.missing))
    Write-MonitorLine ''
    Write-MonitorLine 'CONTEXT' Yellow
    Write-MonitorLine ('Dimensions   : {0}' -f $context.context_dimension)
    $formattedVector = @($context.vector | ForEach-Object {
        Format-MonitorNumber $_
    }) -join ', '
    Write-MonitorLine ('Vector       : [{0}]' -f $formattedVector)
    Write-MonitorLine ''
    Write-MonitorLine 'THOMPSON SAMPLES (ACTUAL SAMPLES USED)' Yellow
    Write-MonitorLine ('Generic      : {0}' -f (Format-MonitorNumber $scores.generic))
    Write-MonitorLine ('Probing      : {0}' -f (Format-MonitorNumber $scores.probing))
    Write-MonitorLine ('Focus        : {0}' -f (Format-MonitorNumber $scores.focus))
    Write-MonitorLine ('Telling      : {0}' -f (Format-MonitorNumber $scores.telling))
    Write-MonitorLine ''
    Write-MonitorLine ('SELECTED MOVE: {0}' -f $Event.selected_move) Green
    Write-MonitorLine ''
    Write-MonitorLine ('=' * 60) Cyan
}

function Show-PostUpdate {
    param([object]$Event)

    $persistence = $Event.persistence
    $mrb1 = $Event.mrb1.scores
    Write-MonitorLine ('=' * 60) Cyan
    Write-MonitorLine 'MOVE SELECTOR - POST UPDATE' Magenta
    Write-MonitorLine ('=' * 60) Cyan
    Write-MonitorLine ''
    Write-MonitorLine ('Time         : {0}' -f (Format-MonitorValue $Event.timestamp))
    Write-MonitorLine ('Student      : {0}' -f (Format-MonitorValue $Event.student_id))
    Write-MonitorLine ('Thread       : {0}' -f (Format-MonitorValue $Event.thread_id))
    Write-MonitorLine ('Attempt      : {0}' -f (Format-MonitorValue $Event.attempt_id))
    Write-MonitorLine ('Turn         : {0}' -f (Format-MonitorValue $Event.turn_index))
    Write-MonitorLine ''
    Write-MonitorLine ('Selected Move: {0}' -f $Event.selected_move) Magenta
    Write-MonitorLine ''
    Write-MonitorLine 'MASTERY' Yellow
    Write-MonitorLine ('Before       : {0}' -f (Format-MonitorNumber $Event.mastery_before))
    Write-MonitorLine ('After        : {0}' -f (Format-MonitorNumber $Event.mastery_after))
    Write-MonitorLine ('Raw Delta    : {0}' -f (Format-MonitorNumber $Event.raw_mastery_delta))
    Write-MonitorLine ''
    Write-MonitorLine 'REWARD' Yellow
    Write-MonitorLine ('Reward Type  : {0}' -f $Event.reward_type)
    Write-MonitorLine ('Headroom     : {0}' -f (Format-MonitorNumber $Event.headroom_denominator))
    Write-MonitorLine ('Normalized   : {0}' -f (Format-MonitorNumber $Event.headroom_normalized_reward))
    Write-MonitorLine ('Final Reward : {0}' -f (Format-MonitorNumber $Event.final_reward))
    Write-MonitorLine ''
    Write-MonitorLine 'LinTS UPDATE' Yellow
    Write-MonitorLine ('Arm          : {0}' -f $Event.updated_arm)
    Write-MonitorLine ('Count Before : {0}' -f $Event.arm_update_count_before)
    Write-MonitorLine ('Count After  : {0}' -f $Event.arm_update_count_after)
    Write-MonitorLine ('Total Updates: {0}' -f $Event.policy_update_count_after)
    Write-MonitorLine ('Persistence  : {0} ({1} -> {2})' -f `
        $persistence.status, $persistence.mode, $persistence.destination)
    Write-MonitorLine ''
    Write-MonitorLine 'MRB1' Yellow
    Write-MonitorLine ('Mistake Identification : {0}' -f `
        (Format-MonitorNumber $mrb1.Mistake_Identification))
    Write-MonitorLine ('Mistake Location       : {0}' -f `
        (Format-MonitorNumber $mrb1.Mistake_Location))
    Write-MonitorLine ('Providing Guidance     : {0}' -f `
        (Format-MonitorNumber $mrb1.Providing_Guidance))
    Write-MonitorLine ('Actionability          : {0}' -f `
        (Format-MonitorNumber $mrb1.Actionability))
    Write-MonitorLine ('Role                    : {0}' -f $Event.mrb1.role) Magenta
    Write-MonitorLine ''
    Write-MonitorLine ('=' * 60) Cyan
}

function Show-MoveSelectorEvent {
    param([object]$Event)

    if (-not (Test-EventFilter $Event)) {
        return
    }
    if ($Compact) {
        Show-CompactEvent $Event
    }
    elseif ($Event.event_type -eq 'PRE_ACTION') {
        Show-PreAction $Event
    }
    elseif ($Event.event_type -eq 'POST_UPDATE') {
        Show-PostUpdate $Event
    }
}

if ($null -ne $script:SessionPath) {
    Write-MonitorLine ('Session file  : {0}' -f $script:SessionPath) Cyan
}
if ($Policy) {
    Show-PolicySummary
}
if ($Once) {
    return
}

Write-MonitorLine 'LIVE MOVE-SELECTOR MONITOR' Cyan
Write-MonitorLine ('Stream        : {0}' -f $streamPath)
Write-MonitorLine 'Debug flag    : ADAPTMATH_MOVE_SELECTOR_DEBUG=1'
if (-not [string]::IsNullOrWhiteSpace($StudentId)) {
    Write-MonitorLine 'Student filter: pseudonymous match configured'
}
if (-not [string]::IsNullOrWhiteSpace($ThreadId)) {
    Write-MonitorLine ('Thread filter : {0}' -f $ThreadId)
}
Write-MonitorLine 'Waiting for new events. Press Ctrl+C to stop.' Green
Write-MonitorLine ''

$position = [int64]0
$carry = ''
$streamObserved = $false
$seenEvents = New-Object 'System.Collections.Generic.HashSet[string]'

if (Test-Path -LiteralPath $streamPath -PathType Leaf) {
    $position = (Get-Item -LiteralPath $streamPath).Length
    $streamObserved = $true
}

while ($true) {
    if (-not (Test-Path -LiteralPath $streamPath -PathType Leaf)) {
        $position = [int64]0
        $carry = ''
        $streamObserved = $false
        Start-Sleep -Milliseconds $PollMilliseconds
        continue
    }

    $fileInfo = Get-Item -LiteralPath $streamPath
    if ($fileInfo.Length -lt $position) {
        # A restarted backend may recreate/truncate its diagnostic stream.
        $position = [int64]0
        $carry = ''
    }
    if (-not $streamObserved) {
        $streamObserved = $true
        Write-MonitorLine 'Diagnostic stream is available.' Green
    }
    if ($fileInfo.Length -eq $position) {
        Start-Sleep -Milliseconds $PollMilliseconds
        continue
    }

    $chunk = ''
    try {
        $fileStream = New-Object System.IO.FileStream(
            $streamPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
        )
        try {
            [void]$fileStream.Seek($position, [System.IO.SeekOrigin]::Begin)
            $reader = New-Object System.IO.StreamReader(
                $fileStream,
                (New-Object System.Text.UTF8Encoding($false)),
                $true,
                4096,
                $true
            )
            try {
                $chunk = $reader.ReadToEnd()
                $position = $fileStream.Length
            }
            finally {
                $reader.Dispose()
            }
        }
        finally {
            $fileStream.Dispose()
        }
    }
    catch {
        Start-Sleep -Milliseconds $PollMilliseconds
        continue
    }

    $combined = $carry + $chunk
    $parts = [regex]::Split($combined, "`r?`n")
    if ($combined.EndsWith("`n")) {
        $carry = ''
        $lineCount = $parts.Count - 1
    }
    else {
        $carry = $parts[-1]
        $lineCount = $parts.Count - 1
    }

    for ($index = 0; $index -lt $lineCount; $index++) {
        $line = $parts[$index].Trim()
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $event = $line | ConvertFrom-Json
            if ($event.schema_version -ne 'adaptmath_move_selector_debug_v1') {
                continue
            }
            $eventKey = '{0}|{1}' -f $event.event_type, $event.action_event_id
            if ($seenEvents.Add($eventKey)) {
                Show-MoveSelectorEvent $event
            }
        }
        catch {
            Write-MonitorLine 'Skipped one malformed diagnostic JSON line.' Yellow
        }
    }

    Start-Sleep -Milliseconds $PollMilliseconds
}
