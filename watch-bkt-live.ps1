[CmdletBinding()]
param(
    [ValidateRange(1, 3600)]
    [int]$RefreshSeconds = 2,

    [ValidateRange(1, 100)]
    [int]$Limit = 15,

    [switch]$Once
)

$ErrorActionPreference = 'Stop'

$memoryRoot = Join-Path $PSScriptRoot 'student-memory-personalization'
if (-not (Test-Path -LiteralPath $memoryRoot -PathType Container)) {
    throw "Memory service directory not found: $memoryRoot"
}

$sql = @'
WITH recent_events AS (
    SELECT *
    FROM student_model.resolved_events
    ORDER BY created_at DESC, event_id DESC
    LIMIT __LIMIT__
)
SELECT
    to_char(created_at AT TIME ZONE 'Asia/Colombo', 'HH24:MI:SS') AS time,
    student_id AS learner,
    skill_name AS skill,
    CASE
        WHEN should_update = 0 THEN 'No evidence'
        WHEN outcome = 1 THEN 'Correct'
        ELSE 'Incorrect'
    END AS result,
    CASE WHEN should_update = 1 THEN 'Yes' ELSE 'No' END AS bkt_update,
    ROUND(mastery_before::numeric, 6) AS mastery_before,
    ROUND(mastery_after::numeric, 6) AS mastery_after,
    ROUND(delta_mastery::numeric, 6) AS mastery_gain,
    ROUND(update_confidence::numeric, 3) AS confidence,
    observation_source AS source
FROM recent_events
ORDER BY created_at, event_id;
'@.Replace('__LIMIT__', [string]$Limit)

Push-Location $memoryRoot
try {
    do {
        if (-not $Once) {
            Clear-Host
        }
        Write-Host "BKT live updates - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        if (-not $Once) {
            Write-Host "Newest event is last. Press Ctrl+C to stop.`n"
        }

        & docker compose exec -T postgres psql `
            -U memory_app `
            -d adaptmath_memory_integration `
            -X `
            -P pager=off `
            -c $sql
        if ($LASTEXITCODE -ne 0) {
            throw "The BKT query failed with exit code $LASTEXITCODE."
        }

        if (-not $Once) {
            Start-Sleep -Seconds $RefreshSeconds
        }
    } while (-not $Once)
}
finally {
    Pop-Location
}
