[CmdletBinding()]
param(
    [ValidateRange(1, 3600)]
    [int]$RefreshSeconds = 2,

    [switch]$Once
)

$ErrorActionPreference = 'Stop'

$memoryRoot = Join-Path $PSScriptRoot 'student-memory-personalization'
if (-not (Test-Path -LiteralPath $memoryRoot -PathType Container)) {
    throw "Memory service directory not found: $memoryRoot"
}

$sql = @'
WITH latest_attempt AS (
    SELECT attempt_id
    FROM research.turn_actions
    WHERE payload->>'observation_origin' = 'live_tutor'
    ORDER BY created_at DESC
    LIMIT 1
)
SELECT
    turn_index AS turn,
    payload->>'selected_arm' AS action,
    CASE
        WHEN status = 'PENDING'
            THEN 'Awaiting next learner response'
        WHEN COALESCE((payload->>'outcome_observed')::boolean, false)
            THEN 'Reward ' || ROUND(
                (payload->>'headroom_normalized_delta')::numeric,
                6
            )::text
        ELSE 'No BKT update'
    END AS outcome,
    CASE
        WHEN status = 'PENDING' THEN 'Pending'
        WHEN COALESCE(
            (payload->>'posterior_update_occurred')::boolean,
            false
        ) THEN 'Yes'
        ELSE 'No'
    END AS lints_update
FROM research.turn_actions
WHERE attempt_id = (SELECT attempt_id FROM latest_attempt)
  AND payload->>'observation_origin' = 'live_tutor'
ORDER BY turn_index;
'@

Push-Location $memoryRoot
try {
    do {
        if (-not $Once) {
            Clear-Host
        }
        Write-Host "LinTS live updates - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        if (-not $Once) {
            Write-Host "Press Ctrl+C to stop.`n"
        }

        & docker compose exec -T postgres psql `
            -U memory_app `
            -d adaptmath_memory_integration `
            -X `
            -P pager=off `
            -c $sql
        if ($LASTEXITCODE -ne 0) {
            throw "The LinTS query failed with exit code $LASTEXITCODE."
        }

        if (-not $Once) {
            Start-Sleep -Seconds $RefreshSeconds
        }
    } while (-not $Once)
}
finally {
    Pop-Location
}
