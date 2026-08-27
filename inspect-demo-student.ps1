[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')]
    [string]$StudentId
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

if (-not (Test-AdaptMathComposeServiceRunning -Service 'postgres')) {
    throw 'PostgreSQL is not running for the AdaptMath Memory stack.'
}

$query = @"
SELECT 'STUDENT' AS section, external_student_id, is_active, created_at
FROM student_memory.students
WHERE external_student_id = '$StudentId';

SELECT 'LEARNING_SESSION' AS section, ls.external_session_id, ls.status,
       ls.started_at, ls.ended_at, COUNT(il.interaction_id) AS interaction_count
FROM student_memory.students s
JOIN student_memory.learning_sessions ls ON ls.student_id = s.student_id
LEFT JOIN student_memory.interaction_logs il ON il.session_id = ls.session_id
WHERE s.external_student_id = '$StudentId'
GROUP BY ls.session_id, ls.external_session_id, ls.status, ls.started_at, ls.ended_at
ORDER BY ls.started_at;

SELECT 'COMPLETED_ATTEMPT' AS section, r.external_attempt_id, r.status,
       r.session_status, r.question_count, r.correct_count, r.incorrect_count,
       r.created_at
FROM student_memory.students s
JOIN student_memory.completed_attempt_receipts r ON r.student_id = s.student_id
WHERE s.external_student_id = '$StudentId'
ORDER BY r.created_at;

SELECT 'INTERACTION_TOTAL' AS section, COUNT(*) AS interaction_count,
       COUNT(*) FILTER (WHERE il.is_correct IS TRUE) AS correct_count,
       COUNT(*) FILTER (WHERE il.is_correct IS FALSE) AS incorrect_count
FROM student_memory.students s
JOIN student_memory.interaction_logs il ON il.student_id = s.student_id
WHERE s.external_student_id = '$StudentId';

SELECT 'MISCONCEPTION' AS section, cs.canonical_name,
       LEFT(sm.display_error, 160) AS display_error,
       sm.occurrence_count, sm.last_seen_at
FROM student_memory.students s
JOIN student_memory.student_misconceptions sm ON sm.student_id = s.student_id
JOIN student_memory.canonical_skills cs ON cs.skill_id = sm.canonical_skill_id
WHERE s.external_student_id = '$StudentId'
ORDER BY sm.last_seen_at DESC;
"@

Push-Location $script:AdaptMathMemoryRoot
try {
    & docker compose exec -T postgres psql `
        -U memory_app `
        -d adaptmath_memory_integration `
        -v ON_ERROR_STOP=1 `
        -P pager=off `
        -c $query
    if ($LASTEXITCODE -ne 0) {
        throw 'The read-only student inspection query failed.'
    }
}
finally {
    Pop-Location
}

