[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$MemoryPort = 8400,
    [ValidateRange(1, 65535)][int]$TutorPort = 8402,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Continue'
. (Join-Path $PSScriptRoot 'adaptmath-local-common.ps1')

$state = $null
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

$selectorMode = if (
    $null -ne $state -and
    $state.PSObject.Properties.Name -contains 'selectorMode'
) {
    [string]$state.selectorMode
}
else {
    'ordinary-md7-r2-tell-c1-v1'
}
$policyLineage = if (
    $null -ne $state -and
    $state.PSObject.Properties.Name -contains 'policyLineage'
) {
    [string]$state.policyLineage
}
elseif ($selectorMode -eq 'md6-rollback-v1') {
    'preserved-md6'
}
else {
    'direct_disjoint_turn_lints_v1'
}
$dataMode = if (
    $null -ne $state -and
    $state.PSObject.Properties.Name -contains 'dataMode'
) {
    [string]$state.dataMode
}
else {
    'unknown'
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
$authSchemaReady = $false
$memorySchemaReady = $false
$tutorPersistenceReady = $false
$studentModelReady = $false
$researchPersistenceReady = $false
if ($postgresUp) {
    Push-Location $script:AdaptMathMemoryRoot
    try { & docker compose exec -T postgres pg_isready -U memory_app -d adaptmath_memory_integration *> $null }
    finally { Pop-Location }
    $databaseOk = $LASTEXITCODE -eq 0
    if ($databaseOk) {
        Push-Location $script:AdaptMathMemoryRoot
        try {
            $schemaChecks = @(& docker compose exec -T postgres psql `
                -U memory_app -d adaptmath_memory_integration -X -At `
                -c "SELECT 'auth|' || ((to_regclass('auth.sessions') IS NOT NULL AND to_regclass('auth.user_accounts') IS NOT NULL)::int); SELECT 'memory|' || ((to_regclass('student_memory.students') IS NOT NULL AND to_regclass('student_memory.completed_attempt_receipts') IS NOT NULL)::int); SELECT 'tutor|' || ((to_regclass('tutor.checkpoints') IS NOT NULL AND to_regclass('tutor.checkpoint_writes') IS NOT NULL)::int); SELECT 'student_model|' || ((to_regclass('student_model.mastery') IS NOT NULL AND to_regclass('student_model.resolved_events') IS NOT NULL)::int); SELECT 'research|' || ((to_regclass('research.policy_states') IS NOT NULL AND to_regclass('research.turn_actions') IS NOT NULL)::int);" 2>$null)
        }
        finally { Pop-Location }
        $authSchemaReady = $schemaChecks -contains 'auth|1'
        $memorySchemaReady = $schemaChecks -contains 'memory|1'
        $tutorPersistenceReady = $schemaChecks -contains 'tutor|1'
        $studentModelReady = $schemaChecks -contains 'student_model|1'
        $researchPersistenceReady = $schemaChecks -contains 'research|1'
    }
}

$llmCredential = if (Test-AdaptMathCredentialPresent) { 'PRESENT' } else { 'MISSING' }

Write-Host "PostgreSQL: $(if ($postgresUp) { 'UP' } else { 'DOWN' })"
Write-Host "Auth schema: $(if ($authSchemaReady) { 'READY' } else { 'NOT READY' })"
Write-Host "Memory schema: $(if ($memorySchemaReady) { 'READY' } else { 'NOT READY' })"
Write-Host "Tutor persistence: $(if ($tutorPersistenceReady) { 'READY' } else { 'NOT READY' })"
Write-Host "Student model: $(if ($studentModelReady) { 'READY' } else { 'NOT READY' })"
Write-Host "Research persistence: $(if ($researchPersistenceReady) { 'READY' } else { 'NOT READY' })"
Write-Host ''
Write-Host "Memory: $(if ($memoryContainerUp) { 'UP' } else { 'DOWN' })"
Write-Host "health: $(if ($memoryHealth.Success) { '200' } else { 'FAIL' })"
Write-Host "ready:  $(if ($memoryReady.Success) { 'READY' } else { 'NOT READY' })"
Write-Host ''
Write-Host "Adaptive Tutor backend: $(if ($tutorHealth.Success) { 'UP' } else { 'DOWN' })"
Write-Host "selector mode: $selectorMode"
Write-Host "policy lineage: $policyLineage"
Write-Host "data mode: $dataMode"
Write-Host "legacy passive import source (not runtime authority): $(Join-Path $script:AdaptMathMoveSelector 'results\md_self_improvement_turn_data_v1\turn_outcomes.jsonl')"
Write-Host "legacy BKT import source (not runtime authority): $(Join-Path $script:AdaptMathMoveSelector 'results\md_self_improvement_turn_data_v1\bkt_activity.jsonl')"
if ($selectorMode -eq 'manual-controlled-live-md7r1-v1') {
    Write-Host 'active selector: MD7-R1 epoch 3'
    Write-Host 'shadow selector: frozen MD6'
    Write-Host 'LinTS learning: DISABLED'
    $manualResults = Join-Path $script:AdaptMathMoveSelector 'results\md7r1_manual_controlled_live_v1'
    Write-Host "manual turn log: $(Join-Path $manualResults 'manual_live_turns.jsonl')"
    Write-Host "BKT activity log: $(Join-Path $manualResults 'bkt_activity.jsonl')"
}
elseif ($selectorMode -eq 'md6-rollback-v1') {
    Write-Host 'active selector: frozen MD6'
    Write-Host 'LinTS learning: ENABLED'
}
elseif ($selectorMode -eq 'md7r1-rollback-v1') {
    Write-Host 'active selector: MD7-R1 epoch 3'
    Write-Host 'legacy tau/bias/delayed-credit LinTS: ENABLED'
}
else {
    Write-Host 'active selector: MD7-R2-TELL-C1 Epoch 2'
    Write-Host 'selector SHA256: ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5'
    $turnMode = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSMode') { [string]$state.turnLinTSMode } else { 'SHADOW' }
    $turnBlocks = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSContextBlocks') { [string]$state.turnLinTSContextBlocks } else { 'S+K+L' }
    $turnDiagnosticBlocks = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSDiagnosticContextBlocks') { [string]$state.turnLinTSDiagnosticContextBlocks } else { 'S+K+L+H+Q' }
    $turnReward = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSRewardMode') { [string]$state.turnLinTSRewardMode } else { 'headroom_normalized' }
    $turnAnchorMode = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSAnchorMode') { [string]$state.turnLinTSAnchorMode } else { 'none' }
    $turnAnchorGamma = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSAnchorGamma') { [string]$state.turnLinTSAnchorGamma } else { '0.0' }
    $turnStatePath = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSStatePath') { [string]$state.turnLinTSStatePath } else { Join-Path $script:AdaptMathTutorBackend 'runtime\turn_lints_shadow_skl_final_v1\policy_state.json' }
    $turnEventRoot = if ($null -ne $state -and $state.PSObject.Properties.Name -contains 'turnLinTSEventRoot') { [string]$state.turnLinTSEventRoot } else { Split-Path -Parent $turnStatePath }
    $turnBehaviorPolicy = if ($turnMode -eq 'RANDOMIZED_WARMSTART') { 'md7_r2_probability_proportional_v1' } elseif ($turnMode -eq 'LIVE') { 'turn_lints_live_v1' } else { 'frozen_md7_top1_v1' }
    $turnContextDimension = 'unknown'
    $turnPosteriorUpdateCount = 'unknown'
    $turnSyntheticRecordCount = 0
    $turnSyntheticEffectiveWeight = 0.0
    $turnRealLiveUpdateCount = 0
    $turnRealLiveActionCount = 0
    $turnRealLiveResolvedRewardCount = 0
    $turnRealLiveNoRewardCount = 0
    $turnRealLivePendingCount = 0
    $turnAgencyOverrideCount = 0
    $turnRealSelections = [ordered]@{ generic = 0; probing = 0; focus = 0; telling = 0 }
    $turnPosteriorUpdatesEnabled = $turnMode -eq 'LIVE'
    $turnMetricsFromPostgres = $false
    if (Test-Path -LiteralPath $turnStatePath) {
        try {
            $turnPolicyState = Get-Content -Raw -LiteralPath $turnStatePath | ConvertFrom-Json
            if ($turnPolicyState.PSObject.Properties.Name -contains 'ordered_feature_names') {
                $turnContextDimension = @($turnPolicyState.ordered_feature_names).Count
            }
            if ($turnPolicyState.PSObject.Properties.Name -contains 'update_count') {
                $turnPosteriorUpdateCount = [int]$turnPolicyState.update_count
            }
        }
        catch {
            Write-Warning "Turn-LinTS state could not be parsed: $turnStatePath"
        }
    }
    $turnManifestPath = Join-Path $turnEventRoot 'lineage_manifest.json'
    if (Test-Path -LiteralPath $turnManifestPath) {
        try {
            $turnManifest = Get-Content -Raw -LiteralPath $turnManifestPath | ConvertFrom-Json
            if ($turnManifest.PSObject.Properties.Name -contains 'behavior_policy') {
                $turnBehaviorPolicy = [string]$turnManifest.behavior_policy
            }
            if ($turnManifest.PSObject.Properties.Name -contains 'context_dimension') {
                $turnContextDimension = [int]$turnManifest.context_dimension
            }
            if ($turnManifest.PSObject.Properties.Name -contains 'posterior_updates_enabled') {
                $turnPosteriorUpdatesEnabled = [bool]$turnManifest.posterior_updates_enabled
            }
            if ($turnManifest.PSObject.Properties.Name -contains 'synthetic_observation_count') {
                $turnSyntheticRecordCount = [int]$turnManifest.synthetic_observation_count
            }
            if ($turnManifest.PSObject.Properties.Name -contains 'synthetic_effective_sample_size') {
                $turnSyntheticEffectiveWeight = [double]$turnManifest.synthetic_effective_sample_size
            }
        }
        catch {
            Write-Warning "Turn-LinTS lineage manifest could not be parsed: $turnManifestPath"
        }
    }
    if ($turnMode -eq 'LIVE' -and $databaseOk -and $researchPersistenceReady) {
        $policyKey = "$(Split-Path -Leaf (Split-Path -Parent $turnStatePath)):$(Split-Path -Leaf $turnStatePath)"
        $escapedPolicyKey = $policyKey.Replace("'", "''")
        $policyMetricsSql = "SELECT json_build_object('context_dimension',p.context_dimension,'update_count',p.total_updates,'actions',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor'),'resolved',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.status='COMPLETED' AND COALESCE((a.payload->>'outcome_observed')::boolean,false)),'no_reward',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.status='COMPLETED' AND NOT COALESCE((a.payload->>'outcome_observed')::boolean,false)),'pending',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.status='PENDING'),'agency',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.payload#>>'{explicit_learner_agency,triggered}'='true'),'generic',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.payload->>'selected_arm'='generic'),'probing',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.payload->>'selected_arm'='probing'),'focus',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.payload->>'selected_arm'='focus'),'telling',(SELECT COUNT(*) FROM research.turn_actions a WHERE a.payload->>'observation_origin'='live_tutor' AND a.payload->>'selected_arm'='telling')) FROM research.policy_states p WHERE p.policy_key='$escapedPolicyKey'"
        Push-Location $script:AdaptMathMemoryRoot
        try {
            $policyMetricsLine = @(& docker compose exec -T postgres psql -U memory_app `
                -d adaptmath_memory_integration -X -At -c $policyMetricsSql 2>$null) | Select-Object -First 1
        }
        finally { Pop-Location }
        if (-not [string]::IsNullOrWhiteSpace($policyMetricsLine)) {
            try {
                $policyMetrics = $policyMetricsLine | ConvertFrom-Json
                $turnContextDimension = [int]$policyMetrics.context_dimension
                $turnPosteriorUpdateCount = [int]$policyMetrics.update_count
                $turnRealLiveActionCount = [int]$policyMetrics.actions
                $turnRealLiveResolvedRewardCount = [int]$policyMetrics.resolved
                $turnRealLiveNoRewardCount = [int]$policyMetrics.no_reward
                $turnRealLivePendingCount = [int]$policyMetrics.pending
                $turnAgencyOverrideCount = [int]$policyMetrics.agency
                foreach ($arm in @('generic', 'probing', 'focus', 'telling')) {
                    $turnRealSelections[$arm] = [int]$policyMetrics.$arm
                }
                $turnMetricsFromPostgres = $true
            }
            catch {
                Write-Warning 'Authoritative Turn-LinTS PostgreSQL metrics could not be parsed.'
            }
        }
    }
    if ($turnMode -eq 'LIVE' -and $turnPosteriorUpdateCount -ne 'unknown') {
        $turnRealLiveUpdateCount = [Math]::Max(0, [int]$turnPosteriorUpdateCount - $turnSyntheticRecordCount)
        $turnEventsPath = Join-Path $turnEventRoot 'turn_events.jsonl'
        if (-not $turnMetricsFromPostgres -and (Test-Path -LiteralPath $turnEventsPath)) {
            foreach ($line in Get-Content -LiteralPath $turnEventsPath) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                try { $event = $line | ConvertFrom-Json } catch { continue }
                if ([string]$event.observation_origin -ne 'live_tutor') { continue }
                $turnRealLiveActionCount++
                if ($event.explicit_learner_agency.triggered) { $turnAgencyOverrideCount++ }
                elseif ($null -ne $event.selected_arm -and $turnRealSelections.Contains([string]$event.selected_arm)) {
                    $turnRealSelections[[string]$event.selected_arm]++
                }
                if ($event.outcome_observed) { $turnRealLiveResolvedRewardCount++ }
                else { $turnRealLiveNoRewardCount++ }
            }
        }
    }
    Write-Host "Turn-LinTS mode: $turnMode"
    Write-Host "behavior policy: $turnBehaviorPolicy"
    Write-Host "context blocks: $turnBlocks"
    Write-Host "diagnostic context blocks: $turnDiagnosticBlocks (logged only)"
    Write-Host "context dimension: $turnContextDimension"
    Write-Host "reward mode: $turnReward"
    Write-Host "action anchor: $turnAnchorMode"
    Write-Host "anchor gamma: $turnAnchorGamma"
    Write-Host "LIVE enabled: $(if ($turnMode -eq 'LIVE') { 'YES' } else { 'NO' })"
    Write-Host "posterior updates enabled: $(if ($turnPosteriorUpdatesEnabled) { 'YES' } else { 'NO' })"
    Write-Host "posterior update count: $turnPosteriorUpdateCount"
    Write-Host "synthetic initialization records: $turnSyntheticRecordCount"
    Write-Host "synthetic initialization effective weight: $turnSyntheticEffectiveWeight"
    Write-Host "real LIVE action count: $turnRealLiveActionCount"
    Write-Host "real LIVE resolved reward count: $turnRealLiveResolvedRewardCount"
    Write-Host "real LIVE no-reward count: $turnRealLiveNoRewardCount"
    Write-Host "real LIVE pending count: $turnRealLivePendingCount"
    Write-Host "real LIVE posterior updates: $turnRealLiveUpdateCount"
    Write-Host "generic real LIVE selections: $($turnRealSelections.generic)"
    Write-Host "probing real LIVE selections: $($turnRealSelections.probing)"
    Write-Host "focus real LIVE selections: $($turnRealSelections.focus)"
    Write-Host "telling real LIVE selections: $($turnRealSelections.telling)"
    Write-Host "agency override count: $turnAgencyOverrideCount"
    Write-Host "posterior authority: $(if ($turnMetricsFromPostgres) { 'PostgreSQL research.policy_states' } else { $turnStatePath })"
    Write-Host "event authority: $(if ($turnMetricsFromPostgres) { 'PostgreSQL research.turn_actions' } else { $turnEventRoot })"
}
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
    $authSchemaReady -and $memorySchemaReady -and $tutorPersistenceReady -and
    $studentModelReady -and $researchPersistenceReady -and
    $memoryHealth.Success -and $memoryReady.Success -and
    $tutorHealth.Success -and $tutorReady.Success -and $frontendUp
if (-not $allReady) { exit 1 }
