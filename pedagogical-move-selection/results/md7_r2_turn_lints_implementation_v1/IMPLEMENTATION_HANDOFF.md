# MD7-R2-TELL-C1 / Turn-LinTS v1 engineering handoff

## Promotion

- Frozen selector: `models/frozen/md7_r2_tell_c1_epoch2/`
- Weight SHA-256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- Fixed labels: `generic`, `probing`, `focus`, `telling`
- Previous MD7-R1 remains unchanged at `models/candidates/md7r1_epoch3/` and is
  available through selector mode `md7r1-rollback-v1` / launcher switch
  `-MD7R1Rollback`. MD6 rollback and manual controlled-live remain available.

## Implementation

New source files:

- `src/self_improvement/turn_lints_context.py`
- `src/self_improvement/turn_lints_reward.py`
- `src/self_improvement/turn_lints_policy.py`
- `src/self_improvement/turn_lints_runtime.py`

Integration/configuration changes are in the Tutor coordinator, selector mode,
Tutor memory adapter, artifact preflight, local start/status scripts, and the
focused `tests/test_turn_lints_v1.py` contract test.

Direct policy arms are exactly `generic`, `probing`, `focus`, and `telling`.
There is one independent `A`/`b` posterior per arm. No action eligibility API,
tau gate, MD7 probability-gap threshold, bias/baseline arm, scaffold gate, or
attempt-level delayed-credit path is present in this lineage.

Modes:

- OFF (default): frozen MD7-R2-TELL-C1 selects directly; no Turn-LinTS sampling
  or update.
- SHADOW: MD7 still controls; a hypothetical direct arm and exact context are
  logged; no update is persisted.
- LIVE: all four direct arms participate; the selected arm controls the Tutor
  and receives one immediate update after its uniquely linked dialogue BKT
  observation. LIVE is never selected automatically.

Configuration uses the existing environment convention:

- `ADAPTIVE_TURN_LINTS_MODE` (default `OFF`)
- `ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS` (default `S`)
- `ADAPTIVE_TURN_LINTS_REWARD_MODE` (default `raw_delta`)
- `ADAPTIVE_TURN_LINTS_RIDGE_LAMBDA` (default `1.0`)
- `ADAPTIVE_TURN_LINTS_EXPLORATION_SCALE` (default `0.20`)
- `ADAPTIVE_TURN_LINTS_SEED` (default `42`)
- `ADAPTIVE_TURN_LINTS_STATE_PATH`
- `ADAPTIVE_TURN_LINTS_EVENT_ROOT`

The defaults are implementation values, not scientifically selected values.

## Context and timing

All implemented blocks are available for presets/arbitrary subsets:

- S: same-state MD7 probabilities in fixed move order.
- K: authoritative action-time `mastery_before`, previous resolved dialogue
  mastery delta, and explicit previous-delta missingness.
- L: previous learner-response reasoning, uncertainty, and clarification
  probabilities from Student Modeling, plus shared missingness. These frozen
  detectors execute while resolving the preceding learner response, so their
  result is available before the next Tutor action. No confusion,
  explanation-request, or direct-help field exists in the inspected runtime.
- H: previous move one-hot, missingness, and prior Tutor-turn count.
- Q: the immediately previous response's four MRB1 expected scores and shared
  missingness.

The six supported named presets are S, S+K, S+L, S+K+L, S+K+L+H, and
S+K+L+H+Q. The context record exposes schema/version, enabled and available
blocks, exact ordered names, numeric vector, dimension, and missingness.

Current-response MRB1 is computed only after action selection. Its four scores
are logged as the action's response-quality outcome, never included in that
action's context or reward, and only become eligible for the next context.
The MRB1 arithmetic mean is not included.

## Reward and persistence

Implemented reward modes:

- `raw_delta`: `mastery_after - mastery_before`
- `headroom_normalized`: symmetric available-headroom normalization in
  `[-1, 1]`

The default is `raw_delta`; neither mode is claimed superior. MRB1 is not part
of either scalar reward.

Default real lineage:

- Root: `adaptive-math-tutor/backend/runtime/adaptive_turn_lints_v1_real/`
- State: `policy_state.json`
- Derived action/outcome log: `turn_events.jsonl`
- Pending action identities: `pending_actions/`
- State schema: `adaptmath_turn_lints_state_v1`
- Algorithm: `direct_disjoint_turn_lints_v1`

State stores selector version/hash, direct arms, context schema/blocks/names and
dimension, reward mode, posterior hyperparameters, creation time, update
counts, and applied action identities. Context, reward, selector, arm,
hyperparameter, data-mode, or schema mismatch fails closed. No legacy
posterior migration exists.

Every LIVE update requires one real selected action, the matching resolver
identity, `should_update=true`, and valid turn masteries. Applied action IDs are
stored atomically with the posterior for durable retry idempotency. Missing,
censored, processing-failed, and resolved-no-update outcomes do not fabricate a
zero reward. Formal three-question assessment can still update BKT but never
calls the Turn-LinTS action update path. The old attempt-completion distribution
is absent from this lineage.

## Files modified

- `pedagogical-move-selection/models/frozen/SHA256SUMS.txt`
- `pedagogical-move-selection/models/frozen/md7_r2_tell_c1_epoch2/*`
- the four new `turn_lints_*` source files listed above
- `adaptive-math-tutor/backend/app/integrations/adaptive_component_coordinator.py`
- `adaptive-math-tutor/backend/app/integrations/adaptive_selector_mode.py`
- `adaptive-math-tutor/backend/app/integrations/tutor_state_memory_adapter.py`
- `adaptive-math-tutor/backend/scripts/check_local_runtime_artifacts.py`
- `adaptive-math-tutor/backend/tests/test_turn_lints_v1.py`
- `start-adaptmath-local.ps1`
- `status-adaptmath-local.ps1`

## Recommended manual terminal audit

Run from the workspace root. These commands were intentionally not run as part
of the implementation handoff:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'pedagogical-move-selection\models\frozen\md7_r2_tell_c1_epoch2\model.safetensors'

& 'adaptive-math-tutor\backend\.venv-integration\Scripts\python.exe' -m pytest -q 'adaptive-math-tutor\backend\tests\test_turn_lints_v1.py'

$env:ADAPTIVE_TURN_LINTS_MODE='SHADOW'
$env:ADAPTIVE_TURN_LINTS_CONTEXT_BLOCKS='S+K+L+H+Q'
$env:ADAPTIVE_TURN_LINTS_REWARD_MODE='raw_delta'
$env:ADAPTIVE_TURN_LINTS_STATE_PATH=(Join-Path (Resolve-Path 'adaptive-math-tutor\backend\runtime') 'turn_lints_shadow_full_raw_v1\policy_state.json')
$env:ADAPTIVE_TURN_LINTS_EVENT_ROOT=(Join-Path (Resolve-Path 'adaptive-math-tutor\backend\runtime') 'turn_lints_shadow_full_raw_v1')
.\start-adaptmath-local.ps1

.\status-adaptmath-local.ps1
```

For a different context or reward audit, use a new state/event directory. Do
not point a changed schema or reward mode at an existing posterior.
