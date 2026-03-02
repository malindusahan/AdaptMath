# Move Selection component

This package contains the research implementation for selecting the next
pedagogical move. It includes the frozen MD selectors, contextual features,
MRB1 response-quality scoring, Turn-LinTS policy logic, outcome attribution,
and PostgreSQL research persistence adapters.

## Important locations

- `src/self_improvement/`: production selection and learning pipeline
- `models/`: versioned scientific model artifacts
- `tests/`: contract and scientific-behavior tests
- `experiments/`: offline evaluation programs
- `results/`: versioned research evidence and reports

The integrated Tutor imports this package through `PEDAGOGICAL_MOVE_REPO`.
Runtime policy snapshots and action evidence are authoritative in the
PostgreSQL `research` schema; frozen models and lineage manifests remain
immutable repository artifacts.

## Current live execution trace

The `-FinalSKLLive` path uses the frozen checkpoint
`models/frozen/md7_r2_tell_c1_epoch2` and the following current production
boundaries. These are the sources used by the live monitor; archived research
implementations are not involved.

| Value or stage | Current source |
| --- | --- |
| MD7 four-class probabilities | `src/self_improvement/md6_inference.py` — `FrozenMD6Inference.predict_probabilities` (the retained class name loads the current MD7 checkpoint) |
| S features | `src/self_improvement/turn_lints_context.py` — `build_turn_lints_context` from the four selector probabilities |
| K features | `apps/tutor/app/integrations/adaptive_component_coordinator.py` — `AdaptiveComponentCoordinator.run_tutor_turn` captures causal `mastery_before` and prior delta; `build_turn_lints_context` encodes them |
| L features | `AdaptiveComponentCoordinator.process_dialogue_turn` retains the preceding learner signals; `build_turn_lints_context` encodes the three signals and missingness |
| Complete 11-D S+K+L context | `src/self_improvement/turn_lints_context.py` — `build_turn_lints_context` |
| Thompson sampling and per-arm sampled scores | `src/self_improvement/turn_lints_policy.py` — `DirectTurnLinTS.select_arm` |
| Selected move and actual sample capture | `src/self_improvement/turn_lints_runtime.py` — `DirectTurnController.select_turn` |
| Tutor generation and MRB1 invocation | `src/self_improvement/turn_lints_runtime.py` — `DirectTurnLinTSPipeline.run_tutor_turn` |
| MRB1 diagnostic scores | `src/self_improvement/mrb1_inference.py` — `MRB1Inference.score_response`; recorded by `DirectTurnController.record_mrb1_scores` |
| Post-response BKT mastery before/after | `AdaptiveComponentCoordinator.process_dialogue_turn` and `TutorAssessmentStudentModelPipeline.process_dialogue_turn` |
| Raw delta and headroom-normalized reward | `src/self_improvement/turn_lints_reward.py` — `turn_rewards` |
| Selected-arm update and counts | `src/self_improvement/turn_lints_policy.py` — `DirectTurnLinTS.update_once` |
| Outcome attribution boundary | `src/self_improvement/turn_lints_runtime.py` — `DirectTurnController.complete_outcome` |
| PostgreSQL action/update/policy persistence | `src/self_improvement/postgres_repository.py` — `complete_action` |
| File fallback persistence | `src/self_improvement/turn_lints_policy.py` — `save_turn_lints_state` |

The local launcher makes PostgreSQL authoritative while the Tutor is running.
The policy is stored in `research.policy_states` with policy key
`turn_lints_live_skl_final_v1:policy_state.json`; action and update evidence is
stored in `research.turn_actions` and `research.policy_updates`. The runtime
JSON at `apps/tutor/runtime/turn_lints_live_skl_final_v1/policy_state.json` is
the validated bootstrap/file fallback. Both stores persist per-arm `A` and `b`,
arm update counts, overall update count, applied update IDs, and RNG state.

## Live move-selector monitor

The observer is opt-in and writes only already-generated decision/update values
to `_local_runtime_logs/move_selector_live.jsonl`. It does not call MD7, MRB1,
the sampler, BKT, reward calculation, policy update, or production persistence.

Enable diagnostics before starting AdaptMath (restart an already-running Tutor
after setting the flag):

```powershell
$env:ADAPTMATH_MOVE_SELECTOR_DEBUG = '1'
.\start-adaptmath-local.ps1 -FinalSKLLive
```

Then, in another PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\diagnostics\watch-move-selector-live.ps1
```

Compact view:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\diagnostics\watch-move-selector-live.ps1 -Compact
```

Show the current read-only PostgreSQL policy summary before watching:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\diagnostics\watch-move-selector-live.ps1 -Policy
```

Optional filters are `-StudentId <id>` and `-ThreadId <id>`. Student IDs are
deterministically pseudonymized before they enter the stream; the filter accepts
either the internal ID (hashed locally for comparison) or its emitted `psn_...`
value. No email, password, token, or conversation text is written. Add
`-SaveSession` to save the rendered terminal view under
`_local_runtime_logs/move-selector-sessions/`. The existing
`scripts/diagnostics/watch-lints-live.ps1` remains available for its compact
database-backed attempt/update table.
