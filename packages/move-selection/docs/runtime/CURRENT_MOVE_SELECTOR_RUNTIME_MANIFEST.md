# Current Move-Selector Runtime Manifest

Audit date: 2026-10-16  
Component root: `packages\move-selection`

## Current integrated configuration

The Tutor currently imports the move-selector component by adding `packages\move-selection` to Python's import path (overridable with `PEDAGOGICAL_MOVE_REPO`). The running Python executable is `apps\tutor\.venv-integration\Scripts\python.exe`; the component-local `.venv` is not used.

The observed current configuration is:

- Base selector mode: `ordinary-md7-r2-tell-c1-v1`
- Selector: MD7-R2-TELL-C1 epoch 2
- Frozen selector path: `models\frozen\md7_r2_tell_c1_epoch2`
- Selector model SHA-256: `ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5`
- Policy: `direct_disjoint_turn_lints_v1`
- Turn-LinTS mode: LIVE
- Context: S+K+L, 11 dimensions (S=4, K=3, L=4)
- Reward: `headroom_normalized`
- Action anchor: `none`, gamma `0`
- Critic: frozen MRB1

## Required move-selection-owned artefacts

| Area | Name/path | Size | LoadedBy | RequiredAtStartup | RequiredDuringTutoring |
|---|---|---:|---|---|---|
| Active selector | `models\frozen\md7_r2_tell_c1_epoch2\config.json` | 926 B | `FrozenMD6Inference` via Tutor coordinator | Yes | Yes |
| Active selector | `models\frozen\md7_r2_tell_c1_epoch2\model.safetensors` | 328,498,416 B | Transformers sequence classifier | Yes | Yes |
| Active selector | `models\frozen\md7_r2_tell_c1_epoch2\tokenizer.json` | 3,558,737 B | local tokenizer loader | Yes | Yes |
| Active selector | `models\frozen\md7_r2_tell_c1_epoch2\tokenizer_config.json` | 358 B | local tokenizer loader | Yes | Yes |
| MD6 rollback/research | `models\frozen\md6` | 332,058,389 B | `FrozenMD6Inference` when MD6 mode/smoke is selected | Conditional | Conditional |
| Selector loader | `src\self_improvement\md6_inference.py` | 6,037 B | Tutor coordinator | Yes | Yes |
| Context Builder | `src\self_improvement\turn_lints_context.py` | 8,782 B | Tutor coordinator/runtime | Yes | Yes |
| Shared move ordering/context helpers | `src\self_improvement\context_builder.py`; `turn_context_builder.py` | 18,646 B | context and inference modules | Yes | Yes |
| Learner State | S+K+L pre-action fields defined by `turn_lints_context.py` | Included above | Tutor state-memory adapter supplies values read-only | Yes | Yes |
| Learner Signals | `LEARNER_SIGNAL_NAMES` and validation in `turn_lints_context.py` | Included above | Tutor coordinator/state-memory adapter | Yes | Yes |
| LinTS | `src\self_improvement\turn_lints_policy.py` | 20,595 B | Tutor coordinator/runtime | Yes | Yes |
| Runtime orchestration | `src\self_improvement\turn_lints_runtime.py` | 37,465 B | Tutor coordinator | Yes | Yes |
| Reward | `src\self_improvement\turn_lints_reward.py` | 1,780 B | Tutor coordinator/runtime | Yes | At outcome/update time |
| MRB1 loader | `src\self_improvement\mrb1_inference.py` | 7,638 B | Tutor coordinator | Yes | Yes |
| MRB1 config | `models\frozen\mrb1\config.json` | 907 B | frozen MRB1 loader | Yes | Yes |
| MRB1 weights | `models\frozen\mrb1\model.safetensors` | 496,281,464 B | custom multitask model | Yes | Yes |
| MRB1 class | `models\frozen\mrb1\mrb1_multitask_model.py` | 2,108 B | dynamic local module loader | Yes | Yes |
| MRB1 tokenizer | `models\frozen\mrb1\tokenizer.json`; `tokenizer_config.json` | 3,559,001 B | local tokenizer loader | Yes | Yes |
| Persistence | `src\self_improvement\postgres_repository.py` | 24,722 B | Turn-LinTS policy/runtime | Yes | Yes |
| Local persistence fallback | state I/O in `src\self_improvement\turn_lints_policy.py` | Included above | `save_turn_lints_state` / `load_turn_lints_state` | Conditional | Conditional |
| Live lineage | `src\self_improvement\turn_lints_live_lineage.py` | 4,744 B | Tutor coordinator in LIVE mode | Yes | At activation/update |
| Activation evidence | `results\turn_lints_live_skl_synthetic_warmstart_v1\DECISION_EVIDENCE.md` and posterior evidence package | 4,463,168 B directory | lineage validation/audit | Yes for lineage | No direct inference |
| Context freeze/config evidence | `results\turn_lints_final_architecture_skl_freeze_v1` | 698,892 B | lineage/audit | Yes for lineage | No direct inference |
| Package initialization | `src\__init__.py`; `src\self_improvement\__init__.py` | 20 B | Python import machinery | Yes | Yes |
| Tests | `tests\test_student_model_v3_bridge.py`; `tests\manual_smoke_student_model_v3_bridge.py` | 44,730 B | test runner | No | No |

## Read-only integration artefacts outside move-selection

These files/directories are required by the integrated system but are owned by Tutor, so they were inspected only and are explicitly outside this cleanup:

| Purpose | Path | Ownership/action |
|---|---|---|
| Move-selector import/configuration and coordination | `apps\tutor\app\integrations\adaptive_component_coordinator.py` | Tutor; read-only, unchanged |
| Selector-mode definition | `apps\tutor\app\integrations\adaptive_selector_mode.py` | Tutor; read-only, unchanged |
| Learner pre-action state bridge | `apps\tutor\app\integrations\tutor_state_memory_adapter.py` | Tutor; read-only, unchanged |
| Live local policy/lineage fallback | `apps\tutor\runtime\turn_lints_live_skl_final_v1` | Tutor; read-only, unchanged |
| Current integration environment | `apps\tutor\.venv-integration` | Tutor; read-only, unchanged |
| Authoritative production persistence | PostgreSQL research schema when enabled | External runtime state; unchanged |

## Retained conditional/rollback paths

- `models\frozen\md6` remains the complete final MD6 runtime package and is load-tested after cleanup.
- `models\candidates\md7r1_epoch3` remains available for the supported rollback/manual selector mode.
- `models\candidates\md7_r2_tell_c1\epoch2` remains as research lineage even though its weights are byte-identical to the active frozen package.
- All unknown or research-significant later checkpoints remain untouched.

## Environment reproducibility

- Declared dependencies: `requirements.txt`.
- Exact removed component-environment capture: `requirements-venv-freeze-before-cleanup.txt` (52 lines, produced by `.venv\Scripts\python.exe -m pip freeze --all`).
- Integrated runtime interpreter: `apps\tutor\.venv-integration\Scripts\python.exe` (outside scope, unchanged).
