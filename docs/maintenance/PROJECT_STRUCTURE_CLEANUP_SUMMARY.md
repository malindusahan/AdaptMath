# AdaptMath project structure cleanup summary

Date: 2026-10-16

Scope: final structure and housekeeping cleanup only. No algorithm, application source, research result, dataset, model, notebook, test, requirement, current policy state, or current database state was deleted or changed.

## Moved files

### Root storage-cleanup evidence

Moved to `docs/maintenance/storage-cleanup-2026-10-16/`:

- `GIT_LFS_POINTER_AUDIT.csv`
- `PRE_GIT_REMOVAL_CRITICAL_FILE_MANIFEST.csv`
- `REPO_CLEANUP_DELETION_LOG.csv`
- `REPO_STORAGE_CLEANUP_SUMMARY.md`

### Diagnostic scripts

Moved to `scripts/diagnostics/`:

- `inspect-demo-student.ps1`
- `watch-bkt-live.ps1`
- `watch-lints-live.ps1`

Each moved script now resolves the workspace root through `..\..`. All three scripts find the root common launcher and `apps/student-memory`, and all parse with zero PowerShell syntax errors.

### Move-selection documentation

- `MD0_TO_MD13_RESEARCH_MAP.md` -> `packages/move-selection/docs/research/`
- `FINAL_LINTS_RESEARCH_MAP.md` -> `packages/move-selection/docs/research/`
- `MOVE_SELECTOR_RESEARCH_ARTEFACT_MAP.md` -> `packages/move-selection/docs/research/`
- `CURRENT_MOVE_SELECTOR_RUNTIME_MANIFEST.md` -> `packages/move-selection/docs/runtime/`
- `MOVE_SELECTION_FINAL_CLEANUP_SUMMARY.md` -> `packages/move-selection/docs/maintenance/`
- `FINAL_OLD_RESEARCH_DELETION_LOG.csv` -> `packages/move-selection/docs/maintenance/`

The completed deletion-list reference in `FINAL_LINTS_RESEARCH_MAP.md` now points to the final log location.

### Historical archives

- `_forensics/20260826_same_attempt_failure_pre_fix/` -> `docs/archive/forensics/20260826_same_attempt_failure_pre_fix/`
- `_integration_backups/20260826_four_component_validation/` -> `docs/archive/integration-backups/20260826_four_component_validation/`
- PostgreSQL unification README, consistent SQLite rollback set, and PostgreSQL pre-unification dump -> `docs/archive/integration-backups/postgres_unification_20260830_183916/`

The archived forensic set contains 14 files and 1,951,587 bytes. The selected integration archive contains 20 files and 17,944,081 bytes.

## Deleted files

Explicitly authorized temporary or obsolete housekeeping files:

- `REPO_CLEANUP_INVENTORY_BEFORE.csv`
- `REPO_CLEANUP_INVENTORY_AFTER.csv`
- `REPO_CLEANUP_PLAN.md`
- `PYTHON_ENVIRONMENT_AUDIT.md`
- `.commit_log`
- `generate_history.sh`
- `packages/move-selection/MOVE_SELECTION_OLD_RESEARCH_DELETION_CANDIDATES.csv`
- `_local_runtime_logs/chrome-ui-test.stderr.log`
- `_local_runtime_logs/chrome-ui-test.stdout.log`

Superseded integration backups removed after preserving the useful rollback/evidence subset:

- `20260825_adaptive_tutor_integration`
- `20260826_gemini_provider_migration`
- `20260826_local_demo_hardening`
- `20260826_memory_adaptive_integration`
- `20260826_memory_validation_fixes`
- duplicated/superseded `configuration` and `filesystem` copies from `postgres_unification_20260830_183916`
- the now-empty root `_integration_backups/` and `_forensics/` containers

Hash comparisons confirmed representative large backup data/model/evidence files had byte-identical current equivalents, including BKT training data, detector weights, turn outcomes, and student-model state. Consistent database rollback artifacts were archived rather than deleted. Deleted material is not recoverable from this workspace unless an external backup exists.

## Retained files

- Root `README.md`, `.gitignore`, and `.gitattributes`
- Root runtime launchers: `adaptmath-local-common.ps1`, `start-adaptmath-local.ps1`, `status-adaptmath-local.ps1`, and `stop-adaptmath-local.ps1`
- All application/package source, tests, requirements, models, datasets, notebooks, results, and research evidence
- All existing non-housekeeping move-selection documentation under `packages/move-selection/docs/`

## Retained runtime directories

- `_local_runtime_logs/`: retained because active Tutor, frontend, and supporting processes have current log files open. Only old August Chrome UI test logs were removed.
- `_local_runtime_state/`: retained unchanged. `adaptmath-local-common.ps1` reads/writes its launcher metadata, and Tutor validation tooling reads `adaptmath-local.json`.
- `apps/tutor/runtime/`: retained unchanged, including current policy and event authority.

## FORENSICS DECISION

Classification: **USEFUL_ARCHIVE**.

The set is a unique pre-fix same-attempt failure snapshot containing checkpoint databases, WAL/SHM files, logs, launcher metadata, and read-only inspection scripts dated 2026-08-26. It is not referenced by current runtime code, but it preserves useful failure evidence; therefore it was archived rather than deleted.

## INTEGRATION BACKUPS DECISION

Decision: **ARCHIVE USEFUL EVIDENCE; DELETE SUPERSEDED COPIES**.

The consistent database rollback set, PostgreSQL dump, migration README, and unique four-component validation snapshots were archived. Dated source/configuration trees and raw duplicated filesystem copies were removed after current equivalents and active authorities were confirmed. The persistence-unification documentation was updated to the archived dump path.

## COMMIT HISTORY FILE DECISION

`.commit_log` and `generate_history.sh` were used only to construct an artificial dated Git history. Nothing else referenced them, `.git` is absent, and neither participates in runtime or research reproduction. Both were deleted. Git configuration files were retained.

## SCRIPT ORGANIZATION DECISION

The four primary launcher/common scripts remain at the repository root. Tests, the root README, documentation, and operational verification scripts intentionally reference these exact root paths, while the launchers use `$PSScriptRoot` as the workspace root. Moving them would require prohibited or unnecessary changes to tests/runtime tooling.

The three unreferenced diagnostic scripts moved to `scripts/diagnostics/` with path resolution adjusted and validated.

## MOVE-SELECTION DOCUMENTATION ORGANIZATION

The move-selection root now contains only its core directories plus `.gitignore`, `README.md`, and `requirements.txt`. Historical/research maps are under `docs/research/`, the current runtime manifest is under `docs/runtime/`, and completed cleanup evidence is under `docs/maintenance/`.

## Verification

- Tutor selector integration import: PASS
- Normal selector mode: `ordinary-md7-r2-tell-c1-v1`
- Active selector path: `packages/move-selection/models/frozen/md7_r2_tell_c1_epoch2`
- PowerShell parse/reference validation: PASS
- Move-selection tests: **17 passed, 0 failed**
- Current MD7, MD6 rollback, MD7-R1 rollback, and MRB1 checkpoints: retained
- LinTS policy/reward implementation: retained
- Final dataset: 713 turns, 31 learners, 92 attempts, 22 skills
- S+K+L context/reward/headroom/grouped-CV/preflight evidence: retained
- Bootstrap replicates: 5,000

The live status command executed after restructuring and resolved the active MD7/Turn-LinTS configuration. It returned a nonzero status because PostgreSQL/Memory service readiness was down at verification time; Tutor and the frontend were running. This is current service availability rather than a path or import failure introduced by housekeeping.

## Final root-level structure

```text
Tutor Max/
|-- .qodo/
|-- .vscode/
|-- _local_runtime_logs/
|-- _local_runtime_state/
|-- apps/
|-- docs/
|-- infrastructure/
|-- packages/
|-- research/
|-- scripts/
|-- tests/
|-- .gitattributes
|-- .gitignore
|-- adaptmath-local-common.ps1
|-- README.md
|-- start-adaptmath-local.ps1
|-- status-adaptmath-local.ps1
`-- stop-adaptmath-local.ps1
```
