# AdaptMath Repository Storage Cleanup Summary

Completion date: 2026-10-16  
Repository root: `D:\Research\Tutor Max`

## Outcome

The repository-wide infrastructure cleanup completed successfully. Four audited unused or obsolete Python environments, verified disposable caches, and the Git metadata/object directory were removed. The active integrated Tutor environment, frontend dependencies, source, notebooks, trained models, datasets, research results, configuration, tests, documentation, migrations, runtime state, and research evidence were retained.

The folder is now a normal local project directory rather than a Git working tree. Git was removed only after the targeted LFS and critical-file safety gate passed.

## Storage result

The before and after totals use the same boundary: project content is counted recursively, while the eight cleanup reports themselves are excluded. This avoids circular CSV sizing and makes the two totals directly comparable.

| Metric | Bytes | GiB |
|---|---:|---:|
| Original comparable project size | 17,260,721,211 | 16.0753 |
| Final comparable project size | 7,861,324,319 | 7.3214 |
| **Net space recovered** | **9,399,396,892** | **8.7539** |
| Gross bytes in deletion log | 9,403,443,416 | 8.7576 |
| Retained-content growth after baseline | 4,046,524 | 0.0038 |

The small gross-to-net difference is retained content that changed after the authoritative before snapshot, including build/runtime/audit activity. It is not an unlogged deletion.

The post-cleanup inventory contains 63,302 files and 7,707 directories (including the repository root), for 71,009 CSV rows.

## Deletion breakdown

| Category | Files logged | Bytes removed | GiB | Evidence |
|---|---:|---:|---:|---|
| Unused Python environments | 68,672 | 2,180,127,180 | 2.0304 | Three relocated/broken environments; active replacement or Docker runtime retained |
| Obsolete generated environment | 47,836 | 1,352,923,048 | 1.2600 | Exact 129-package duplicate of active Tutor environment; no runtime reference |
| Disposable caches | 438 | 4,646,939 | 0.0043 | Python/test/lint/notebook caches outside active dependency installations |
| `.git/lfs` | 30 | 4,420,771,206 | 4.1172 | All tracked working payloads verified hydrated before removal |
| `.git/objects` | 4,726 | 1,359,464,442 | 1.2661 | History explicitly not required; critical working files protected separately |
| Other `.git` metadata | 42 | 85,510,601 | 0.0796 | Refs, index, cursor, configuration, and other repository metadata |
| Other safe infrastructure outside the categories above | 0 | 0 | 0.0000 | Nothing else was approved under the narrowed continuation scope |
| **Gross total** | **121,744** | **9,403,443,416** | **8.7576** | Every byte-bearing path recorded in the deletion log |

The authoritative before inventory measured `.git` at 5,861,822,622 bytes. By deletion time it measured 5,865,746,249 bytes because the object area grew by 3,923,627 bytes during audit activity. `.git/lfs` and non-object metadata did not change. The deletion log records the actual sizes removed.

## Removed environments

| Path | Bytes | Result |
|---|---:|---|
| `apps\tutor\.venv` | 997,533,251 | Removed; broken relocated Python 3.13 environment |
| `apps\student-memory\.venv` | 1,171,528,960 | Removed; broken relocated Python 3.13 environment; Docker runtime retained |
| `packages\student-modeling\.venv` | 11,064,969 | Removed; broken relocated environment with no unique usable packages |
| `_local_generated\obsolete-adaptive-root-venv` | 1,352,923,048 | Removed; exact package duplicate of the active environment |

Retained active environment: `apps\tutor\.venv-integration` (Python 3.12.7). The startup script still points to this environment, its interpreter exists, and the running Tutor service continued returning HTTP 200 after cleanup.

Both active frontend `node_modules` trees were retained. The continuation instructions restricted cache removal to verified Python/test/lint/notebook caches outside active dependency folders, so npm cache trees were also left intact and are not claimed as recovered space.

## Git and LFS safety gate

The targeted pre-removal gate passed:

- `GIT_LFS_POINTER_AUDIT.csv`: 29/29 rows reported `PointerDetected = No`, `Hydrated = Yes`, `Status = HYDRATED_HASH_MATCH`, and `SHA256 = OID`.
- The 29 tracked working payloads were checked directly for existence, expected size, and the LFS pointer header; 29/29 passed.
- `PRE_GIT_REMOVAL_CRITICAL_FILE_MANIFEST.csv`: 133/133 protected files matched their recorded path, size, and SHA-256 immediately before deletion.
- All 133 protected files still existed at the expected sizes after deletion.
- No unresolved LFS working-tree payload was found by the targeted check.
- `.git` is absent after cleanup; no `git clean`, `git reset`, `git checkout`, `git gc`, or `git init` was run.

## Post-cleanup verification

### Move Selection

- Test suite: **17 passed**.
- Frozen MD6 loaded and produced a valid four-probability vector.
- Active MD7-R2-TELL-C1 epoch 2 loaded and produced a valid four-probability vector.
- Frozen MRB1 loaded and returned four valid continuous task scores.
- S+K+L context dimension: **11**.
- Direct Turn-LinTS selection, headroom-normalized reward, one posterior update, state save, and state reload all passed using isolated temporary persistence.

### Tutor

- `app.local_runtime:app` imported; FastAPI reported 11 routes.
- Live `/health` returned HTTP 200.
- Current and legacy complexity model artifacts both loaded; the current model produced a finite in-range prediction.
- Test comparison: **306 passed, 3 failed, 23 warnings, 8 subtests passed**, exactly matching the pre-cleanup baseline failure set.

### Student Modeling

- BKT artifact loaded with 95 skills and produced a mastery prediction.
- Curriculum artifact loaded and validated with 11 curriculum skills.
- Reasoning, uncertainty, and clarification detector artifacts all deserialized.
- Pytest collection reproduced the same three pre-cleanup stale-import/API errors; no new error appeared.

### Student Personalization Memory

- Docker backend and PostgreSQL containers remained healthy.
- `/health` and `/ready` both returned HTTP 200.
- FastAPI app imported in the backend container.
- The frozen learning-state model and imputer loaded with the expected 12-feature contract.
- Fine-tuned MiniLM-v2 topic extraction assets loaded and performed inference.
- Container suite comparison: **600 passed, 2 failed, 1 skipped**, exactly matching the pre-cleanup baseline failure set.

### Frontend

- Retained dependencies remained intact.
- Vitest: **2 files passed, 5 tests passed**.
- TypeScript/Vite production build completed successfully (only the existing chunk-size advisory was emitted).

## Pre-existing issues not caused by cleanup

1. Tutor has three research-state/data-count assertions whose stored runtime evidence has advanced beyond fixed test expectations: update count 548 versus 500, real live updates 48 versus 0, and reconstructed contexts 174 versus 98.
2. Student Modeling collection has three stale API imports: `extract_concepts`; `load_graph`/`update_graph`/`detect_regressions`/`save_graph`; and `create_cold_start_profile`/`upgrade_profile`/`assign_student_id`.
3. Student Memory has two container-only path assumptions that access `Path(__file__).resolve().parents[3]` from `/app/tests`, raising `IndexError`.
4. Student Modeling's preserved `requirements.txt` contains an apparent concatenated entry (`scikit-learn<1.5pytest`); it was reported but not edited.
5. Loading older detector artifacts under the retained newer scikit-learn emits version-compatibility warnings, as it did before cleanup.

No application or research code was changed to hide or repair these unrelated baseline issues.

## Protected-content declaration

| Content type | Deleted? |
|---|---|
| Source code | **NO** |
| Notebooks | **NO** |
| Trained models or checkpoints | **NO** |
| Datasets | **NO** |
| Research results, metrics, evaluation, or ablation evidence | **NO** |
| Configuration, dependency definitions, tests, documentation, curriculum, SQL, or migrations | **NO** |

## Evidence files

- `REPO_CLEANUP_INVENTORY_BEFORE.csv` - authoritative comparable before snapshot.
- `REPO_CLEANUP_PLAN.md` - decisions, protected scope, and largest-item audit.
- `PYTHON_ENVIRONMENT_AUDIT.md` - environment classification and reproducibility evidence.
- `GIT_LFS_POINTER_AUDIT.csv` - 29 tracked LFS payload results.
- `PRE_GIT_REMOVAL_CRITICAL_FILE_MANIFEST.csv` - 133 protected paths, sizes, hashes, and readability checks.
- `REPO_CLEANUP_DELETION_LOG.csv` - 121,744 removed file paths; every row marked runtime-verified after deletion.
- `REPO_CLEANUP_INVENTORY_AFTER.csv` - authoritative comparable after snapshot.
- `REPO_STORAGE_CLEANUP_SUMMARY.md` - this final reconciliation.

## Final status

- New runtime/test failures caused by cleanup: **NONE**.
- Active environment retained: `apps\tutor\.venv-integration`.
- Git removed: **YES**.
- LFS tracked files verified: **29/29**.
- Critical manifest files verified: **133/133**.
- All 121,744 logged deletion rows reconciled and post-cleanup verified.
