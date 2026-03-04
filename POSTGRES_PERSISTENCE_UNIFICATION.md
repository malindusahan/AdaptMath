# AdaptMath PostgreSQL persistence unification

This document records the storage inventory discovered from the running code
and configuration on 2026-08-30. It distinguishes active runtime authority from
rollback copies, scientific model artifacts, derived diagnostics, and test data.

## Ownership and migration version

One PostgreSQL 16 server and database are shared. Alembic in
`student-memory-personalization` is the sole DDL owner. Revisions
`0012_postgres_unification` and `0013_policy_state_ordered_json` reproducibly
create the five-schema layout and preserve frozen policy map ordering from zero.
Runtime services verify the head migration; they do not create tables.

| Schema | Owner | Authoritative structured state |
|---|---|---|
| `auth` | authentication repository | hashed, expiring, revocable sessions; Phase-A `user_accounts` view |
| `student_memory` | Memory repositories | accounts, students, sessions, receipts, interactions, misconceptions, projections |
| `tutor` | official LangGraph PostgreSQL saver | checkpoints, blobs, pending writes, thread metadata |
| `student_model` | BKT repository adapter | observations, immutable effective priors, mastery, resolved-event ledger |
| `research` | research repository | decisions, contexts, MRB1, policy snapshots/updates, pending actions, experience evidence |

`student_memory.user_accounts` remains the physical account table during Phase
A because its existing student foreign key and ORM mappings are known-good.
`auth.user_accounts` is a compatibility/ownership view. Password hashes, IDs,
roles, and timestamps are not rewritten.

## Legacy persistence inventory

| Component | Source and format | Authority before cutover | Schema/version and contents | Read/write, transaction and concurrency behavior | Recovery |
|---|---|---|---|---|---|
| Tutor | `adaptive-math-tutor/backend/runtime/adaptive_live_real_v1/checkpoints.sqlite3` (+ WAL/SHM), SQLite | LangGraph graph state for the active runtime | Official LangGraph SQLite saver tables; 858 logical checkpoints, 4,545 pending writes, 35 threads at backup | `app/core/persistence.py`; one SQLite connection, WAL, busy timeout; saver commits graph checkpoints and writes | reopen by `thread_id`; process-local controller state still deliberately fails closed with HTTP 409 when it cannot be reconstructed |
| Tutor fallback | `adaptive-math-tutor/backend/runtime/adaptmath_checkpoints.sqlite3`, SQLite | older/default rollback store, not selected by active launcher | same official saver family | same adapter in `TUTOR_PERSISTENCE=sqlite` mode | retained unchanged |
| Student model | `adaptive-math-tutor/backend/runtime/adaptive_live_real_v1/student_model/data/meta_agent.db`, SQLite | active BKT state | `PRAGMA user_version=2`; students, attempts, resolved events, effective priors, mastery, sessions | `student-modeling/db/database.py` and `core/knowledge_graph.py`; per-operation transactions; stable resolver IDs; chronological `created_at, attempt_id` ordering | exact state can be reopened; no historical recomputation required for migration |
| Student-model fallback | `student-modeling/data/meta_agent.db`, SQLite | repository-local rollback/reference state | same logical schema/version family | same repository in SQLite mode | retained unchanged |
| Active policy | `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/policy_state.json`, JSON | current direct Turn-LinTS LIVE posterior | complete versioned payload including A/b state, arm/update counts, applied IDs, hyperparameters, RNG state; 548 updates and 11-D current S+K+L context | `turn_lints_policy.py` / `state_io.py`; atomic file replacement; in-process lock around controller mutation | reload exact payload; audited lineage manifest remains immutable filesystem configuration |
| Legacy C3 policy | runtime `adaptive_demo*` / MD6 and MD7 `policy_state.json` files, JSON | rollback/research lineages | canonical legacy C3 has exactly 9 features | policy state adapters; atomic replacement | retained unchanged and importable by stable policy key |
| Active turn evidence | `adaptive-math-tutor/backend/runtime/turn_lints_live_skl_final_v1/turn_events.jsonl`, JSONL | direct selector turn ledger | 59 completed actions at backup; complete structured records | append under process/file locks; action IDs are idempotency boundary | replay/import is idempotent; original retained |
| Pending research | runtime `*/pending_actions/*.json`, JSON | durable staged action ledgers | 9 files across lineages at backup | write-temp/replace in legacy mode | migrated to `research.turn_actions` as `PENDING`; legacy files retained |
| Legacy attempt evidence | `adaptive-math-tutor/backend/runtime/adaptive_demo_md7r1_real_v1/attempts.jsonl`, JSONL | MD7 rollback experiment evidence | 12 records at backup | append-only experience logger with stable record content | idempotent import by deterministic source identity; original retained |
| Passive research dataset | `pedagogical-move-selection/results/md_self_improvement_turn_data_v1/*.jsonl`, JSONL | derived/offline research dataset, not live mastery authority | turn, assessment, attempt and BKT-activity projections plus manifest | append-only collector; pseudonymized student identity | new runtime writes use `research.experience_records`; existing dataset remains a reproducibility artifact |
| Authentication accounts | `student_memory.user_accounts`, PostgreSQL | account authority | existing Alembic schema and FKs | SQLAlchemy Unit of Work transactions | preserved in place; exposed as `auth.user_accounts` view |
| Authentication sessions | process `_ACTIVE_SESSIONS` dictionary | bearer-session authority before cutover | `mem_sess_*` raw tokens mapped to identity; no expiry | process-local mutation only | existing in-memory tokens cannot be fabricated/migrated and expire at cutover; new tokens store SHA-256 only in PostgreSQL |
| Memory | `student_memory` in `adaptmath_memory_integration`, PostgreSQL | Memory authority | Alembic revisions 0001-0011 before unification; 15 existing tables | bounded SQLAlchemy pool, Unit of Work transactions, receipt/interaction idempotency | PostgreSQL dump plus existing service recovery behavior |

JSON lineage manifests, frozen selector metadata, BKT parameter JSON, skill
metadata, `.safetensors`, `.joblib`, and other trained artifacts are immutable
configuration/model artifacts rather than runtime learner state. They remain on
the filesystem/Git LFS. Generated analysis outputs and test fixtures likewise
remain files and are not made runtime authorities.

The current promoted direct Turn-LinTS policy uses its already-frozen 11-D
S+K+L schema. That is not legacy C3. The legacy C3 builder and `research.c3_contexts`
remain strictly 9-D; the migration does not conflate or alter either schema.

## Transaction and failure boundaries

- Auth session creation/revocation shares the account Unit of Work and fails
  closed if PostgreSQL is unavailable. Raw bearer tokens are never persisted.
- The official LangGraph PostgreSQL saver preserves checkpoint IDs, parents,
  blobs, pending writes and resume semantics. `tutor.threads` is diagnostic
  metadata; the checkpoint remains authoritative.
- BKT uses stable resolver IDs and a PostgreSQL transaction-scoped advisory
  lock for each sorted `(student_id, skill)` stream. It re-reads the ordered
  observations after locking and executes the unchanged predictor exactly once.
- Research uses stable action/update/event IDs and unique constraints. A direct
  policy update and its full policy snapshot commit in one research transaction.
  Passive action completion and its experience projection also commit together.
- There is intentionally no giant cross-service transaction. BKT remains the
  mastery authority; research mastery/reward values are immutable evidence.
- No conversation archive was added. Tutor checkpoint state remains the only
  conversation authority, so old dialogue is not injected into new sessions.
- The optional Memory outbox was not added; that would expand migration risk.

## Configuration and rollback modes

The launcher derives one password-bearing URL from the ignored Memory `.env`
without printing it and exports `ADAPTMATH_DATABASE_URL`. Schema names are
fixed and validated as `auth`, `student_memory`, `tutor`, `student_model`, and
`research`. Bounded service-local pools are used.

Rollback switches:

```text
AUTH_SESSION_PERSISTENCE=memory
TUTOR_PERSISTENCE=sqlite
STUDENT_MODEL_PERSISTENCE=sqlite
RESEARCH_PERSISTENCE=file
```

Rollback is intentionally one-way for a controlled test period: PostgreSQL-only
events created after cutover are not reverse-migrated into the legacy files.

## Backup and restore

Pre-mutation backup:

```text
_integration_backups/postgres_unification_20260830_183916
```

The directory contains a raw filesystem copy, consistent SQLite snapshots,
configuration/Alembic copies, and a PostgreSQL custom-format dump. Restore only
with all AdaptMath writers stopped.

To restore SQLite rollback stores, copy the corresponding file from
`sqlite_consistent` back to its documented source path. Copying is preferable
to restoring the raw WAL/SHM set.

To restore the pre-unification PostgreSQL database, first preserve the current
database, then create a clean target and restore the custom dump. Passwords
must be supplied through an ignored environment file or interactive prompt,
never embedded in a command or document:

```powershell
docker cp "_integration_backups/postgres_unification_20260830_183916/postgres/adaptmath_memory_pre_unification.dump" adaptmath_memory_integration_postgres:/tmp/adaptmath_memory_pre_unification.dump
docker compose exec postgres pg_restore --username memory_app --dbname <clean_restore_database> --no-owner --no-privileges /tmp/adaptmath_memory_pre_unification.dump
```

After restoring or selecting legacy files, set the four rollback switches,
restart the stack, and run `status-adaptmath-local.ps1`. Do not drop the current
database or overwrite a legacy file until its replacement has been separately
verified.

## Verification databases retained

- `adaptmath_postgres_unification_verify_20260830_183916`: migrations from zero
- `adaptmath_postgres_unification_verify_20260830_183916_from_lega`: restored
  pre-unification dump upgraded in place and populated from copied legacy state

PostgreSQL truncated the second requested identifier at its 63-byte limit. Both
verification databases are intentionally retained.

## Read-only research query examples

```sql
SELECT selected_arm, count(*) FROM research.selector_decisions GROUP BY selected_arm;
SELECT attempt_id, turn_index, mistake_identification, mistake_location,
       providing_guidance, actionability
FROM research.mrb1_scores ORDER BY attempt_id, turn_index;
SELECT policy_key, total_updates, updated_at FROM research.policy_states;
SELECT target_skill, avg(delta_mastery), avg(reward)
FROM research.experience_records GROUP BY target_skill;
```
