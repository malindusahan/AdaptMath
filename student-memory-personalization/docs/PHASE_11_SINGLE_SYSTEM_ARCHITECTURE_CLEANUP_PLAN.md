# Phase 11 Final Single-System Architecture Cleanup Plan

## 1. Decision Status

This document freezes the refactoring and migration direction for the **Student Personalization Memory Layer** before PostgreSQL and Topic Extractor implementation begins.

It is an architecture plan, not an implementation change.

```text
Production source changes: 0
Database schema changes:   0
API behavior changes:      0
ML artifact changes:       0
Model retraining:          0
```

The existing 234-test system remains the regression boundary throughout migration.

## 2. Architecture Principles

The following rules are frozen.

### 2.1 One final system

The final deliverable is one Student Personalization Memory Layer. Existing validated capabilities are retained as submodules and expanded through controlled refactoring.

### 2.2 Preserve behavior before changing storage

PostgreSQL migration must preserve externally visible behavior before new domain capabilities are added. The current SQLite repositories remain available during the compatibility period.

### 2.3 Application logic must not depend on a database driver

Services, feature logic, learning-state inference, and API contracts must not import `sqlite3`, database file paths, SQLAlchemy sessions, or concrete repository implementations.

### 2.4 Raw interactions are the source of truth

Short-Term, Long-Term, Concept, misconception, learning-state, repair, and signal memory are derived or linked records that must retain provenance to raw evidence and be reproducible where applicable.

### 2.5 Frozen model semantics remain stable

The saved learning-state model, imputer, 12-feature order, cold-start behavior, evidence levels, and behavioral-coverage meanings must not change during persistence migration. Any later model change requires a separate research and evaluation gate.

### 2.6 API compatibility is deliberate

The existing endpoints remain supported during migration:

```text
POST /memory/update
GET  /memory/{student_id}/current
GET  /memory/{student_id}/history
GET  /health
```

Routes stay thin and call application services. Routes must not query concrete repositories directly in the target architecture.

### 2.7 PostgreSQL migrations are authoritative

Alembic migrations become the production schema authority. The current `database/schema.sql` remains a SQLite reference during compatibility work, not the future PostgreSQL authority.

## 3. Current Dependency Findings

The current implementation is well tested but has several storage couplings:

1. `StudentStateService` imports `DatabasePath`, SQLite history functions, and `sqlite3.Connection`.
2. `MemoryUpdateService` opens the SQLite connection and coordinates concrete repository functions itself.
3. `production_feature_builder.py` and `behavioural_coverage.py` import `HistoricalInteraction` from a SQLite repository module.
4. Read routes call `state_repository` directly and pass `DEFAULT_DATABASE_PATH`.
5. Pydantic API schemas, repository return records, and service responses share one large `interaction.py` module.
6. Tests correctly validate behavior but many are coupled to concrete SQLite functions and temporary database paths.

These findings define the required seams; they do not invalidate the current behavior.

## 4. Disposition Vocabulary

| Decision | Meaning |
|---|---|
| `KEEP` | Retain behavior and location unless a later listed split is required. |
| `EXTEND` | Retain and add final-system capability. |
| `REFACTOR` | Preserve behavior while changing boundaries or structure. |
| `MIGRATE` | Port persistence/implementation to the final PostgreSQL architecture. |
| `REPLACE` | Stop using the current implementation in production after parity and cutover. |
| `REMOVE LATER` | Keep during transition; remove only after explicit replacement and regression evidence. |

## 5. Exact Current-File Disposition Map

### 5.1 API layer

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/api/app.py` | `KEEP / EXTEND` | Remain the FastAPI entry point. Add lifespan/configuration, readiness, dependency wiring, and additional routers without moving business logic into the app. |
| `src/api/memory_routes.py` | `REFACTOR / EXTEND` | Preserve current URLs and response behavior. Split into focused routers after service interfaces exist. Replace direct state-repository calls with query services. |
| `src/api/__init__.py` | `KEEP` | Retain package boundary. |

### 5.2 Database layer

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/database/connection.py` | `REPLACE / REMOVE LATER` | Keep for SQLite compatibility and isolated tests during migration. PostgreSQL engine/session/configuration and Alembic replace it in production only after parity and cutover tests. |
| `src/database/assessment_repository.py` | `MIGRATE` | Preserve request validation, duplicate-question protection, null behavior, and write semantics in a SQLAlchemy repository implementing an application-facing protocol. |
| `src/database/history_repository.py` | `REFACTOR / MIGRATE` | Move `HistoricalInteraction` out of the persistence module into a storage-neutral domain type. Implement the history protocol in PostgreSQL while retaining a temporary SQLite adapter. |
| `src/database/misconception_repository.py` | `REFACTOR / MIGRATE` | Keep normalization/deduplication as domain/application logic; migrate persistence, uniqueness, counts, and retrieval to PostgreSQL. |
| `src/database/state_repository.py` | `MIGRATE` | Preserve immutable snapshot, latest-current-state, context-isolation, and chronological history semantics in PostgreSQL. |
| `src/database/__init__.py` | `KEEP` | Retain package; it becomes the persistence/infrastructure package. |
| `database/schema.sql` | `KEEP AS REFERENCE / REMOVE LATER FROM PRODUCTION PATH` | Retain as the current SQLite reference and historical evidence. Alembic migrations become production authority. |
| `database/student_memory.db` | `REMOVE AS PRODUCTION DEPENDENCY` | Remain runtime-generated only for current compatibility/isolated tests; never become required production state. |

### 5.3 Feature layer

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/features/production_feature_builder.py` | `KEEP / REFACTOR` | Freeze the 12 feature definitions, order, window semantics, and missingness rules. Change only its input type from a repository-owned record to a storage-neutral historical-interaction domain object. |
| `src/features/behavioural_coverage.py` | `KEEP / REFACTOR` | Preserve coverage classification exactly; accept the same storage-neutral history type. |
| `src/features/__init__.py` | `KEEP` | Retain package. |

### 5.4 Model layer and artifacts

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/models/learning_state_model.py` | `KEEP` | Keep artifact loading, feature validation/order enforcement, cold-start bypass, inference, and typed response behavior unchanged except dependency-import cleanup if schemas are split. |
| `src/models/__init__.py` | `KEEP` | Retain package. |
| `artifacts/learning_state_model.joblib` | `KEEP` | Immutable frozen runtime artifact unless a separately approved retraining phase occurs. |
| `artifacts/learning_state_imputer.joblib` | `KEEP` | Immutable frozen runtime artifact. |
| `artifacts/learning_state_model_metadata.json` | `KEEP` | Authoritative frozen feature/model metadata. |
| `artifacts/runtime_manifest.json` | `KEEP / EXTEND` | Preserve current checksums and add future component artifacts/version references without rewriting historical values. |
| Future `src/models/topic_extractor.py` | `ADD LATER` | Introduce only after ontology, dataset, baseline, evaluation, and artifact decisions are complete. It must not be added during PostgreSQL foundation work. |

### 5.5 Service layer

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/services/student_state_service.py` | `REFACTOR` | Keep prepared-feature inference. Replace SQLite path/connection parameters and concrete history calls with injected history-query interfaces. |
| `src/services/memory_update_service.py` | `REFACTOR` | Preserve the atomic workflow and response semantics. Inject a unit of work plus repository/service interfaces; transaction begin/commit/rollback belongs to the unit of work. |
| `src/services/__init__.py` | `KEEP` | Retain package. |
| Future context/query services | `ADD LATER` | Add dedicated current-memory, history, session, concept, Tutor, Planner, FAPR, live-question, and Meta-signal services rather than expanding one monolithic service. |

### 5.6 Schema layer

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/schemas/interaction.py` | `KEEP FOR COMPATIBILITY / REFACTOR` | Preserve existing public field names and validation. Split into focused request/response modules only with import-compatibility re-exports and contract tests. Do not use Pydantic API schemas as ORM models. |
| `src/schemas/__init__.py` | `KEEP / EXTEND` | Re-export stable public schemas when modules are split. |

### 5.7 Research/preprocessing utility

| Current path | Decision | Frozen direction |
|---|---|---|
| `src/utils/preprocessing.py` | `KEEP AS RESEARCH EVIDENCE / MOVE LATER` | It is not part of online inference. Retain reproducibility, then move to a clearly named research/script area only after notebook/path compatibility is verified. |
| `src/utils/__init__.py` | `KEEP` | Retain until the preprocessing disposition is completed. |

### 5.8 Tests

| Current tests | Decision | Frozen direction |
|---|---|---|
| All current `tests/test_*.py` | `KEEP` | Preserve the 234-test suite as the compatibility regression boundary. |
| Repository tests | `KEEP / EXTEND` | Continue SQLite adapter tests where useful; add shared repository-contract tests and real PostgreSQL integration tests. |
| Service tests | `REFACTOR / EXTEND` | Test against injected fakes/protocols so application behavior is storage-independent. |
| API tests | `KEEP / EXTEND` | Preserve existing contracts and add dependency overrides for PostgreSQL-backed services. |
| Model/feature tests | `KEEP` | Must remain database-independent after domain-type extraction. |
| Future tests | `ADD` | PostgreSQL migrations/constraints/concurrency; ontology/NLP evaluation; memory reconstruction; context contracts; expanded API; UI; security/load/system tests. |

## 6. Target Source Layout

The following layout is frozen as the intended end state. It will be created incrementally; empty placeholder modules must not be added merely to resemble the tree.

```text
src/
├── api/
│   ├── app.py
│   ├── dependencies.py
│   └── routers/
│       ├── health.py
│       ├── memory.py
│       ├── context.py
│       ├── interactions.py
│       └── repairs.py
│
├── core/
│   ├── config.py
│   ├── exceptions.py
│   └── logging.py
│
├── domain/
│   ├── interactions.py
│   ├── history.py
│   ├── skills.py
│   ├── learning_states.py
│   ├── misconceptions.py
│   ├── repairs.py
│   └── signals.py
│
├── database/
│   ├── base.py
│   ├── session.py
│   ├── unit_of_work.py
│   ├── models/
│   │   ├── student.py
│   │   ├── session.py
│   │   ├── skill.py
│   │   ├── interaction.py
│   │   ├── memory.py
│   │   ├── misconception.py
│   │   ├── learning_state.py
│   │   ├── topic_extraction.py
│   │   └── repair.py
│   └── repositories/
│       ├── assessments.py
│       ├── history.py
│       ├── memory.py
│       ├── misconceptions.py
│       ├── learning_states.py
│       └── repairs.py
│
├── repositories/
│   ├── protocols.py
│   └── unit_of_work.py
│
├── features/
│   ├── production_feature_builder.py
│   └── behavioural_coverage.py
│
├── memory/
│   ├── short_term.py
│   ├── long_term.py
│   ├── concept.py
│   └── aggregation.py
│
├── models/
│   ├── learning_state_model.py
│   └── topic_extractor.py
│
├── services/
│   ├── memory_update_service.py
│   ├── student_state_service.py
│   ├── memory_query_service.py
│   ├── context_service.py
│   ├── topic_extraction_service.py
│   ├── repair_service.py
│   └── signal_service.py
│
└── schemas/
    ├── assessments.py
    ├── memory.py
    ├── context.py
    ├── topics.py
    ├── repairs.py
    └── common.py

alembic/
├── env.py
└── versions/
alembic.ini
```

### Why repository protocols are outside `database/`

Application services depend on abstract repository and unit-of-work protocols in `src/repositories/`. SQLAlchemy implementations live under `src/database/repositories/`. This prevents the application layer from depending on a persistence technology.

### Why Alembic is top-level

Alembic's conventional project files remain at the repository root. They import SQLAlchemy metadata from `src/database/base.py` and model modules. They are not mixed into request handling or application services.

## 7. Target Dependency Direction

Allowed direction:

```text
API routers
    ↓
Application services
    ↓
Repository / Unit-of-Work protocols
    ↓
SQLAlchemy PostgreSQL adapters

Application services
    ↓
Domain records + feature logic + model interfaces
```

Prohibited direction:

```text
Feature logic        → concrete database repositories
Model inference      → database sessions
API routes           → SQLAlchemy queries
Domain objects       → Pydantic API schemas
Services             → sqlite3 or psycopg driver objects
ORM models           → FastAPI request/response behavior
```

## 8. Frozen Application Interfaces

Exact Python protocol signatures will be finalized during Phase 12 design, but these capability boundaries are frozen:

```text
HistoryReader
→ get student history
→ get canonical-skill history
→ support chronological cutoff

AssessmentWriter
→ store one evaluated assessment and its raw interactions

MisconceptionRepository
→ upsert normalized misconception evidence
→ retrieve isolated misconception history

LearningStateRepository
→ append immutable snapshot
→ replace/upsert current state
→ retrieve current and chronological history

MemoryUnitOfWork
→ expose repositories
→ commit
→ rollback
→ close via context-manager semantics
```

The storage-neutral `HistoricalInteraction` domain record must include all fields needed by the frozen feature and coverage logic. PostgreSQL-specific ORM instances must be converted at the repository boundary.

## 9. Migration and Cutover Sequence

The order below is mandatory unless changed through a documented architecture decision.

### Stage 1 — Establish storage-neutral boundaries

1. Move history records to `src/domain/`.
2. Define repository/unit-of-work protocols.
3. Adapt current SQLite functions behind compatibility adapters.
4. Inject interfaces into services.
5. Move API read behavior behind query services.
6. Keep all 234 tests passing and add interface-contract tests.

### Stage 2 — Add PostgreSQL foundation

1. Add pinned SQLAlchemy, psycopg, and Alembic dependencies.
2. Add environment-driven configuration with no committed secrets.
3. Create engine/session/unit-of-work infrastructure.
4. Create final domain-oriented ORM models and Alembic migrations.
5. Validate migrations against a real PostgreSQL instance.

### Stage 3 — Implement and prove repository parity

1. Implement PostgreSQL repositories.
2. Run the same contract suite against SQLite compatibility and PostgreSQL adapters where semantics overlap.
3. Validate null behavior, constraints, ordering, isolation, rollback, and atomicity.
4. Add concurrent-update and query/index tests.

### Stage 4 — Cut over production wiring

1. Select PostgreSQL adapters through application configuration/dependency injection.
2. Run all compatibility, API, and end-to-end tests.
3. Validate the existing external API against PostgreSQL.
4. Make Alembic the production schema authority.

### Stage 5 — Retire SQLite production coupling

Only after successful cutover:

- remove database-path parameters from application/public service APIs;
- stop production initialization through `SCHEMA_SQL`;
- remove direct SQLite imports from services/features/API;
- retain SQLite only for explicitly approved isolated tests;
- remove obsolete compatibility code in a separately reviewed cleanup commit.

## 10. Backward-Compatibility Rules

During Phase 12:

1. Existing endpoint paths and HTTP meanings remain unchanged.
2. Existing Pydantic request and response fields remain accepted/emitted.
3. Missing attempt, hint, and response-time values remain null/unavailable, never fabricated as zero.
4. The completed assessment remains included when calculating the state returned for the next activity.
5. Current-state retrieval remains read-only and does not rerun the model.
6. History remains chronological and supports context isolation.
7. State snapshots remain append-only; current memory remains the latest materialized state.
8. Learning-state artifacts and feature order remain checksum/verifiably unchanged.

Any deliberate compatibility break requires a versioned API decision, migration guidance, tests, and a development-log entry.

## 11. Testing Strategy During Refactoring

The test suite will be organized conceptually into:

```text
tests/unit/
→ domain, feature, model, schema, and service tests with fakes

tests/contracts/
→ repository and component interface behavior

tests/integration/sqlite/
→ temporary compatibility adapter tests where retained

tests/integration/postgresql/
→ real migration, constraint, transaction, query, and concurrency tests

tests/api/
→ compatibility and expanded API tests

tests/system/
→ full interaction-to-memory flows
```

Existing tests will not be moved merely for cosmetic structure. Movement happens only when the associated boundary is implemented, and must preserve test history and behavior.

## 12. Explicit Non-Actions in Step 2

Phase 11 Step 2 does not:

- install PostgreSQL dependencies;
- add SQLAlchemy or Alembic code;
- create ORM models or migrations;
- create the Topic Extractor;
- modify endpoint behavior;
- move current source files;
- remove SQLite code;
- modify model artifacts; or
- reorganize tests.

## 13. Architecture Freeze Result

The refactoring direction is now explicit:

```text
Learning-state artifacts/model       KEEP
12-feature semantics                 KEEP
Behavioral coverage semantics        KEEP
Feature inputs                       REFACTOR to domain history type
FastAPI app                           KEEP / EXTEND
Existing endpoints                   KEEP / ADAPT behind services
Pydantic validation                  KEEP / SPLIT compatibly
Atomic update concept                KEEP / REFACTOR to unit of work
Misconception logic                  KEEP / persistence MIGRATE
SQLite repositories                  MIGRATE, then REMOVE from production
SQLite runtime database              REMOVE as production dependency
PostgreSQL + Alembic                 ADD in Phase 12
Topic Extractor                      ADD only after research foundation
Current tests                        KEEP as regression boundary
```

Phase 12 may begin PostgreSQL work only under these boundaries.

