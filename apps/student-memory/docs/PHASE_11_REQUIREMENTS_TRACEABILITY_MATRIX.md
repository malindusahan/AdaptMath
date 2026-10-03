# Phase 11 Requirements Traceability Matrix

## 1. Purpose and Scope

This matrix is the master requirements checklist for the continuing development of the **Student Personalization Memory Layer** after Phase 10.

It connects each client requirement to:

- its responsible component;
- its current implementation status;
- durable repository evidence;
- the remaining gap;
- planned future work;
- the expected interface or output; and
- the validation evidence required before completion.

This document is requirements and design evidence only. Phase 11 Step 1 introduced no production-code, database-schema, API, or machine-learning artifact changes.

## 2. Status Vocabulary

Only the following status values are used:

| Status | Meaning |
|---|---|
| `COMPLETE` | The current system satisfies the stated requirement and has durable evidence. |
| `PARTIAL` | A useful implementation exists, but the final client scope is not fully satisfied. |
| `MISSING` | The required capability is not implemented in the current system. |
| `REPLACE / REFACTOR` | An existing capability must be migrated or substantially redesigned for the final architecture. |
| `EXTERNAL COMPONENT RESPONSIBILITY` | The decision or behavior belongs to another component; Memory only supplies or stores evidence. |

## 3. Component Ownership Boundary

| Component | Owned responsibility | Explicitly not owned by Memory |
|---|---|---|
| Student Personalization Memory Layer | Remember raw and derived evidence; extract canonical skills; build memory summaries; predict the existing learning state; expose auditable context and signals. | Teaching, correctness judgment, instructional planning, repair selection, final mastery/BKT, or learning-path decisions. |
| Tutor | Generate explanations, questions, and teaching responses using Memory context. | Persistent student-memory management. |
| Evaluator | Decide correctness, identified errors, and feedback. | Long-term storage and learning-state inference. |
| Planner | Select instructional plans and sequencing using Memory context. | Persistent evidence ownership. |
| FAPR-LB | Select and execute a repair strategy. | Persistent repair-history ownership. |
| Meta-Agent | Perform mastery/BKT, knowledge-graph, and learning-path reasoning using Memory signals. | Raw evidence storage and canonical memory retrieval. |

## 4. Master Traceability Matrix

### Topic, ontology, and live-question flow

| ID | Client requirement | Source | Owner | Status | Current implementation / evidence | Gap and required future work | Planned phase | Expected API / output | Validation and durable evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| REQ-TOPIC-001 | Map a natural-language student question to a canonical academic skill and return skill identity, confidence, method, alternatives, and review/abstention context. | Final client scope: Topic / Skill Detection | Memory — Topic Extractor | `MISSING` | No topic-extraction runtime module, artifact, endpoint, or test exists. Current API requires caller-supplied `topic` and optional `subtopic`. | Build keyword, TF-IDF/cosine, pretrained MiniLM, fine-tuned MiniLM, hybrid selection, calibration, and safe abstention. | Phases 13–17 | Structured topic-extraction result with canonical skill, confidence, method, alternatives, and `needs_review`. | Dataset and ontology versions; research notebooks; held-out metrics; error analysis; saved artifacts; unit/integration tests. | Open |
| REQ-ONTOLOGY-001 | Maintain a versioned canonical skill ontology with stable skill identifiers, names, aliases, and hierarchy/context metadata. | Final client scope: canonical skill ontology | Memory — Skill Ontology | `MISSING` | Current storage uses free-text `topic` and `subtopic`; no canonical skill table or ontology artifact exists. | Design, source, version, validate, and persist the ontology; define alias and unknown-skill policy. | Phase 13 | Canonical skill records and lookup interface. | Ontology file/database evidence; provenance; coverage report; uniqueness and hierarchy tests. | Open |
| REQ-LIVE-001 | Accept a raw student question, extract its canonical skill automatically, retrieve relevant memory, and return personalization context without manual topic selection. | Final client scope: Live Student Question Flow | Memory | `MISSING` | Current routes in `src/api/memory_routes.py` require explicit topic context and expose no live-text endpoint. | Integrate Topic Extractor, ontology, context retrieval, uncertainty handling, and a versioned API contract. | Phases 17 and 20 | Live-text/question-context endpoint and typed response. | End-to-end tests from raw text to canonical skill and student context; abstention tests; API examples. | Open |

### Raw and derived student memory

| ID | Client requirement | Source | Owner | Status | Current implementation / evidence | Gap and required future work | Planned phase | Expected API / output | Validation and durable evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| REQ-RAW-001 | Preserve a permanent source-of-truth interaction log containing available student, session, problem, utterance, Tutor response, answer, expected answer, correctness, error, behavior, skill, time, and provenance evidence. | Final client scope: Raw Interaction Memory | Memory — Interaction Ingestion | `PARTIAL` | `assessments` and `assessment_interactions` in `src/database/connection.py`; transactional writes in `src/database/assessment_repository.py`; null-preserving behavior tests in `tests/test_assessment_repository.py`. | Add student/session/canonical-skill identities, raw utterance, Tutor response, explicit timestamps/source/provenance, extraction reference, and broader event semantics. Make the log the declared reconstruction source of truth. | Phases 12 and 18 | Append-oriented interaction/event record and retrieval interface. | PostgreSQL constraints; provenance tests; reconstruction tests; transaction and concurrency tests. | Open |
| REQ-STM-001 | Maintain current-session memory: current skill, recent interactions/messages, recent correctness/accuracy, attempts, hints, confusion, and session status. | Final client scope: Short-Term Memory | Memory — Short-Term Memory | `PARTIAL` | Recent historical windows are constructed in `src/features/production_feature_builder.py`; chronological retrieval exists in `src/database/history_repository.py`. | Introduce explicit sessions, recent utterance/message context, confusion signals, session lifecycle/status, and reproducible STM summaries. | Phase 18 | Session-scoped memory/context response. | Deterministic reconstruction tests; session-isolation tests; summary-vs-log consistency report. | Open |
| REQ-LTM-001 | Maintain cross-session learner memory: totals, concepts encountered, overall accuracy, attempts/hints/response history, recurring difficulty, and support evidence. | Final client scope: Long-Term Memory | Memory — Long-Term Memory | `PARTIAL` | Overall historical counts/trends feed the production feature builder and learning-state service; assessment history is persisted. | Add explicit cross-session aggregation, encountered-skill history, durable summary provenance/versioning, and deterministic rebuild from raw records. | Phase 18 | Student-level long-term memory response. | Multi-session aggregation tests; rebuild equivalence; audit provenance. | Open |
| REQ-CONCEPT-001 | Maintain student-specific memory for each canonical skill, including interactions, correct/wrong counts, accuracy, misconceptions, state, and support evidence. | Final client scope: Concept-Based Memory | Memory — Concept Memory | `PARTIAL` | Topic/subtopic-filtered history and current state exist in `src/database/history_repository.py` and `src/database/state_repository.py`; topic isolation is tested. | Replace free-text learning-area identity with canonical skills and add explicit reproducible concept summaries and context retrieval. | Phases 13 and 18 | Canonical-skill-specific memory record and API response. | Student/skill isolation, aggregation, reconstruction, and ontology-FK tests. | Open |
| REQ-MIS-001 | Persist recurring misconceptions with student and learning-context isolation and occurrence history. | Final client scope: Misconception Memory | Memory — Misconception Memory | `COMPLETE` | `student_misconceptions` schema; normalization/deduplication/update logic in `src/database/misconception_repository.py`; coverage in `tests/test_misconception_repository.py`. | Future migration to canonical skill/PostgreSQL must preserve behavior; semantic merging remains a documented future enhancement, not required for current completion. | Phases 12 and 18 (migration only) | Misconception list/counts in relevant memory context. | Existing regression tests plus migration parity tests. | Complete; migration pending under database requirement |
| REQ-STATE-001 | Retain the existing evidence-aware learning-state engine with `NEEDS_SUPPORT`, `DEVELOPING`, `STRONG`, and cold-start `UNAVAILABLE`. | Final client scope: Existing Learning-State Intelligence | Memory — Learning-State Engine | `COMPLETE` | Frozen model/imputer/metadata in `artifacts/`; inference in `src/models/learning_state_model.py`; 12-feature builder; evidence and behavioral coverage; held-out metrics in `artifacts/runtime_manifest.json`; model tests. | Preserve as a submodule of the final Memory system and migrate history access without changing the frozen research interpretation unless separately revalidated. | Phases 12 and 18 (integration only) | Typed state result with evidence level, strength, coverage, and `model_used`. | Artifact hashes; regression tests; PostgreSQL parity tests; documented limitations. | Complete; integration pending under database requirement |

### Context and cross-component integration

| ID | Client requirement | Source | Owner | Status | Current implementation / evidence | Gap and required future work | Planned phase | Expected API / output | Validation and durable evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| REQ-TUTOR-CTX-001 | Supply Tutor-ready personalization evidence: current skill, recent performance, state, mistakes, concept history, and support evidence. | Final client scope: Tutor Context | Memory — Context Builders | `PARTIAL` | Current-state and history endpoints expose state/evidence; misconceptions and interactions are stored internally. | Build one typed Tutor context combining canonical skill, recent evidence, concept memory, misconceptions, and provenance without prescribing pedagogy. | Phase 19 | Tutor-context endpoint/schema. | Contract tests with Tutor fixtures; context completeness and isolation tests. | Open |
| REQ-PLANNER-CTX-001 | Supply Planner-ready session performance, attempt/hint behavior, topic history, and state/evidence information. | Final client scope: Planner Context | Memory — Context Builders | `PARTIAL` | State/evidence and stored behavioral fields exist; no Planner-specific aggregate contract exists. | Add session-aware Planner context and canonical-skill histories. | Phase 19 | Planner-context endpoint/schema. | Contract tests; multi-session and missing-behavior tests. | Open |
| REQ-EVAL-INT-001 | Receive and atomically store Evaluator correctness, errors, feedback, and optional behavior evidence using the established assessment contract. | Final client scope: Evaluator Integration | Memory — Interaction Ingestion | `COMPLETE` | `AssessmentMemoryUpdateRequest`; `POST /memory/update`; `MemoryUpdateService`; atomicity and API tests. Missing behavior remains null. | Extend/migrate the contract for sessions, canonical skills, raw utterances, and PostgreSQL while preserving compatibility. | Phases 12, 17, and 20 (extension only) | Backward-compatible evaluator update response. | Existing 234-test baseline; compatibility, migration, and integration tests. | Complete; extensions tracked separately |
| REQ-FAPR-CTX-001 | Provide FAPR-LB with current skill, recent attempts/hints/correctness/response time, previous repair/outcomes, misconceptions, latest utterance, and state evidence. | Final client scope: FAPR-LB Context | Memory — Context Builders | `MISSING` | No repair-aware context schema, repository, or endpoint exists. | Add repair persistence first, then build an auditable FAPR context response. | Phase 19 | FAPR-context endpoint/schema. | Contract fixtures; context provenance; missing-data and student/skill isolation tests. | Open |
| REQ-REPAIR-001 | Persist repair actions and outcomes so later interactions can use previous repair evidence. | Final client scope: Repair Outcome Memory | Memory — Repair Outcome Memory | `MISSING` | No repair action/outcome table, schema, service, or test exists. | Define repair identity, strategy, outcome, timestamps, linkage, provenance, and append/update policy. | Phases 12 and 19 | Repair-outcome write and history interfaces. | Constraint, transaction, chronology, and FAPR integration tests. | Open |
| REQ-META-SIG-001 | Produce auditable events/signals such as correct, incorrect, clarification request, confusion, and repeated misunderstanding for Meta-Agent consumption. | Final client scope: Meta-Agent Signals | Memory — Meta Signal Generator | `PARTIAL` | Correctness and identified errors are stored; repeated misconceptions can be counted. No unified signal taxonomy or output contract exists. | Define signal taxonomy, derivation/provenance, confusion/clarification capture, and retrieval/API contract. | Phase 19 | Typed, auditable meta-signal stream/context. | Rule tests; provenance tests; contract tests; false-positive review. | Open |

### Persistence, API, UI, reliability, and reproducibility

| ID | Client requirement | Source | Owner | Status | Current implementation / evidence | Gap and required future work | Planned phase | Expected API / output | Validation and durable evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| REQ-DB-PG-001 | Use PostgreSQL as the professional production persistence layer while retaining SQLite only where useful for isolated tests. | Final client scope: Professional Database | Memory — Persistence | `REPLACE / REFACTOR` | Current production architecture uses SQLite via `src/database/connection.py` and repository modules; schema reference is `database/schema.sql`. | Design the final domain model first, then implement PostgreSQL without treating migration as a mechanical table copy. Preserve tested behavior and compatibility. | Phase 12 | PostgreSQL-backed repositories and configuration. | Migration evidence; parity tests; real PostgreSQL integration/concurrency tests; query/index report. | Open |
| REQ-DB-MIG-001 | Provide SQLAlchemy models, Alembic migrations, constraints, indexes, transactions, and reproducible fresh-database setup. | Final client scope: Professional Database | Memory — Persistence | `MISSING` | No SQLAlchemy/Alembic runtime exists in the active project. | Add pinned dependencies, environment-safe configuration, ORM mappings, migrations, transaction pattern, and health/readiness validation. | Phase 12 | Migration-managed database foundation. | Upgrade/downgrade tests; fresh database test; rollback/isolation/concurrency tests. | Open |
| REQ-API-CORE-001 | Preserve health, memory update, current-memory retrieval, and chronological state-history HTTP capabilities with validation and sanitized errors. | Final client scope: Professional API / existing contract | Memory — API | `COMPLETE` | `src/api/app.py`, `src/api/memory_routes.py`; `POST /memory/update`; current/history routes; API and hardening tests. | Preserve or version compatibility during final API expansion. | Phase 20 (compatibility only) | Existing endpoints and response schemas. | Existing API tests plus compatibility suite. | Complete; extension pending under API expansion |
| REQ-API-EXP-001 | Add readiness, student/session/concept/recent-interaction/question/live-text/FAPR/meta-signal/repair APIs. | Final client scope: Professional API | Memory — API | `MISSING` | Only the core health/update/current/history surface exists. | Design versioned typed contracts after underlying domain capabilities exist. | Phase 20 | Expanded professional API surface. | OpenAPI snapshot; endpoint tests; component contract tests; error/security tests. | Open |
| REQ-UI-001 | Provide a professional Next.js/TypeScript Student UI and Research/Memory Inspector; Streamlit is not the final interface. | Final client scope: Professional UI | Memory project — Web Interface | `MISSING` | No frontend project exists in the active repository. | Define UX and API client, implement both interfaces, visualize evidence/uncertainty responsibly, and test accessibility/usability. | Phase 21 | Student UI and research inspector. | UI tests; accessibility checks; screenshots/demo evidence; end-to-end API tests. | Open |
| REQ-REL-001 | Ensure atomic updates, input validation, null-safe behavior, state/history isolation, sanitized failures, and deterministic retrieval. | Final client scope: Security / Reliability | Memory | `COMPLETE` | Shared transaction in `MemoryUpdateService`; repository rollback tests; input hardening; null-preserving behavior; 234-test baseline. | Revalidate equivalently under PostgreSQL and expanded APIs. | Phases 12, 20, and 22 (continued assurance) | Reliable service behavior and documented failure semantics. | Existing tests; future concurrency/load/fault-injection evidence. | Complete for current scope |
| REQ-OBS-001 | Provide health/readiness, structured operational diagnostics, migration/runtime status, and safe auditability. | Final client scope: Professional operation | Memory | `PARTIAL` | `/health`, OpenAPI, runtime manifest, sanitized errors, and development log exist. | Add dependency-aware readiness, structured logging/correlation, migration status, safe metrics, and operational runbook. | Phases 20 and 22 | Health/readiness/operational evidence. | Failure-mode tests; log review; readiness tests; runbook. | Open |
| REQ-SEC-001 | Protect credentials and student data; validate authorization, privacy, retention, and least-privilege operation. | Final client scope: Security / Reliability | Memory plus deployment owner | `PARTIAL` | `.gitignore`, `.env.example`, sanitized errors, input validation, and no embedded credentials are documented. No authentication/authorization or full privacy policy exists. | Threat model; secret configuration; auth boundary; least-privilege DB role; retention/deletion policy; dependency/security review. | Phase 22 | Documented and tested security controls. | Threat model; auth tests; secret scan; dependency audit; privacy/retention evidence. | Open |
| REQ-DOC-001 | Maintain reproducible setup, architecture, API, development decisions, runtime/artifact manifests, and phase evidence. | Final client scope: Documentation / Reproducibility | Memory | `COMPLETE` | `README.md`, `docs/PROJECT_DEVELOPMENT_LOG.md`, runtime manifest, schema reference, pinned requirements, notebooks, and artifact metadata. | Continue updating documentation and this matrix after every completed phase. | Every future phase | Durable documentation/evidence package. | Phase evidence gates; link/path checks; clean Git commits. | Complete and ongoing |
| REQ-TEST-001 | Maintain automated and research validation, including leakage-safe evaluation and reproducible runtime/API tests. | Final client scope: Testing / Evaluation | Memory | `COMPLETE` | 234 automated tests; historical leakage analysis; student-aware held-out model evaluation; artifact validation; API smoke evidence in log/manifest. | Expand with NLP held-out evaluation, PostgreSQL integration/concurrency, new APIs, UI, security, and final system tests. | Every future phase; final Phase 23 | Test and evaluation reports tied to requirements. | CI-style test output, research notebooks/tables, manifests, and final traceability closure. | Complete for current scope and ongoing |

### Explicit external decision ownership

| ID | Client requirement | Source | Owner | Status | Current implementation / evidence | Gap and required future work | Planned phase | Expected API / output | Validation and durable evidence | Final status |
|---|---|---|---|---|---|---|---|---|---|---|
| REQ-EXT-TUTOR-001 | Generate the actual teaching response and choose how to explain or scaffold. | Component ownership boundary | Tutor | `EXTERNAL COMPONENT RESPONSIBILITY` | Memory will supply evidence only. | Define context contract; do not duplicate Tutor logic. | Phase 19 contract work | Tutor consumes Memory context. | Cross-component contract test. | Owned externally |
| REQ-EXT-EVAL-001 | Judge correctness, identify errors, and create evaluation feedback. | Component ownership boundary | Evaluator | `EXTERNAL COMPONENT RESPONSIBILITY` | Existing API accepts Evaluator results and does not calculate correctness. | Preserve boundary while extending ingestion. | Phases 17 and 20 contract work | Evaluator posts evidence to Memory. | Contract tests with evaluator payloads. | Owned externally |
| REQ-EXT-PLANNER-001 | Choose instructional sequencing and learning plans. | Component ownership boundary | Planner | `EXTERNAL COMPONENT RESPONSIBILITY` | Memory stores/retrieves evidence but makes no plan. | Define context contract; do not implement planning policy. | Phase 19 contract work | Planner consumes Memory context. | Cross-component contract test. | Owned externally |
| REQ-EXT-FAPR-001 | Select and execute the repair strategy. | Component ownership boundary | FAPR-LB | `EXTERNAL COMPONENT RESPONSIBILITY` | No repair selection exists in Memory. | Memory must store outcomes and provide context only. | Phase 19 contract work | FAPR-LB consumes context and reports outcome. | Context/writeback contract tests. | Owned externally |
| REQ-EXT-META-001 | Perform BKT/mastery, knowledge-graph updates, and final learning-path decisions. | Component ownership boundary | Meta-Agent | `EXTERNAL COMPONENT RESPONSIBILITY` | Existing learning-state prediction is not represented as BKT/mastery. | Provide auditable evidence signals only; never duplicate Meta-Agent reasoning. | Phase 19 contract work | Meta-Agent consumes signal/context output. | Cross-component contract and semantic-boundary review. | Owned externally |

## 5. Quantitative Audit Summary

| Status | Count |
|---|---:|
| `COMPLETE` | 7 |
| `PARTIAL` | 9 |
| `MISSING` | 8 |
| `REPLACE / REFACTOR` | 1 |
| `EXTERNAL COMPONENT RESPONSIBILITY` | 5 |
| **Total requirements** | **30** |

Completeness checks:

```text
Unmapped client requirements:       0
Requirements without owner:         0
Requirements without status:        0
Open in-scope gaps without a phase:  0
```

## 6. Change-Control Rule

This matrix must be updated at the end of every future phase. A requirement may be marked complete only when its planned implementation and durable validation evidence both exist. Terminal output alone is not sufficient evidence for research-significant work.

## 7. Phase 11 Step 2 Architecture Cross-Reference

The final single-system refactoring direction, target source layout, dependency rules, compatibility policy, and PostgreSQL cutover sequence are frozen in:

```text
docs/PHASE_11_SINGLE_SYSTEM_ARCHITECTURE_CLEANUP_PLAN.md
```

Step 2 changed no requirement status. It converts the Step 1 audit into an implementation-safe architecture plan for Phase 12 and later phases.
